#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include "pbb-bench-types.hpp"
#include <scoring_1124.hpp>        // 本机原版 PBB，CPU 金标准。
#include "pbb_device_scoring.hpp" // 由同一批本地 PBB 源生成的 CUDA 评分。
#include "pbb-config.hpp"
#include "pbb_device_prefilter.hpp"

using Clock = std::chrono::steady_clock;
static_assert(M_N == families && pbb_device::M_N == families);
__constant__ Launch device_input;

__device__ void score_one(Id first, U i, bool capture, Counts* counts, Record* records) {
    char name[name_limit + 1];
    make_name(device_input, first + i, name);
    pbb_device::Name state;
    init_candidate(state, device_input);
    const auto plan = make_plan<pbb_device::ScorePlan>(device_input.rules);
    const auto score = pbb_device::score_full(name, device_input.name_len, state, &plan);
    const auto flags = selection(device_input.rules, score);
    if (score.flag) atomicAdd(&counts->passed, 1u);
    if (flags.keep) atomicAdd(&counts->keep, 1u);
    if (flags.blue) atomicAdd(&counts->blue, 1u);
    if (flags.ptt) atomicAdd(&counts->ptt, 1u);
    if (capture || flags.keep || flags.blue) {
        const U slot = capture ? i : atomicAdd(&counts->written, 1u);
        records[slot] = wire(i, flags, score);
    }
}

__global__ void evaluate(Id first, U n, bool capture, Counts* counts, Record* records) {
    const U i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) score_one(first, i, capture, counts, records);
}

// 先走同一原版一级门；后核仍运行完整评分，不缓存或改写门后的语义。
template<int Block>
__global__ void select_ids(Id first, U n, bool capture, Counts* counts, Record* records, U* indices) {
    __shared__ unsigned words[64 * Block];
    const U i = blockIdx.x * Block + threadIdx.x;
    if (i >= n) return;
    if (pbb_device::prefilter<Block>(device_input, first + i, words + threadIdx.x)) {
        const U slot = atomicAdd(&counts->eligible, 1u);
        indices[slot] = i;
    } else if (capture) {
        // 原 score_full 在一级门失败处返回全零结果。
        Record rejected{};
        rejected.index = i;
        records[i] = rejected;
    }
}

__global__ void evaluate_selected(Id first, bool capture, Counts* counts, Record* records, const U* indices) {
    const U i = blockIdx.x * blockDim.x + threadIdx.x;
    const U n = counts->eligible;
    for (U slot = i; slot < n; slot += gridDim.x * blockDim.x) score_one(first, indices[slot], capture, counts, records);
}

static void check(const char* operation, cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#define CHECK(expr) check(#expr, (expr))

struct Batch { Counts counts{}; std::vector<Record> records; double kernel_s = 0; };
struct Device {
    Counts *counts = nullptr, *host_counts = nullptr;
    U* indices = nullptr;
    bool split = false;
    Record *records = nullptr, *host_records = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    Device() = default;
    Device(const Device&) = delete;
    ~Device() {
        if (stream) cudaStreamSynchronize(stream);
        if (counts) cudaFree(counts);
        if (records) cudaFree(records);
        if (indices) cudaFree(indices);
        if (host_counts) cudaFreeHost(host_counts);
        if (host_records) cudaFreeHost(host_records);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
    }
    void init(const Launch& input, size_t capacity, bool use_split) {
        split = use_split;
        if (split) CHECK(cudaMalloc(&indices, capacity * sizeof(*indices)));
        CHECK(cudaMemcpyToSymbol(device_input, &input, sizeof(input)));
        CHECK(cudaMalloc(&counts, sizeof(*counts)));
        CHECK(cudaMalloc(&records, capacity * sizeof(*records)));
        CHECK(cudaMallocHost(&host_counts, sizeof(*host_counts)));
        CHECK(cudaMallocHost(&host_records, capacity * sizeof(*host_records)));
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cudaEventCreate(&begin));
        CHECK(cudaEventCreate(&end));
    }
    Batch run(Id first, U n, U block, bool capture) {
        CHECK(cudaMemsetAsync(counts, 0, sizeof(*counts), stream));
        CHECK(cudaEventRecord(begin, stream));
        if (split) {
            #define FRONT(b) select_ids<b><<<(n + b - 1) / b, b, 0, stream>>>(first, n, capture, counts, records, indices)
            switch (block) {
                case 32: FRONT(32); break;
                case 64: FRONT(64); break;
                case 128: FRONT(128); break;
                default: throw std::runtime_error("split path supports block 32/64/128");
            }
            #undef FRONT
            CHECK(cudaGetLastError());
            evaluate_selected<<<16, 32, 0, stream>>>(first, capture, counts, records, indices);
        } else {
            evaluate<<<(n + block - 1) / block, block, 0, stream>>>(first, n, capture, counts, records);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(end, stream));
        CHECK(cudaMemcpyAsync(host_counts, counts, sizeof(*counts), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        Batch result;
        result.counts = *host_counts;
        if (capture) result.counts.written = n;
        if (result.counts.eligible > n || (split && result.counts.passed > result.counts.eligible) || result.counts.written > n || result.counts.passed > n || result.counts.keep > result.counts.passed
            || result.counts.blue > result.counts.passed || result.counts.ptt > result.counts.passed)
            throw std::runtime_error("invalid GPU count");
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, begin, end));
        result.kernel_s = double(ms) / 1000;
        if (result.counts.written) {
            CHECK(cudaMemcpyAsync(host_records, records, result.counts.written * sizeof(*records), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            result.records.assign(host_records, host_records + result.counts.written);
            const auto less = [](const auto& a, const auto& b) { return a.index < b.index; };
            if (!capture) std::sort(result.records.begin(), result.records.end(), less);
            if (result.records.back().index >= n || std::adjacent_find(result.records.begin(), result.records.end(),
                    [](const auto& a, const auto& b) { return a.index == b.index; }) != result.records.end())
                throw std::runtime_error("invalid GPU IDs");
        }
        return result;
    }
};

static Batch cpu(const Launch& input, Id first, U n, U threads, bool capture) {
    threads = std::min(threads, n);
    std::vector<Batch> parts(threads);
    std::vector<std::thread> workers;
    const auto plan = make_plan<ScorePlan>(input.rules);
    for (U t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            auto& local = parts[t];
            Name state{};
            init_candidate(state, input);
            char name[name_limit + 1];
            for (U i = U(Id(n) * t / threads); i < U(Id(n) * (t + 1) / threads); ++i) {
                make_name(input, first + i, name);
                const auto score = score_full(name, input.name_len, state, &plan);
                const auto flags = selection(input.rules, score);
                if (flags.keep != score_output_keep(plan, score, input.rules.any_gate != 0)) std::terminate();
                local.counts.passed += score.flag;
                local.counts.keep += flags.keep;
                local.counts.blue += flags.blue;
                local.counts.ptt += flags.ptt;
                if (capture || flags.keep || flags.blue) local.records.push_back(wire(i, flags, score));
            }
        });
    }
    for (auto& worker : workers) worker.join();
    Batch out;
    for (const auto& part : parts) {
        out.counts.passed += part.counts.passed;
        out.counts.keep += part.counts.keep;
        out.counts.blue += part.counts.blue;
        out.counts.ptt += part.counts.ptt;
        out.records.insert(out.records.end(), part.records.begin(), part.records.end());
    }
    out.counts.written = U(out.records.size());
    return out;
}

static bool equal(const Record& a, const Record& b) {
    const auto near = [](double x, double y) { return std::isfinite(x) && std::isfinite(y) && std::abs(x - y) <= 1e-11 * (1 + std::abs(y)); };
    return a.index == b.index && a.passed == b.passed && a.keep == b.keep && a.blue == b.blue && a.ptt == b.ptt
        && a.xp == b.xp && a.xd == b.xd && a.sum == b.sum && a.tag == b.tag && a.shadow == b.shadow
        && a.illusion == b.illusion && a.pp_ref == b.pp_ref && a.qp_ref == b.qp_ref
        && std::equal(std::begin(a.scores), std::end(a.scores), b.scores)
        && std::equal(std::begin(a.potential), std::end(a.potential), b.potential)
        && std::equal(std::begin(a.props), std::end(a.props), b.props)
        && std::equal(std::begin(a.ids), std::end(a.ids), b.ids)
        && std::equal(std::begin(a.freqs), std::end(a.freqs), b.freqs)
        && near(a.effective, b.effective) && near(a.strength, b.strength) && near(a.deduction, b.deduction);
}

static void fields(std::ostream& file, const Record& r) {
    file << ',' << r.keep << ',' << r.blue << ',' << r.ptt << ',' << r.sum;
    for (int v : r.scores) file << ',' << v;
    for (int v : r.potential) file << ',' << v;
    for (int v : r.props) file << ',' << v;
    for (int v : r.ids) file << ',' << v;
    for (int v : r.freqs) file << ',' << v;
}
static void csv_field(std::ostream& file, std::string_view s) {
    file << '"';
    for (char c : s) { if (c == '"') file << '"'; file << c; }
    file << '"';
}

static void validate(const Options& opts, const Launch& input, Device& gpu) {
    Id id = opts.start;
    U remaining = opts.validate;
    if (remaining > std::numeric_limits<Id>::max() - id) throw std::runtime_error("validation range overflow");
    Id passed = 0, keep = 0, blue = 0, ptt = 0;
    while (remaining) {
        const U n = std::min(remaining, opts.batch);
        const auto expected = cpu(input, id, n, opts.threads, true);
        const auto actual = gpu.run(id, n, opts.block, true);
        if (actual.records.size() != n) throw std::runtime_error("validation output count mismatch at " + std::to_string(id));
        for (U i = 0; i < n; ++i) {
            if (!equal(actual.records[i], expected.records[i])) {
                std::cerr << "CPU passed=" << expected.records[i].passed; fields(std::cerr, expected.records[i]);
                std::cerr << "\nGPU passed=" << actual.records[i].passed; fields(std::cerr, actual.records[i]); std::cerr << '\n';
                throw std::runtime_error("validation field mismatch: ID " + std::to_string(id + i));
            }
        }
        const auto same_counts = [&](const Counts& c) {
            return c.passed == expected.counts.passed && c.keep == expected.counts.keep
                && c.blue == expected.counts.blue && c.ptt == expected.counts.ptt;
        };
        if (!same_counts(actual.counts)) throw std::runtime_error("validation stage count mismatch at " + std::to_string(id));
        // 还要实际调度性能路径，验证非捕获时只写两个集合的并集。
        const auto emitted = gpu.run(id, n, opts.block, false);
        if (!same_counts(emitted.counts)) throw std::runtime_error("filtered stage count mismatch at " + std::to_string(id));
        size_t slot = 0;
        for (const auto& r : expected.records) if (r.keep || r.blue) {
            if (slot >= emitted.records.size() || !equal(emitted.records[slot], r))
                throw std::runtime_error("filtered output mismatch: ID " + std::to_string(id + r.index));
            ++slot;
        }
        if (slot != emitted.records.size()) throw std::runtime_error("extra filtered records at " + std::to_string(id));
        passed += actual.counts.passed; keep += actual.counts.keep; blue += actual.counts.blue; ptt += actual.counts.ptt;
        remaining -= n; id += n;
    }
    std::cout << "validated=" << opts.validate << " passed=" << passed << " keep=" << keep << " blue=" << blue << " nonzero_ptt=" << ptt << '\n';
}

template<class Runner>
static void measure(const Options& opt, const Launch& input, std::string label, Runner run, std::ostream& metrics) {
    const U warmup = U(std::min<Id>(opt.batch, std::numeric_limits<Id>::max() - opt.start));
    run(opt.start, warmup);
    const auto start = Clock::now();
    std::ofstream out(opt.out / (label + ".csv"), std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << "id,full_name,keep,blue,ptt,sum";
    for (const char* s : family_names) out << ',' << s;
    for (const char* s : family_names) out << ",ptt_" << s;
    for (int i = 0; i < 8; ++i) out << ",prop" << i;
    for (int i = 0; i < 16; ++i) out << ",skill" << i;
    for (int i = 0; i < 16; ++i) out << ",freq" << i;
    out << '\n';
    Id id = opt.start, passed = 0, keep = 0, blue = 0, ptt = 0, eligible = 0;
    double kernel = 0;
    while (Clock::now() - start < std::chrono::seconds(opt.seconds)) {
        const U n = U(std::min<Id>(opt.batch, std::numeric_limits<Id>::max() - id));
        if (!n) throw std::runtime_error("ID interval exhausted before deadline");
        const auto b = run(id, n);
        for (const auto& r : b.records) {
            char name[name_limit + 1];
            make_name(input, id + r.index, name);
            out << id + r.index << ',';
            csv_field(out, std::string(name) + '@' + opt.team);
            fields(out, r);
            out << '\n';
        }
        passed += b.counts.passed; keep += b.counts.keep; blue += b.counts.blue; ptt += b.counts.ptt; eligible += b.counts.eligible;
        kernel += b.kernel_s;
        id += n;
    }
    out.flush();
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    const Id n = id - opt.start;
    const double rate = double(n) / elapsed;
    metrics << std::fixed << std::setprecision(9) << label << ',' << n << ',' << passed << ',' << keep << ',' << blue << ',' << ptt
        << ',' << elapsed << ',' << kernel << ',' << rate << ',' << rate * 86400 / 1e12 << ',' << rate * 604800 / 1e12 << ',' << eligible << '\n';
    metrics.flush();
    std::cout << label << " elapsed=" << elapsed << " count=" << n << " scored=" << passed << " keep=" << keep
              << " blue=" << blue << " nonzero_ptt=" << ptt << " names/s=" << rate << std::endl;
}

int main(int argc, char** argv) try {
    auto opts = arguments(argc, argv);
    const auto plan = rules(opts);
    const auto input = prepare(opts, plan);
    if (!fs::create_directory(opts.out)) throw std::runtime_error("output directory already exists");
    Device gpu;
    if (opts.validate || opts.backend != "cpu") {
        cudaDeviceProp dev{};
        int id = 0;
        CHECK(cudaGetDevice(&id));
        CHECK(cudaGetDeviceProperties(&dev, id));
        std::cout << "GPU " << dev.name << '\n';
        gpu.init(input, opts.batch, opts.cuda_path == "split");
    }
    if (opts.validate) { validate(opts, input, gpu); return 0; }
    std::ofstream metrics(opts.out / "metrics.csv", std::ios::binary);
    metrics.exceptions(std::ios::badbit | std::ios::failbit);
    metrics << "backend,candidates,passed,normal,blue,nonzero_ptt,e2e_s,kernel_s,names_s,T_per_day,T_per_week,eligible\n";
    if (opts.backend != "cuda") measure(opts, input, "cpu", [&](Id first, U n) { return cpu(input, first, n, opts.threads, false); }, metrics);
    if (opts.backend != "cpu") measure(opts, input, "cuda", [&](Id first, U n) { return gpu.run(first, n, opts.block, false); }, metrics);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "PBB bench: " << e.what() << '\n';
    return 1;
}
