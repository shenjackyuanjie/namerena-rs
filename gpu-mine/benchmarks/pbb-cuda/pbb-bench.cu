#include "pbb-run.hpp"
#include <cuda_runtime.h>
#include "pbb_device_scoring.hpp" // 从同一份 PBB 源生成的 CUDA 评分。
#include "pbb_device_prefilter.hpp"

static_assert(pbb_device::M_N == families);
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

struct Device {
    Counts *counts = nullptr, *host_counts = nullptr;
    U* indices = nullptr;
    bool split = false;
    int score_grid = 0;
    static constexpr int score_block = 64;
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
    void init(const Launch& input, const Options& opts) {
        cudaDeviceProp property{};
        int device = 0;
        CHECK(cudaGetDevice(&device));
        CHECK(cudaGetDeviceProperties(&property, device));
        std::cerr << "GPU " << property.name << '\n';
        const size_t capacity = opts.batch;
        split = opts.cuda_path == "split";
        // grid-stride 后核至少铺满全部 SM，多派发几波缓解 PTT 造成的长尾。
        score_grid = 8 * property.multiProcessorCount;
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
    Batch run(Id first, U n, U block, bool capture, bool = false) {
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
            evaluate_selected<<<score_grid, score_block, 0, stream>>>(first, capture, counts, records, indices);
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

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) try {
    std::vector<std::string> utf8;
    std::vector<const char*> args;
    utf8.reserve(argc);
    args.reserve(argc);
    for (int i = 0; i < argc; ++i) utf8.push_back(fs::path(argv[i]).u8string());
    for (const auto& value : utf8) args.push_back(value.c_str());
    return pbb_main<Device>(argc, args.data(), "cuda");
} catch (const std::exception& e) {
    std::cerr << u8"CUDA \u53c2\u6570\uff1a" << e.what() << '\n';
    return 1;
}
#else
int main(int argc, char** argv) { return pbb_main<Device>(argc, argv, "cuda"); }
#endif
