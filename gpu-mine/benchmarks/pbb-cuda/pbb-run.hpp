#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "pbb-bench-types.hpp"
#include <scoring_1124.hpp> // 本机原版 PBB 标量实现。
#include "pbb-config.hpp"

using Clock = std::chrono::steady_clock;
static_assert(M_N == families);

struct Batch {
    Counts counts{};
    std::vector<Record> records;
    double kernel_s = 0;
    // 仅在 WGSL 验证时保留捕获档位和正式压缩路径的下标。
    std::vector<U> gate_levels, gate_indices;
};

static std::atomic<bool> stopped{false};
static_assert(std::atomic<bool>::is_always_lock_free);
#ifdef _WIN32
static BOOL WINAPI stop_handler(DWORD code) {
    if (code != CTRL_C_EVENT && code != CTRL_BREAK_EVENT) return FALSE;
    stopped.store(true, std::memory_order_relaxed);
    return TRUE;
}
static void install_stop_handler() {
    if (!SetConsoleCtrlHandler(stop_handler, TRUE)) throw std::runtime_error(u8"\u65e0\u6cd5\u6ce8\u518c Ctrl+C \u5904\u7406\u5668");
}
#else
static void stop_handler(int) { stopped.store(true, std::memory_order_relaxed); }
static void install_stop_handler() {
    if (std::signal(SIGINT, stop_handler) == SIG_ERR || std::signal(SIGTERM, stop_handler) == SIG_ERR)
        throw std::runtime_error(u8"\u65e0\u6cd5\u6ce8\u518c\u4e2d\u65ad\u5904\u7406\u5668");
}
#endif

// selected 按批内下标升序排列。捕获时给未过 Q 门的候选保留原版全零记录。
static Batch cpu(const Launch& input, Id first, U n, U threads, bool capture,
                 const std::vector<U>* selected = nullptr, bool capture_gate = false) {
    const U jobs = selected ? U(selected->size()) : n;
    threads = std::min(threads, jobs);
    Batch out;
    if (capture_gate) out.gate_levels.resize(n);
    if (selected) {
        out.counts.eligible = jobs;
        if (capture) {
            out.records.resize(n);
            for (U i = 0; i < n; ++i) out.records[i].index = i;
        }
    }
    const auto plan = make_plan<ScorePlan>(input.rules);
    std::vector<Batch> parts(threads);
    std::vector<std::exception_ptr> errors(threads);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    const auto evaluate = [&](U t) {
        try {
            auto& local = parts[t];
            Name state{};
            init_candidate(state, input);
            char name[name_limit + 1];
            for (U pos = U(Id(jobs) * t / threads); pos < U(Id(jobs) * (t + 1) / threads); ++pos) {
                const U i = selected ? (*selected)[pos] : pos;
                make_name(input, first + i, name);
                if (capture_gate) {
                    int props[7];
                    state.load_name(name, input.name_len);
                    out.gate_levels[i] = U(gate_level_of(state.name_base, props));
                }
                const auto score = score_full(name, input.name_len, state, &plan);
                const auto flags = selection(input.rules, score);
                if (flags.keep != score_output_keep(plan, score, input.rules.any_gate != 0))
                    throw std::runtime_error(u8"\u4e3b\u6536\u96c6\u5224\u5b9a\u4e0e PBB \u539f\u7248\u4e0d\u540c");
                local.counts.passed += score.flag;
                local.counts.keep += flags.keep;
                local.counts.blue += flags.blue;
                local.counts.ptt += flags.ptt;
                if (selected && capture) out.records[i] = wire(i, flags, score);
                else if (capture || flags.keep || flags.blue) local.records.push_back(wire(i, flags, score));
            }
        } catch (...) { errors[t] = std::current_exception(); }
    };
    try {
        for (U t = 0; t < threads; ++t) workers.emplace_back(evaluate, t);
    } catch (...) {
        for (auto& worker : workers) worker.join();
        throw;
    }
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
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
    const auto close_enough = [](double x, double y) { return std::isfinite(x) && std::isfinite(y) && std::abs(x - y) <= 1e-11 * (1 + std::abs(y)); };
    return a.index == b.index && a.passed == b.passed && a.keep == b.keep && a.blue == b.blue && a.ptt == b.ptt
        && a.xp == b.xp && a.xd == b.xd && a.sum == b.sum && a.tag == b.tag && a.shadow == b.shadow
        && a.illusion == b.illusion && a.pp_ref == b.pp_ref && a.qp_ref == b.qp_ref
        && std::equal(std::begin(a.scores), std::end(a.scores), b.scores)
        && std::equal(std::begin(a.potential), std::end(a.potential), b.potential)
        && std::equal(std::begin(a.props), std::end(a.props), b.props)
        && std::equal(std::begin(a.ids), std::end(a.ids), b.ids)
        && std::equal(std::begin(a.freqs), std::end(a.freqs), b.freqs)
        && close_enough(a.effective, b.effective) && close_enough(a.strength, b.strength) && close_enough(a.deduction, b.deduction);
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

template<class Device>
static void validate(const Options& opts, const Launch& input, Device& gpu) {
    Id id = opts.start;
    U remaining = opts.validate;
    if (remaining > std::numeric_limits<Id>::max() - id) throw std::runtime_error("validation range overflow");
    Id passed = 0, keep = 0, blue = 0, ptt = 0;
    while (remaining) {
        const U n = std::min(remaining, opts.batch);
        if (stopped.load(std::memory_order_relaxed)) return;
        const auto actual = gpu.run(id, n, opts.block, true);
        const bool verify_gate_indices = !actual.gate_levels.empty();
        const bool check_gate_count = opts.cuda_path == "split";
        const auto expected = cpu(input, id, n, opts.threads, true, nullptr, check_gate_count);
        if (check_gate_count && actual.counts.eligible != std::count_if(expected.gate_levels.begin(), expected.gate_levels.end(),
                                                                       [](U level) { return level != 0; }))
            throw std::runtime_error("Q gate count mismatch at " + std::to_string(id));
        if (verify_gate_indices && actual.gate_levels != expected.gate_levels) {
            const auto mismatch = std::mismatch(actual.gate_levels.begin(), actual.gate_levels.end(), expected.gate_levels.begin());
            throw std::runtime_error(u8"Q \u95e8\u6863\u4f4d\u4e0d\u540c\uff0cID " + std::to_string(id + (mismatch.first - actual.gate_levels.begin())));
        }
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
        const auto emitted = gpu.run(id, n, opts.block, false, verify_gate_indices);
        if (!same_counts(emitted.counts) || emitted.counts.eligible != actual.counts.eligible)
            throw std::runtime_error("filtered stage count mismatch at " + std::to_string(id));
        if (verify_gate_indices) {
            size_t slot = 0;
            for (U i = 0; i < n; ++i) {
                if (!expected.gate_levels[i]) continue;
                if (slot == emitted.gate_indices.size() || emitted.gate_indices[slot] != i)
                    throw std::runtime_error(u8"Q \u95e8\u538b\u7f29\u4e0b\u6807\u4e0d\u540c ID " + std::to_string(id + i));
                ++slot;
            }
            if (slot != emitted.gate_indices.size()) throw std::runtime_error("extra Q gate indices at " + std::to_string(id));
        }
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

static void write_header(std::ostream& out) {
    out << "id,full_name,keep,blue,ptt,sum";
    for (const char* s : family_names) out << ',' << s;
    for (const char* s : family_names) out << ",ptt_" << s;
    for (int i = 0; i < 8; ++i) out << ",prop" << i;
    for (int i = 0; i < 16; ++i) out << ",skill" << i;
    for (int i = 0; i < 16; ++i) out << ",freq" << i;
    out << '\n';
}

static void write_batch(std::ostream& out, const Options& opt, const Launch& input, Id first, const Batch& batch) {
    for (const auto& record : batch.records) {
        const Id id = first + record.index;
        char name[name_limit + 1];
        make_name(input, id, name);
        out << id << ',';
        csv_field(out, std::string(name) + '@' + opt.team);
        fields(out, record);
        out << '\n';
    }
}

struct Progress {
    Id next, passed = 0, keep = 0, blue = 0, ptt = 0, eligible = 0;
    double kernel = 0;

    void add(const Batch& batch, U n) {
        passed += batch.counts.passed;
        keep += batch.counts.keep;
        blue += batch.counts.blue;
        ptt += batch.counts.ptt;
        eligible += batch.counts.eligible;
        kernel += batch.kernel_s;
        next += n;
    }
};

template<class Runner>
static void mine(const Options& opt, const Launch& input, const std::string& label, Runner run) {
    const auto start = Clock::now();
    std::ofstream out(opt.out / "results.csv", std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    write_header(out);
    out.flush();
    Progress progress{opt.start};
    auto last_report = start;
    const auto report = [&](const char* status) {
        const auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        const Id n = progress.next - opt.start;
        const double rate = double(n) / std::max(elapsed, 1e-9);
        std::cerr << std::fixed << std::setprecision(2) << label << ' ' << status
                  << " count=" << n << " keep=" << progress.keep << " blue=" << progress.blue
                  << " nonzero_ptt=" << progress.ptt << " next_id=" << progress.next
                  << " names/s=" << rate << " T/day=" << rate * 86400 / 1e12
                  << " T/week=" << rate * 604800 / 1e12 << '\n';
    };
    while (progress.next < *opt.end && !stopped.load(std::memory_order_relaxed)) {
        const U n = U(std::min<Id>(opt.batch, *opt.end - progress.next));
        const auto batch = run(progress.next, n);
        write_batch(out, opt, input, progress.next, batch);
        progress.add(batch, n);
        if (Clock::now() - last_report >= std::chrono::seconds(opt.report)) {
            out.flush();
            report(u8"\u8fd0\u884c\u4e2d");
            last_report = Clock::now();
        }
    }
    out.flush();
    report(progress.next == *opt.end ? u8"\u5df2\u5b8c\u6210" : u8"\u5df2\u505c\u6b62");
}

template<class Runner>
static void measure(const Options& opt, const Launch& input, const std::string& label, Runner run, std::ostream& metrics) {
    const U warmup = U(std::min<Id>(opt.batch, std::numeric_limits<Id>::max() - opt.start));
    run(opt.start, warmup);
    const auto start = Clock::now();
    std::ofstream out(opt.out / (label + ".csv"), std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    write_header(out);
    Progress progress{opt.start};
    while (!stopped.load(std::memory_order_relaxed) && Clock::now() - start < std::chrono::seconds(opt.seconds)) {
        const U n = U(std::min<Id>(opt.batch, std::numeric_limits<Id>::max() - progress.next));
        if (!n) throw std::runtime_error(u8"\u8ba1\u65f6\u7ed3\u675f\u524d\u5df2\u7528\u5b8c ID \u533a\u95f4");
        const auto batch = run(progress.next, n);
        write_batch(out, opt, input, progress.next, batch);
        progress.add(batch, n);
    }
    out.flush();
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    const Id n = progress.next - opt.start;
    const double rate = double(n) / elapsed;
    metrics << std::fixed << std::setprecision(9) << label << ',' << n << ',' << progress.passed << ','
            << progress.keep << ',' << progress.blue << ',' << progress.ptt << ',' << elapsed << ','
            << progress.kernel << ',' << rate << ',' << rate * 86400 / 1e12 << ',' << rate * 604800 / 1e12
            << ',' << progress.eligible << '\n';
    metrics.flush();
    std::cout << label << " elapsed=" << elapsed << " count=" << n << " scored=" << progress.passed
              << " keep=" << progress.keep << " blue=" << progress.blue << " nonzero_ptt=" << progress.ptt
              << " names/s=" << rate << std::endl;
}

template<class Device>
static int pbb_main(int argc, const char* const* argv, const std::string& device) try {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << "PBB " << device << u8" \u6d4b\u53f7\n"
                  << u8"  --mine --out \u65b0\u76ee\u5f55 [--start ID] [--end ID | --count N] [--report-interval \u79d2]\n"
                  << u8"  --out \u65b0\u76ee\u5f55 [--backend " << device << "|cpu|both] [--seconds 60 | --validate N]\n"
                  << u8"  \u516c\u7528\u9009\u9879\uff1a--plan \u6587\u4ef6 --team \u961f\u540d --name-template \u6a21\u677f --batch N --block N --threads N\n"
                  << u8"  \u6d4b\u53f7\u7ed3\u679c\u5199\u5165 results.csv\uff1bCtrl+C \u5b8c\u6210\u5f53\u524d\u6279\u6b21\u540e\u62a5\u544a next_id\u3002\n";
        return 0;
    }
    auto opts = arguments(argc, argv, device);
    const auto plan = rules(opts);
    const auto input = prepare(opts, plan);
    install_stop_handler();
    if (!fs::create_directory(opts.out)) throw std::runtime_error(u8"\u8f93\u51fa\u76ee\u5f55\u5df2\u7ecf\u5b58\u5728");
    Device gpu;
    if ((opts.validate || opts.backend != "cpu") && (!opts.mine || opts.start != *opts.end))
        gpu.init(input, opts);
    const auto run_cpu = [&](Id first, U n) { return cpu(input, first, n, opts.threads, false); };
    const auto run_gpu = [&](Id first, U n) { return gpu.run(first, n, opts.block, false); };
    if (opts.validate) {
        validate(opts, input, gpu);
    } else if (opts.mine) {
        if (opts.backend == "cpu") mine(opts, input, "cpu", run_cpu);
        else mine(opts, input, device, run_gpu);
    } else {
        std::ofstream metrics(opts.out / "metrics.csv", std::ios::binary);
        metrics.exceptions(std::ios::badbit | std::ios::failbit);
        metrics << "backend,candidates,passed,normal,blue,nonzero_ptt,e2e_s,kernel_s,names_s,T_per_day,T_per_week,eligible\n";
        if (opts.backend != device) measure(opts, input, "cpu", run_cpu, metrics);
        if (opts.backend != "cpu" && !stopped.load(std::memory_order_relaxed)) measure(opts, input, device, run_gpu, metrics);
    }
    return stopped.load(std::memory_order_relaxed) ? 130 : 0;
} catch (const std::exception& e) {
    std::cerr << "PBB " << device << u8"\uff1a" << e.what() << '\n';
    return 1;
}
