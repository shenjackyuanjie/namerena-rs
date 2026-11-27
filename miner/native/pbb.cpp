#if defined(_MSC_VER) && !defined(__clang__)
#define __restrict__ __restrict
#endif
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <immintrin.h>
#include <mutex>
#include <string_view>
#include <thread>
#include "pbb-run.hpp"
#include "pbb-prefilter.hpp"

// 纯 Rust 前核（miner/src/pbb_front.rs）：ID 置换 + UTF-8 键字 + 八路交错 KSA +
// SSE2 保留区解交错 + AVX2/BMI2 属性压缩 + 四平面整数门控（收不齐 31 项属性时也在 Rust 内闭环复核），
// 99.99% 的候选全程不回到 C++。
extern "C" {
void* tswn_rust_front_create(const Launch* launch);
void tswn_rust_front_destroy(void* front);
void tswn_rust_front_levels(const void* front, Id id, U* out);
U tswn_rust_front_scan_hits(const void* front, Id first_id, U count, U* hit_offsets, U* hit_levels);
unsigned tswn_rust_front_width();
size_t tswn_rust_launch_size();
}

static_assert(!PBB_HAS_SIMD && N == 256 && M == 128 && K == 64 && skill_cnt == 40);
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);

namespace pbb_cpu {
// Rust 前核句柄。Launch 的 POD 字段由 Rust 侧直接读取，初始化时核对布局。
class RustFront {
    void* handle = nullptr;
public:
    explicit RustFront(const Launch& launch) {
        if (tswn_rust_launch_size() != sizeof(Launch) || tswn_rust_front_width() != front_width)
            throw std::runtime_error(u8"Rust \u524d\u6838\u4e0e C++ \u5e03\u5c40\u4e0d\u4e00\u81f4");
        handle = tswn_rust_front_create(&launch);
        if (!handle) throw std::runtime_error(u8"Rust \u524d\u6838\u521d\u59cb\u5316\u5931\u8d25");
    }
    RustFront(const RustFront&) = delete;
    RustFront& operator=(const RustFront&) = delete;
    ~RustFront() { if (handle) tswn_rust_front_destroy(handle); }
    void levels(Id id, U (&out)[front_width]) const { tswn_rust_front_levels(handle, id, out); }
    U scan_hits(Id first_id, U count, U* hit_offsets, U* hit_levels) const {
        return tswn_rust_front_scan_hits(handle, first_id, count, hit_offsets, hit_levels);
    }
};

class Device {
    // 64 字节对齐彻底消除跨核伪共享。
    struct alignas(64) Part {
        Counts counts{};
        std::vector<Record> rows;
        std::vector<U> indices;
        std::exception_ptr error;
    };
    struct Task {
        Id first;
        U n, active;
        bool capture, verify_indices;
        Batch* result;
    } task{};

    static constexpr U kSpinLimit = 4096;
    static constexpr U kDynamicChunk = 4096; // front_width (8) 的整数倍

    const Launch* input = nullptr;
    ScorePlan plan{};
    // TSWN_PBB_FRONT=rust（默认）| cpp | scalar 用于同二进制 A/B。
    std::optional<RustFront> rust_front;
    std::optional<Front> cpp_front;
    bool dynamic_steal = true;
    std::vector<Part> parts;
    std::vector<std::thread> workers;

    alignas(64) std::atomic<Id> epoch{0};
    alignas(64) std::atomic<U> pending{0};
    alignas(64) std::atomic<U> next_cursor{0};
    alignas(64) std::atomic<bool> closed{false};
    // 仅在自旋超时退让时才走条件变量休眠，热循环零锁开销。
    std::mutex sleep_mutex;
    std::condition_variable sleep_cv;
    std::atomic<U> sleepers{0};

    void run_range(Part& part, Name& name_state, char (&name)[name_limit + 1], U begin, U end) {
        auto evaluate = [&](U i, U level) {
            // UINT32_MAX: 标量基线无需额外重复 KSA 去统计 Q 门。
            if (level != std::numeric_limits<U>::max()) {
                if (task.capture) task.result->gate_levels[i] = level;
                if (level) {
                    ++part.counts.eligible;
                    if (task.verify_indices) part.indices.push_back(i);
                } else {
                    if (task.capture) task.result->records[i].index = i;
                    return;
                }
            }
            make_name(*input, task.first + i, name);
            const auto scored = score_full(name, input->name_len, name_state, &plan);
            const auto flags = selection(input->rules, scored);
            part.counts.passed += scored.flag;
            part.counts.keep += flags.keep;
            part.counts.blue += flags.blue;
            part.counts.ptt += flags.ptt;
            if (task.capture) task.result->records[i] = wire(i, flags, scored);
            else if (flags.keep || flags.blue) part.rows.push_back(wire(i, flags, scored));
        };
        U i = begin;
        const bool aggressive = rust_front.has_value() || cpp_front.has_value();
        if (aggressive) {
            if (rust_front && !task.capture) {
                // 非捕获模式按块走 Rust 批量扫描，仅回传非零门控，省去 99.99% 跨语言边界调用。
                constexpr U kScanBatch = 4096;
                U hit_offsets[kScanBatch];
                U hit_levels[kScanBatch];
                while (end - i >= front_width) {
                    const U count = std::min(kScanBatch, (end - i) & ~(front_width - 1));
                    const U hits = rust_front->scan_hits(task.first + i, count, hit_offsets, hit_levels);
                    for (U h = 0; h < hits; ++h) evaluate(i + hit_offsets[h], hit_levels[h]);
                    i += count;
                }
            } else {
                for (; end - i >= front_width; i += front_width) {
                    U levels[front_width];
                    if (rust_front) rust_front->levels(task.first + i, levels);
                    else cpp_front->levels(task.first + i, levels);
                    for (U lane = 0; lane < front_width; ++lane) evaluate(i + lane, levels[lane]);
                }
            }
        }
        for (; i < end; ++i) {
            U level = std::numeric_limits<U>::max();
            if (aggressive || task.capture || task.verify_indices) {
                make_name(*input, task.first + i, name);
                name_state.load_name(name, input->name_len);
                int props[7];
                level = U(gate_level_of(name_state.name_base, props));
            }
            evaluate(i, level);
        }
    }

    void work(U worker) noexcept {
        auto& part = parts[worker];
        part.counts = {};
        part.rows.clear();
        part.indices.clear();
        part.error = nullptr;
        try {
            Name name_state{};
            init_candidate(name_state, *input);
            char name[name_limit + 1];
            if (dynamic_steal && task.active > 1 && task.n >= kDynamicChunk * 2) {
                for (;;) {
                    const U begin = next_cursor.fetch_add(kDynamicChunk, std::memory_order_relaxed);
                    if (begin >= task.n) break;
                    const U end = std::min(begin + kDynamicChunk, task.n);
                    run_range(part, name_state, name, begin, end);
                }
            } else {
                const U begin = U(Id(task.n) * worker / task.active);
                const U end = U(Id(task.n) * (worker + 1) / task.active);
                run_range(part, name_state, name, begin, end);
            }
        } catch (...) { part.error = std::current_exception(); }
    }

    void worker_loop(U worker) {
        Id seen = 0;
        for (;;) {
            U spins = 0;
            while (epoch.load(std::memory_order_acquire) == seen) {
                if (closed.load(std::memory_order_acquire)) return;
                if (++spins < kSpinLimit) {
                    _mm_pause();
                } else {
                    sleepers.fetch_add(1, std::memory_order_acq_rel);
                    std::unique_lock<std::mutex> lock(sleep_mutex);
                    sleep_cv.wait(lock, [&] {
                        return closed.load(std::memory_order_acquire) ||
                               epoch.load(std::memory_order_acquire) != seen;
                    });
                    sleepers.fetch_sub(1, std::memory_order_relaxed);
                    spins = 0;
                }
            }
            if (closed.load(std::memory_order_acquire)) return;
            seen = epoch.load(std::memory_order_acquire);
            if (worker < task.active) {
                work(worker);
                pending.fetch_sub(1, std::memory_order_release);
            }
        }
    }

public:
    Device() = default;
    Device(const Device&) = delete;
    ~Device() {
        closed.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(sleep_mutex);
        }
        sleep_cv.notify_all();
        for (auto& worker : workers) worker.join();
    }

    void init(const Launch& launch, const Options& opts) {
        input = &launch;
        plan = make_plan<ScorePlan>(launch.rules);
        const char* select = std::getenv("TSWN_PBB_FRONT");
        const std::string_view mode = select ? std::string_view(select) : std::string_view("rust");
        if (!opts.cpu_scalar && mode != "scalar") {
            if (mode == "cpp") cpp_front.emplace(launch);
            else rust_front.emplace(launch);
        }
        const char* sched = std::getenv("TSWN_PBB_SCHED");
        if (sched && std::string_view(sched) == "static") {
            dynamic_steal = false;
        }
        parts.resize(opts.threads);
        workers.reserve(opts.threads - 1);
        for (U i = 1; i < opts.threads; ++i) workers.emplace_back([this, i] { worker_loop(i); });
    }

    Batch run(Id first, U n, U, bool capture, bool verify_indices = false) {
        Batch out;
        if (!n) return out;
        if (capture) {
            out.gate_levels.resize(n);
            out.records.resize(n);
        }
        const U active = std::min(n, U(parts.size()));
        task = {first, n, active, capture, verify_indices, &out};
        next_cursor.store(0, std::memory_order_relaxed);
        const bool used_dynamic = dynamic_steal && active > 1 && n >= kDynamicChunk * 2;
        if (active > 1) {
            pending.store(active - 1, std::memory_order_relaxed);
            epoch.fetch_add(1, std::memory_order_release);
            if (sleepers.load(std::memory_order_acquire) > 0) {
                std::lock_guard<std::mutex> lock(sleep_mutex);
                sleep_cv.notify_all();
            }
        }
        work(0); // 主线程也参与评分；一线程模式无需跨线程唤醒。
        if (active > 1) {
            U spins = 0;
            while (pending.load(std::memory_order_acquire) != 0) {
                if (++spins < kSpinLimit) _mm_pause();
                else {
                    std::this_thread::yield();
                    spins = 0;
                }
            }
        }
        for (U i = 0; i < active; ++i) {
            const auto& part = parts[i];
            if (part.error) std::rethrow_exception(part.error);
            out.counts.passed += part.counts.passed;
            out.counts.keep += part.counts.keep;
            out.counts.blue += part.counts.blue;
            out.counts.ptt += part.counts.ptt;
            out.counts.eligible += part.counts.eligible;
        }
        if (!capture) out.records.reserve(size_t(out.counts.keep) + out.counts.blue);
        if (verify_indices) out.gate_indices.reserve(out.counts.eligible);
        for (U i = 0; i < active; ++i) {
            const auto& part = parts[i];
            out.records.insert(out.records.end(), part.rows.begin(), part.rows.end());
            out.gate_indices.insert(out.gate_indices.end(), part.indices.begin(), part.indices.end());
        }
        // 动态块窃取可能让不同 worker 收集到交错区间，汇总后按 index 升序还原确定性顺序。
        if (used_dynamic) {
            if (!capture && out.records.size() > 1) {
                std::sort(out.records.begin(), out.records.end(),
                          [](const Record& a, const Record& b) { return a.index < b.index; });
            }
            if (verify_indices && out.gate_indices.size() > 1) {
                std::sort(out.gate_indices.begin(), out.gate_indices.end());
            }
        }
        out.counts.written = U(out.records.size());
        return out;
    }
};
} // namespace pbb_cpu

extern "C" int tswn_pbb_main(int argc, const char* const* argv) noexcept {
    return pbb_main<pbb_cpu::Device, true>(argc, argv, "cpu");
}
