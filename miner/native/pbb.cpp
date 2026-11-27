#if defined(_MSC_VER) && !defined(__clang__)
#define __restrict__ __restrict
#endif
#include <condition_variable>
#include <mutex>
#include "pbb-run.hpp"
#include "pbb-prefilter.hpp"

static_assert(!PBB_HAS_SIMD && N == 256 && M == 128 && K == 64 && skill_cnt == 40);
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);

namespace pbb_cpu {
class Device {
    struct Part {
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

    const Launch* input = nullptr;
    ScorePlan plan{};
    std::optional<Front> front;
    std::vector<Part> parts;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable available, finished;
    Id epoch = 0;
    U pending = 0;
    bool closed = false;

    void work(U worker) noexcept {
        auto& part = parts[worker];
        part.counts = {};
        part.rows.clear();
        part.indices.clear();
        part.error = nullptr;
        try {
            const U begin = U(Id(task.n) * worker / task.active);
            const U end = U(Id(task.n) * (worker + 1) / task.active);
            Name name_state{};
            init_candidate(name_state, *input);
            // 顺序枚举（无填充置换）时，同一 worker 的连续 id 段共享名字前缀。把这段共同前缀
            // 一次性预处理进 name_state，后续候选的 KSA 就能从更靠后的位置起步——省下的步数
            // 等于少做的 KSA 步数，因为 KSA 固定做满 256 步，前置共享的那段是纯赚。
            // 有填充置换时 id→名字不再单调，首尾相同不等于中间也相同，保持 prepare() 给的短前缀。
            if (!input->fill_shuffle && end > begin) {
                char head[name_limit + 1];
                char tail[name_limit + 1];
                make_name(*input, task.first + begin, head);
                make_name(*input, task.first + end - 1, tail);
                int common = 0;
                while (common < input->name_len && head[common] == tail[common]) ++common;
                if (common > name_state.PRELEN) {
                    name_state.PRELEN = common;
                    name_state.load_prefix(head, input->name_len);
                }
            }
            char name[name_limit + 1];
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
            if (front) {
                for (; end - i >= front_width; i += front_width) {
                    U levels[front_width];
                    front->levels(task.first + i, levels);
                    for (U lane = 0; lane < front_width; ++lane) evaluate(i + lane, levels[lane]);
                }
            }
            for (; i < end; ++i) {
                U level = std::numeric_limits<U>::max();
                if (front || task.capture || task.verify_indices) {
                    make_name(*input, task.first + i, name);
                    name_state.load_name(name, input->name_len);
                    int props[7];
                    level = U(gate_level_of(name_state.name_base, props));
                }
                evaluate(i, level);
            }
        } catch (...) { part.error = std::current_exception(); }
    }

    void worker_loop(U worker) {
        Id seen = 0;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            available.wait(lock, [&] { return closed || epoch != seen; });
            if (closed) return;
            seen = epoch;
            if (worker >= task.active) continue;
            lock.unlock();
            work(worker);
            lock.lock();
            if (!--pending) finished.notify_one();
        }
    }

public:
    Device() = default;
    Device(const Device&) = delete;
    ~Device() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
        }
        available.notify_all();
        for (auto& worker : workers) worker.join();
    }

    void init(const Launch& launch, const Options& opts) {
        input = &launch;
        plan = make_plan<ScorePlan>(launch.rules);
        if (!opts.cpu_scalar) front.emplace(launch);
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
        {
            std::lock_guard<std::mutex> lock(mutex);
            task = {first, n, active, capture, verify_indices, &out};
            pending = active - 1;
            ++epoch;
        }
        if (active > 1) available.notify_all();
        work(0); // 主线程也参与评分；一线程模式无需跨线程唤醒。
        if (active > 1) {
            std::unique_lock<std::mutex> lock(mutex);
            finished.wait(lock, [&] { return pending == 0; });
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
        out.counts.written = U(out.records.size());
        return out;
    }
};
} // namespace pbb_cpu

extern "C" int tswn_pbb_main(int argc, const char* const* argv) noexcept {
    return pbb_main<pbb_cpu::Device, true>(argc, argv, "cpu");
}
