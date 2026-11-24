#include "pbb-run.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include "pbb_device_scoring.hpp" // 从同一份 PBB 源生成的 CUDA 评分。
#include "pbb_device_prefilter.hpp"

static_assert(pbb_device::M_N == families);
__constant__ Launch device_input;

template<int Width = 1>
__device__ void score_one(Id first, U i, bool capture, Counts* counts, Record* records) {
    char name[name_limit + 1];
    make_name(device_input, first + i, name);
    pbb_device::Name state;
    init_candidate(state, device_input);
    const auto plan = make_plan<pbb_device::ScorePlan>(device_input.rules);
    const auto score = pbb_device::score_full<Width>(name, device_input.name_len, state, &plan);
    if ((threadIdx.x & (Width - 1)) != 0) return;
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

// 批内编号仅占低 24 位。高位标记未收满 31 个属性字节、须由完整排列复核的候选。
constexpr U uncertain_q = 1u << 31;
constexpr int q_prefix = 88;
enum class FrontPath { Legacy, Fixed34, Generic };

template<int Block, FrontPath Path>
__global__ void select_ids(Id first, U n, bool capture, Counts* counts, Record* records,
                           U* indices, U* snapshots, U snapshot_cap) {
    __shared__ U words[64 * Block];
    const U i = blockIdx.x * Block + threadIdx.x;
    if (i >= n) return;
    const pbb_device::prefilter_detail::Box<Block> box{words + threadIdx.x};
    int props[7];
    bool uncertain = false, eligible = false;
    if constexpr (Path != FrontPath::Legacy) {
#if defined(__CUDA_ARCH__)
        if constexpr (Path == FrontPath::Fixed34) box.template schedule_cropped<q_prefix>(device_input, first + i);
        else box.template schedule_generic<q_prefix>(device_input, first + i);
        U packed[8];
        uncertain = box.template collect_packed<q_prefix / 4>(packed) < 31;
        eligible = uncertain || pbb_device::gate_level_of(reinterpret_cast<const pbb_device::u8_t*>(packed), props);
#endif
    } else {
        pbb_device::u8_t base[31];
        box.schedule(device_input, first + i);
        box.collect(base);
        eligible = pbb_device::gate_level_of(base, props);
    }
    if (eligible) {
        const U slot = atomicAdd(&counts->eligible, 1u);
        indices[slot] = i | (uncertain ? uncertain_q : 0);
        if constexpr (Path == FrontPath::Legacy) {
            if (slot < snapshot_cap) {
                #pragma unroll 8
                for (int j = 0; j < 64; ++j) snapshots[slot * 64 + j] = words[j * Block + threadIdx.x];
            }
        }
    } else if (capture) {
        Record rejected{};
        rejected.index = i;
        records[i] = rejected;
    }
}

// G3 单独执行；连续候选独占共享内存列，避免 PRGA 随机访问落到局部内存。
template<int Block>
struct GateName {
    const pbb_device::prefilter_detail::Box<Block> box;
    pbb_device::u8_t name_base[31], freq[16], skill[40];
    int V;

    __device__ __forceinline__ U read(U i) const { return box.cell(i); }
    __device__ __forceinline__ U prga(U i, U& j) const {
        auto& a = box.cell(i);
        const U x = a;
        j = (j + x) & 255u;
        auto& b = box.cell(j);
        const U y = b;
        a = y;
        b = x;
        return read((x+y)&255u);
    }

    __device__ __forceinline__ U prga2(U i, U& j) const {
        const U high = prga(i, j), low = prga(i+1, j);
        return (high<<8) | low;
    }

    __device__ __forceinline__ void calc_skills() {
        // 逆扫 128 个有效字节中的末 68 项：16 个四元组、2 个加成二元组。
        // 槽内顺序不影响最小值，扫描后仍用同一份完整 S-box 跑 PRGA。
        U left = 68, minimum = 63, boost[2];
        for (int word = 63;word >= 0 && left; --word) {
            const U packed = box.column[word*Block];
            #pragma unroll
            for (int b = 3; b >= 0; --b) {
                const U u=((packed>>(b*8))*181u+71u)&255u;
                if (u<128u && left) {
                    minimum=min(minimum,(u+89u)&63u);
                    --left;
                    if (left>=4 ? !(left&3u) : !(left&1u)) {
                        if(left>=4)freq[(left-4)/4]=max(int(minimum)-10,0);
                        else boost[left/2]=minimum;
                        minimum=63;
                    }
                }
            }
        }
        #pragma unroll 4
        for (int i = 0; i < 40; ++i) skill[i]=i;
        U move=0, j=0;
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < 40; ++i) {
                const U raw=prga2((p*40+i)*2+1,j);
                // ((raw % 40) + move + skill[i]) % 40 与一次取余相同。
                move = (move + raw + skill[i]) % 40u;
                const auto x=skill[i];
                skill[i]=skill[move]; skill[move]=x;
            }
        }
        int last=-1;
        for (int i = 0; i < 16; ++i) {
            if(skill[i]>=35)freq[i]=0;
            else if(skill[i]<25 && freq[i])last=i;
        }
        if(last>=0)freq[last]*=2;
        for (int i = 14; i < 16; ++i)
            if (i != last && freq[i]) freq[i] += min(U(freq[i]), boost[i-14]);
    }
};

template<int Block, bool Capture, FrontPath Path>
__global__ void select_g3(Id first, Counts* counts, Record* records, const U* in,
    const U* snapshots, U snapshot_cap, U* out, U* survivors, U* rejected) {
    __shared__ U words[64*Block];
    const U n=counts->eligible;
    for (U slot = blockIdx.x * Block + threadIdx.x; slot < n; slot += Block * gridDim.x) {
        const U queued=in[slot], id=queued&~uncertain_q;
        GateName<Block> state{{words+threadIdx.x}};
        // 裁剪核只传候选号；在 PRGA 所在的列直接重建完整排列。
        if constexpr (Path == FrontPath::Fixed34) {
            state.box.schedule_aligned(device_input, first + id);
        } else if constexpr (Path == FrontPath::Generic) {
            state.box.schedule_generic(device_input, first + id);
        } else if (slot >= snapshot_cap) {
            // 通用模板的有界快照满后，重建当前候选。
            state.box.schedule(device_input, first + id);
        } else {
            #pragma unroll 4
            for (int w=0;w<64;++w) words[w*Block+threadIdx.x]=snapshots[slot*64+w];
        }
        state.box.collect(state.name_base);
        state.V=pbb_device::prefilter_detail::finish_base(state.name_base);
        if (Path != FrontPath::Legacy && (queued & uncertain_q)) {
            int p[7];
            if (!pbb_device::gate_level_of(state.name_base,p))atomicAdd(rejected,1u);
        }
        if (pbb_device::qualify_g3(state)) out[atomicAdd(survivors,1u)]=id;
        else if constexpr(Capture) score_one(first,id,true,counts,records);
    }
}
// 一个 block 恰好一个 warp；每个 Width 线程小组负责一个候选。
template<int Width>
__global__ void evaluate_g3(Id first, bool capture, Counts* counts, Record* records,
                            const U* indices, const U* survivors) {
    const U n = *survivors;
    constexpr U groups = 32 / Width;
    for (U slot = blockIdx.x * groups + threadIdx.x / Width; slot < n; slot += gridDim.x * groups)
        score_one<Width>(first, indices[slot], capture, counts, records);
}

static void check(const char* operation, cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#define CHECK(expr) check(#expr, (expr))

struct Device {
    Counts *counts = nullptr, *host_counts = nullptr;
    U *indices = nullptr, *snapshots = nullptr, *g3_indices = nullptr, *survivors = nullptr;
    U snapshot_cap = 0, host_capacity = 0;
    U* fill_data = nullptr;
    FillSlot* fill_slots = nullptr;
    bool split = false, cropped = false;
    FrontPath front_path = FrontPath::Legacy;
    U *rejected_q = nullptr, *host_rejected_q = nullptr;
    int score_grid = 0, tail_grid = 0;
    Record *records = nullptr, *host_records = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr, q_done = nullptr, g3_done = nullptr;
    double stage_s[3]{};
    Device() = default;
    Device(const Device&) = delete;
    ~Device() {
        if (stream) cudaStreamSynchronize(stream);
        if (counts) cudaFree(counts);
        if (records) cudaFree(records);
        if (indices) cudaFree(indices);
        if (snapshots) cudaFree(snapshots);
        if (g3_indices) cudaFree(g3_indices);
        if (survivors) cudaFree(survivors);
        if (rejected_q) cudaFree(rejected_q);
        if (fill_data) cudaFree(fill_data);
        if (fill_slots) cudaFree(fill_slots);
        if (host_rejected_q) cudaFreeHost(host_rejected_q);
        if (host_counts) cudaFreeHost(host_counts);
        if (host_records) cudaFreeHost(host_records);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (q_done) cudaEventDestroy(q_done);
        if (g3_done) {
            cudaEventDestroy(g3_done);
            std::cerr << "CUDA 分段（含预热）：Q=" << stage_s[0] << "s G3=" << stage_s[1]
                      << "s Score=" << stage_s[2] << "s\n";
        }
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
        if (split && !input.fill_slots_count && input.fill_len == encoded_width
            && input.i_pre == input.prefix_len + 1 && input.j_pre == input.prefix_len) {
            if (input.prefix_len == 34 && !input.suffix_len) front_path = FrontPath::Fixed34;
            else if (input.prefix_len >= 5 && input.prefix_len <= 50 && !std::getenv("PBB_CUDA_LEGACY_FRONT"))
                front_path = FrontPath::Generic;
        }
        cropped = front_path != FrontPath::Legacy;
        if (cropped) {
            CHECK(cudaMalloc(&rejected_q, sizeof(U)));
            CHECK(cudaMallocHost(&host_rejected_q, sizeof(U)));
        }
        score_grid = 8 * property.multiProcessorCount;
        // 让更多块各自开始处理一个小组批次，减轻 PTT 较重的候选拖慢同一块的后续批次。
        tail_grid = 12 * property.multiProcessorCount;
        if (split) {
            CHECK(cudaMalloc(&indices, capacity * sizeof(*indices)));
            if (!cropped) {
                snapshot_cap = U((capacity + 7) / 8);
                CHECK(cudaMalloc(&snapshots, size_t(snapshot_cap) * 256));
            }
            CHECK(cudaMalloc(&g3_indices, capacity * sizeof(*g3_indices)));
            CHECK(cudaMalloc(&survivors, sizeof(*survivors)));
        }
        Launch device_launch = input;
        if (input.fill_slots_count) {
            const auto data_bytes = input.fill_data_count * sizeof(*fill_data);
            const auto slot_bytes = input.fill_slots_count * sizeof(*fill_slots);
            CHECK(cudaMalloc(&fill_data, data_bytes));
            CHECK(cudaMalloc(&fill_slots, slot_bytes));
            CHECK(cudaMemcpy(fill_data, input.fill_data, data_bytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(fill_slots, input.fill_slots, slot_bytes, cudaMemcpyHostToDevice));
            device_launch.fill_data = fill_data;
            device_launch.fill_slots = fill_slots;
        }
        CHECK(cudaMemcpyToSymbol(device_input, &device_launch, sizeof(device_launch)));
        if (split) {
            using namespace pbb_device::prefilter_detail;
            PreparedFront front{};
            for (int j = 0; j < 256; ++j) front.words[j / 4] |= U(input.prefix_box[j]) << (8 * (j & 3));
            front.pairs[0] = static_cast<unsigned char>(input.prefix[0]) << 8;
            for (int j = 1; j < 16; ++j) {
                const auto a = static_cast<unsigned char>(input.prefix[2*j-1]);
                const auto b = static_cast<unsigned char>(input.prefix[2*j]);
                front.pairs[j] = a | (b << 8);
            }
            front.pairs[16] = U(static_cast<unsigned char>(input.prefix[31]))
                | (U(static_cast<unsigned char>(input.prefix[32])) << 8);
            front.pairs[17] = static_cast<unsigned char>(input.prefix[33]);
            CHECK(cudaMemcpyToSymbol(prepared_front, &front, sizeof(front)));
            if (front_path == FrontPath::Generic) {
                PreparedGeneric generic{};
                const int fill = input.i_pre, fill_end = fill + encoded_width, cycle = input.name_len + 1;
                for (int i = 0, k = 0; i < N; ++i) {
                    U byte = 0;
                    if (k > 0 && k < fill) byte = static_cast<unsigned char>(input.prefix[k - 1]);
                    if (k >= fill_end) byte = static_cast<unsigned char>(input.suffix[k - fill_end]);
                    generic.pairs[i / 2] |= byte << (8 * (i & 1));
                    generic.words[i / 4] |= byte << (8 * (i & 3));
                    if (++k == cycle) k = 0;
                }
                generic.crop_fill = fill;
                while (generic.crop_fill + encoded_width <= q_prefix) generic.crop_fill += cycle;
                CHECK(cudaMemcpyToSymbol(prepared_generic, &generic, sizeof(generic)));
            }
        }
        CHECK(cudaMalloc(&counts, sizeof(*counts)));
        CHECK(cudaMalloc(&records, capacity * sizeof(*records)));
        CHECK(cudaMallocHost(&host_counts, sizeof(*host_counts)));
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cudaEventCreate(&begin));
        CHECK(cudaEventCreate(&end));
        if (split && std::getenv("PBB_CUDA_PROFILE")) {
            CHECK(cudaEventCreate(&q_done));
            CHECK(cudaEventCreate(&g3_done));
        }
    }
    Batch run(Id first, U n, U block, bool capture, bool = false) {
        CHECK(cudaMemsetAsync(counts, 0, sizeof(*counts), stream));
        if (split) CHECK(cudaMemsetAsync(survivors, 0, sizeof(*survivors), stream));
        if (cropped) CHECK(cudaMemsetAsync(rejected_q, 0, sizeof(U), stream));
        CHECK(cudaEventRecord(begin, stream));
        if (split) {
            #define FRONT(b, path) select_ids<b, FrontPath::path><<<(n + b - 1) / b, b, 0, stream>>>(first, n, capture, counts, records, indices, snapshots, snapshot_cap)
            #define CHOOSE(b) if (front_path == FrontPath::Fixed34) { FRONT(b, Fixed34); } \
                else if (front_path == FrontPath::Generic) { FRONT(b, Generic); } else { FRONT(b, Legacy); }
            switch (block) {
                case 32: CHOOSE(32); break;
                case 64: CHOOSE(64); break;
                case 96: CHOOSE(96); break;
                case 128: CHOOSE(128); break;
                default: throw std::runtime_error("split path supports block 32/64/96/128");
            }
            #undef CHOOSE
            #undef FRONT
            CHECK(cudaGetLastError());
            if (q_done) CHECK(cudaEventRecord(q_done, stream));
            #define G3(b, cap, path) select_g3<b, cap, FrontPath::path><<<score_grid, b, 0, stream>>>(first, counts, records, indices, snapshots, snapshot_cap, g3_indices, survivors, rejected_q)
            if (front_path == FrontPath::Fixed34) {
                if (capture) G3(64, true, Fixed34);
                else G3(96, false, Fixed34);
            } else if (front_path == FrontPath::Generic) {
                if (capture) G3(64, true, Generic);
                else G3(96, false, Generic);
            } else {
                if (capture) G3(64, true, Legacy);
                else G3(64, false, Legacy);
            }
            #undef G3
            CHECK(cudaGetLastError());
            if (g3_done) CHECK(cudaEventRecord(g3_done, stream));
            evaluate_g3<8><<<tail_grid, 32, 0, stream>>>(first, capture, counts, records, g3_indices, survivors);
        } else {
            evaluate<<<(n + block - 1) / block, block, 0, stream>>>(first, n, capture, counts, records);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(end, stream));
        CHECK(cudaMemcpyAsync(host_counts, counts, sizeof(*counts), cudaMemcpyDeviceToHost, stream));
        if (cropped) CHECK(cudaMemcpyAsync(host_rejected_q, rejected_q, sizeof(U), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        Batch result;
        result.counts = *host_counts;
        if (cropped) {
            if (*host_rejected_q > result.counts.eligible) throw std::runtime_error("invalid Q gate count");
            result.counts.eligible -= *host_rejected_q;
        }
        if (capture) result.counts.written = n;
        if (result.counts.eligible > n || (split && result.counts.passed > result.counts.eligible) || result.counts.written > n || result.counts.passed > n || result.counts.keep > result.counts.passed
            || result.counts.blue > result.counts.passed || result.counts.ptt > result.counts.passed)
            throw std::runtime_error("invalid GPU count");
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, begin, end));
        result.kernel_s = double(ms) / 1000;
        if (g3_done) {
            const cudaEvent_t boundaries[] = {begin, q_done, g3_done, end};
            for (size_t i = 0; i < 3; ++i) {
                CHECK(cudaEventElapsedTime(&ms, boundaries[i], boundaries[i + 1]));
                stage_s[i] += double(ms) / 1000;
            }
        }
        if (result.counts.written) {
            if (result.counts.written > host_capacity) {
                if (host_records) CHECK(cudaFreeHost(host_records));
                host_records = nullptr;
                CHECK(cudaMallocHost(&host_records, size_t(result.counts.written) * sizeof(*records)));
                host_capacity = result.counts.written;
            }
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
