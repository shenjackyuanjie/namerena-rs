// 独立 CUDA 对照：GPU 生成 Base16384 名字、双 KSA、属性/技能/影子、同源 2.0 模型。
// 接口只供 examples/cuda-ab 使用；一个进程只创建一个 CUDA 工作负载。
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <new>

using U = uint32_t;
using Byte = uint8_t;
struct Header { U count, prop_pass, scored, shadows; };
struct Record {
    U index, props[8], ids[16], freqs[16];
    int32_t shadow;
    float xp, xd;
};
struct Readback { Header header; float kernel_ms; const Record* records; };
static_assert(sizeof(Header) == 16 && sizeof(Record) == 176);

// key[0] 是前导零。宿主已附好 ?shadow；token 0x100..0x111 引用生成字节。
struct Input { U prefix[64], key[264], len, first, j, single_slot; };
__constant__ Input input;

__device__ __forceinline__ U utf8_3(U c) {
    return (0xe0u | (c >> 12)) | ((0x80u | ((c >> 6) & 63u)) << 8)
                              | ((0x80u | (c & 63u)) << 16);
}

__device__ __forceinline__ U key_byte(const U (&chunks)[6], U k) {
    U token = input.key[k];
    if (token < 256) return token;
    token &= 255;
    return (chunks[token / 3] >> ((token % 3) * 8)) & 255;
}

// 每线程独占一个转置的列，同行线程不会读写彼此的字，无需 barrier。
template<int Lanes> struct Sbox {
    U* column;
    __device__ __forceinline__ void restore() const {
        for (U i = 0; i < 64; ++i) column[i * Lanes] = input.prefix[i];
    }
    __device__ __forceinline__ static U byte(U word, U slot) {
        return (word >> ((slot & 3) * 8)) & 255;
    }
    __device__ __forceinline__ U& packed(U slot) const { return column[(slot >> 2) * Lanes]; }
    __device__ __forceinline__ U get(U slot) const { return byte(packed(slot), slot); }
    __device__ __forceinline__ U exchange(U a, U b, U word_a, U cur) const {
        U word_b = packed(b);
        U other = byte(word_b, b);
        // XOR 仅改两个目标字节；同字时第二次写入要合并第一次的更新。
        U delta = cur ^ other;
        U update_a = word_a ^ (delta << ((a & 3) * 8));
        packed(a) = update_a;
        packed(b) = (((a ^ b) < 4) ? update_a : word_b) ^ (delta << ((b & 3) * 8));
        return other;
    }
    __device__ __forceinline__ void ksa_step(U i, U& j, U key) const {
        U word = packed(i);
        U cur = byte(word, i);
        j = (j + key + cur) & 255;
        exchange(i, j, word, cur);
    }
    __device__ __forceinline__ void ksa_literal(U& i, U& j, U begin, U end) const {
        const U stop = min(256u, i + end - begin);
        for (U k = begin; i < stop; ++i, ++k) ksa_step(i, j, input.key[k]);
    }
    __device__ __forceinline__ void ksa_single_slot(const U (&chunks)[6], U extra) const {
        const U first = input.first;
        const U end = input.len + extra + 1;
        #pragma unroll 1
        for (U round = 0; round < 2; ++round) {
            U i = round == 0 ? first : 0;
            U j = round == 0 ? input.j : 0;
            U prefix_begin = i; // 首轮从快照继续；次轮从含前导零的前缀开始。
            while (i < 256) {
                ksa_literal(i, j, prefix_begin, first);
                if (i > 256 - 18) {
                    for (U k = first; i < 256; ++i, ++k) ksa_step(i, j, key_byte(chunks, k));
                    break;
                }
                #pragma unroll 6
                for (int c = 0; c < 6; ++c) {
                    const U word = chunks[c];
                    ksa_step(i++, j, word & 255);
                    ksa_step(i++, j, (word >> 8) & 255);
                    ksa_step(i++, j, word >> 16);
                }
                ksa_literal(i, j, first + 18, end); // 名字后缀和影子字节都是常量。
                prefix_begin = 0;
            }
        }
    }
    __device__ __forceinline__ void ksa(const U (&chunks)[6], U extra) const {
        restore();
        if (input.single_slot) {
            ksa_single_slot(chunks, extra);
            return;
        }
        for (U round = 0; round < 2; ++round) {
            U i = round == 0 ? input.first : 0;
            U k = i, j = round == 0 ? input.j : 0;
            for (; i < 256; ++i) {
                ksa_step(i, j, key_byte(chunks, k));
                k = k == input.len + extra ? 0 : k + 1;
            }
        }
    }
    template<bool All>
    __device__ __forceinline__ U collect(Byte* base, U& slot, U filled, U need) const {
        if constexpr (All) {
            // 完整扫描按同样的字节顺序收集，每个共享字只读取一次。
            for (U i = 0; i < 64; ++i) {
                const U word = column[i * Lanes];
                #pragma unroll
                for (U j = 0; j < 4; ++j) {
                    const U value = ((word >> (8 * j)) * 181 + 160) & 255;
                    if (value >= 89 && value < 217) base[filled++] = Byte(value & 63);
                }
            }
            slot = 256;
        } else {
            for (; slot < 256 && filled < need; ++slot) {
                const U value = (get(slot) * 181 + 160) & 255;
                if (value >= 89 && value < 217) base[filled++] = Byte(value & 63);
            }
        }
        return filled;
    }
    __device__ __forceinline__ U prga(U& a, U& b) const {
        a = (a + 1) & 255;
        U word = packed(a);
        U cur = byte(word, a);
        b = (b + cur) & 255;
        U other = exchange(a, b, word, cur);
        return get((cur + other) & 255);
    }
};

// PBB 的可移植部分：对 HP 的 10 个元素用固定 compare-exchange 网络。
template<bool Network>
__device__ __forceinline__ U properties(const Byte* base, U* props) {
    U hp[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) hp[i] = base[i];
    if constexpr (Network) {
        #define CE(i, j) { U x = hp[i], y = hp[j]; hp[i] = min(x, y); hp[j] = max(x, y); }
        CE(0,5); CE(1,6); CE(2,7); CE(3,8); CE(4,9);
        CE(0,3); CE(1,4); CE(5,8); CE(6,9);
        CE(0,2); CE(3,6); CE(7,9);
        CE(0,1); CE(2,4); CE(5,7); CE(8,9);
        CE(1,2); CE(3,5); CE(4,6); CE(7,8);
        CE(1,3); CE(2,5); CE(4,7); CE(6,8);
        CE(2,3); CE(6,7);
        CE(3,4); CE(5,6);
        CE(4,5);
        #undef CE
    } else {
        for (int i = 1; i < 10; ++i) {
            U cur = hp[i];
            int j = i;
            for (; j > 0 && hp[j - 1] > cur; --j) hp[j] = hp[j - 1];
            hp[j] = cur;
        }
    }
    U sum = props[0] = 154 + hp[3] + hp[4] + hp[5] + hp[6];
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        U x = base[3 * i + 7], y = base[3 * i + 8], z = base[3 * i + 9];
        props[i] = 36 + max(min(x, y), min(max(x, y), z));
        sum += 3 * props[i];
    }
    return sum;
}

template<int Lanes, int Mode>
__global__ void mine(Header* output, Record* records, const float2* __restrict__ weights, uint64_t start, U count,
                     U prop, U xp, U xd, bool capture) {
    __shared__ U values[64 * Lanes];
    U index = blockIdx.x * Lanes + threadIdx.x;
    if (index >= count) return;
    uint64_t id = start + index;
    U lo = U(id), hi = U(id >> 32);
    const U chunks[6] = {
        utf8_3(0x4e00 + ((hi >> 18) & 0x3fff)), utf8_3(0x4e00 + ((hi >> 4) & 0x3fff)),
        utf8_3(0x4e00 + (((hi << 10) | (lo >> 22)) & 0x3fff)),
        utf8_3(0x4e00 + ((lo >> 8) & 0x3fff)), utf8_3(0x4e00 + ((lo & 255) << 6)),
        utf8_3(0x3d01),
    };
    const Sbox<Lanes> box{values + threadIdx.x};
    box.ksa(chunks, 0);
    Byte base[128];
    U props[8], slot = 0;
    U filled = box.template collect<(Mode != 0)>(base, slot, 0, Mode == 0 ? 31 : 128);
    U sum = properties<Mode == 2>(base, props);
    if (!capture && sum <= 3 * prop) return;
    if constexpr (Mode == 0) box.template collect<false>(base, slot, filled, 128);
    atomicAdd(&output->prop_pass, 1u);

    // 技能 ID <40；频率基础至多 53，倍增/附加后仍不超过 106。
    Byte ids[40], freqs[16];
    #pragma unroll
    for (U i = 0; i < 40; ++i) ids[i] = i;
    U a = 0, b = 0, s = 0;
    for (int round = 0; round < 2; ++round) {
        #pragma unroll 1
        for (U i = 0; i < 40; ++i) {
            U high = box.prga(a, b), low = box.prga(a, b);
            U cur = ids[i];
            // s、余数、cur 都小于 40；旧和小于 256，取模可以合并。
            s = (((high << 8) | low) + s + cur) % 40u;
            ids[i] = ids[s];
            ids[s] = Byte(cur);
        }
    }
    int last = -1;
    for (U i = 0; i < 16; ++i) {
        U k = 64 + i * 4;
        U p = min(min(U(base[k]), U(base[k+1])), min(U(base[k+2]), U(base[k+3])));
        freqs[i] = 0;
        if (p > 10 && ids[i] < 35) {
            freqs[i] = Byte(p - 10);
            if (p < 35) last = int(i);
        }
    }
    if (last >= 0) freqs[last] *= 2;
    if (last != 14 && freqs[14]) freqs[14] += min(min(U(base[60]), U(base[61])), U(freqs[14]));
    if (last != 15 && freqs[15]) freqs[15] += min(min(U(base[62]), U(base[63])), U(freqs[15]));

    float features[44] = {};
    for (U i = 0; i < 8; ++i) features[i] = float(props[i]);
    for (U i = 0; i < 16; ++i) if (freqs[i]) features[ids[i] + 8] = float(freqs[i]);
    int32_t shadow = 0;
    if (features[32] > 0.0f) {
        atomicAdd(&output->shadows, 1u);
        box.ksa(chunks, 7);
        slot = 0;
        // 影子只使用前 31 个属性值，不需要计算技能。
        box.template collect<false>(base, slot, 0, 31);
        U shadow_props[8];
        U shadow_sum = properties<Mode == 2>(base, shadow_props);
        shadow = int32_t(shadow_sum) - 3 * 7 * 36 - 9 * int32_t(shadow_props[7] - 36) - 630;
        features[43] = float(shadow) / 3.0f * features[32] / 100.0f;
    }
    if (features[42] > 0.0f) features[42] += 20.0f;

    U indices[32], size = 0;
    for (U i = 0; i < 44; ++i) {
        float f = features[i];
        if (f != 0.0f) {
            indices[size] = i;
            features[size++] = f;
        }
    }
    float2 score = weights[0];
    for (U i = 0; i < size; ++i) {
        float2 w = weights[1 + indices[i]];
        score.x += w.x * features[i];
        score.y += w.y * features[i];
    }
    for (U i = 0; i < size; ++i) {
        U feature = indices[i];
        U row = 45 + feature * (89 - feature) / 2 - feature;
        for (U j = i; j < size; ++j) {
            float2 w = weights[row + indices[j]];
            float term = features[i] * features[j];
            score.x += w.x * term;
            score.y += w.y * term;
        }
    }
    atomicAdd(&output->scored, 1u);
    bool hit = (xp == 0 || score.x + 10.0f >= float(xp)) && (xd == 0 || score.y + 10.0f >= float(xd));
    if (!capture && !hit && isfinite(score.x) && isfinite(score.y)) return;
    Record& result = records[atomicAdd(&output->count, 1u)];
    result.index = index;
    for (int i = 0; i < 8; ++i) result.props[i] = props[i];
    for (int i = 0; i < 16; ++i) { result.ids[i] = ids[i]; result.freqs[i] = freqs[i]; }
    result.shadow = shadow;
    result.xp = score.x;
    result.xd = score.y;
}

struct Bench {
    U capacity = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    float2* weights = nullptr;
    Header* output = nullptr;
    Header* header = nullptr;
    Record* records = nullptr;
    ~Bench() {
        if (stream) cudaStreamSynchronize(stream);
        if (output) cudaFree(output);
        if (weights) cudaFree(weights);
        if (header) cudaFreeHost(header);
        if (records) cudaFreeHost(records);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
    }
};

#define API extern "C" __declspec(dllexport)
#define CUDA(call) do { cudaError_t error = (call); if (error != cudaSuccess) return error; } while (0)

static cudaError_t setup(Bench& b, const Input& data, const void* weights) {
    CUDA(cudaMemcpyToSymbol(input, &data, sizeof(data)));
    CUDA(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreate(&b.begin));
    CUDA(cudaEventCreate(&b.end));
    CUDA(cudaMalloc(&b.weights, 1035 * sizeof(float2)));
    CUDA(cudaMemcpy(b.weights, weights, 1035 * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA(cudaMalloc(&b.output, sizeof(Header) + size_t(b.capacity) * sizeof(Record)));
    CUDA(cudaMallocHost(&b.header, sizeof(Header)));
    return cudaMallocHost(&b.records, size_t(b.capacity) * sizeof(Record));
}

API int cuda_mine_open(const U* state, const U* map, U len, U first, U j,
                       const void* weights, U capacity, void** handle) {
    if (len > 256 || first > len || j > 255 || capacity == 0) return cudaErrorInvalidValue;
    auto* b = new (std::nothrow) Bench;
    if (!b) return cudaErrorMemoryAllocation;
    b->capacity = capacity;
    Input data{};
    for (U i = 0; i < 64; ++i) {
        data.prefix[i] = state[4*i] | (state[4*i+1] << 8) | (state[4*i+2] << 16) | (state[4*i+3] << 24);
    }
    for (U i = 0; i < len; ++i) data.key[i+1] = map[i];
    for (U i = 0; i < 7; ++i) data.key[len+1+i] = "?shadow"[i];
    data.len = len;
    data.first = first;
    data.j = j;
    // 只在快照恰位于唯一连续的 18 个生成字节之前且其余字节全固定时分段。
    if (first != 0 && first + 18 <= len + 1) {
        data.single_slot = 1;
        for (U k = 1; k <= len; ++k) {
            const bool valid = k >= first && k < first + 18
                ? data.key[k] == 0x100u + k - first : data.key[k] < 256;
            if (!valid) { data.single_slot = 0; break; }
        }
    }
    auto error = setup(*b, data, weights);
    if (error != cudaSuccess) { delete b; return error; }
    *handle = b;
    return cudaSuccess;
}

API int cuda_mine_run(void* handle, uint64_t start, U count, U prop, U xp, U xd, U capture,
                     U lanes, U mode, Readback* result) {
    Bench& b = *static_cast<Bench*>(handle);
    if (count == 0 || count > b.capacity || (lanes != 32 && lanes != 64 && lanes != 128) || mode > 2)
        return cudaErrorInvalidValue;
    CUDA(cudaMemsetAsync(b.output, 0, sizeof(Header), b.stream));
    CUDA(cudaEventRecord(b.begin, b.stream));
    U groups = (count + lanes - 1) / lanes;
    auto* device_records = reinterpret_cast<Record*>(reinterpret_cast<char*>(b.output) + sizeof(Header));
    #define RUN(l, m) mine<l, m><<<groups, l, 0, b.stream>>>(b.output, device_records, b.weights, start, count, prop, xp, xd, capture != 0)
    #define BLOCK(l) if (mode == 0) { RUN(l, 0); } else if (mode == 1) { RUN(l, 1); } else { RUN(l, 2); }
    if (lanes == 32) { BLOCK(32); } else if (lanes == 64) { BLOCK(64); } else { BLOCK(128); }
    #undef BLOCK
    #undef RUN
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(b.end, b.stream));
    CUDA(cudaMemcpyAsync(b.header, b.output, sizeof(Header), cudaMemcpyDeviceToHost, b.stream));
    CUDA(cudaStreamSynchronize(b.stream));
    CUDA(cudaEventElapsedTime(&result->kernel_ms, b.begin, b.end));
    result->header = *b.header;
    if (b.header->count > count) return cudaErrorInvalidValue;
    if (b.header->count) {
        CUDA(cudaMemcpyAsync(b.records, device_records, size_t(b.header->count) * sizeof(Record),
                             cudaMemcpyDeviceToHost, b.stream));
        CUDA(cudaStreamSynchronize(b.stream));
    }
    result->records = b.records;
    return cudaSuccess;
}

API void cuda_mine_close(void* handle) { delete static_cast<Bench*>(handle); }
API const char* cuda_mine_error(int error) { return cudaGetErrorString(static_cast<cudaError_t>(error)); }
