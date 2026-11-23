// 独立 CUDA 对照：GPU 生成 Base16384 名字、双 KSA、属性/技能/影子、同源 2.0 模型。
// 接口只供 examples/cuda-ab 使用；一个进程只创建一个 CUDA 工作负载。
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <new>

using U = uint32_t;
struct Header { U count, prop_pass, scored, shadows; };
struct Record {
    U index, props[8], ids[16], freqs[16];
    int32_t shadow;
    float xp, xd;
};
struct Readback { Header header; float kernel_ms; const Record* records; };
static_assert(sizeof(Header) == 16 && sizeof(Record) == 176);

// key[0] 是前导零。宿主已附好 ?shadow；token 0x100..0x111 引用生成字节。
struct Input { U prefix[64], key[264], len, first, j; };
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
    __device__ __forceinline__ U get(U slot) const {
        return (column[(slot >> 2) * Lanes] >> ((slot & 3) * 8)) & 255;
    }
    __device__ __forceinline__ void set(U slot, U value) const {
        U& word = column[(slot >> 2) * Lanes];
        U shift = (slot & 3) * 8;
        word = (word & ~(255u << shift)) | (value << shift);
    }
    __device__ __forceinline__ void ksa(const U (&chunks)[6], U extra) const {
        restore();
        for (U round = 0; round < 2; ++round) {
            U i = round == 0 ? input.first : 0;
            U k = i, j = round == 0 ? input.j : 0;
            for (; i < 256; ++i) {
                U cur = get(i);
                j = (j + key_byte(chunks, k) + cur) & 255;
                U other = get(j);
                set(i, other);
                set(j, cur); // 两槽在同一 word 时，必须读刚更新过的值。
                k = k == input.len + extra ? 0 : k + 1;
            }
        }
    }
    template<bool All>
    __device__ __forceinline__ U collect(U* base, U& slot, U filled, U need) const {
        for (; slot < 256 && (All || filled < need); ++slot) {
            U value = (get(slot) * 181 + 160) & 255;
            if (value >= 89 && value < 217) base[filled++] = value & 63;
        }
        return filled;
    }
    __device__ __forceinline__ U prga(U& a, U& b) const {
        a = (a + 1) & 255;
        U cur = get(a);
        b = (b + cur) & 255;
        U other = get(b);
        set(a, other);
        set(b, cur);
        return get((cur + other) & 255);
    }
};

// PBB 的可移植部分：对 HP 的 10 个元素用固定 compare-exchange 网络。
template<bool Network>
__device__ __forceinline__ U properties(const U* base, U* props) {
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
    U base[128], props[8], slot = 0;
    U filled = box.template collect<(Mode != 0)>(base, slot, 0, Mode == 0 ? 31 : 128);
    U sum = properties<Mode == 2>(base, props);
    if (!capture && sum <= 3 * prop) return;
    if constexpr (Mode == 0) box.template collect<false>(base, slot, filled, 128);
    atomicAdd(&output->prop_pass, 1u);

    U ids[40], freqs[16];
    #pragma unroll
    for (U i = 0; i < 40; ++i) ids[i] = i;
    U a = 0, b = 0, s = 0;
    for (int round = 0; round < 2; ++round) {
        #pragma unroll 1
        for (U i = 0; i < 40; ++i) {
            U high = box.prga(a, b), low = box.prga(a, b);
            U rnd = ((high << 8) | low) % 40;
            U cur = ids[i];
            s = ((s + rnd + cur) & 255) % 40;
            ids[i] = ids[s];
            ids[s] = cur;
        }
    }
    int last = -1;
    for (U i = 0; i < 16; ++i) {
        U k = 64 + i * 4;
        U p = min(min(base[k], base[k+1]), min(base[k+2], base[k+3]));
        freqs[i] = 0;
        if (p > 10 && ids[i] < 35) {
            freqs[i] = p - 10;
            if (p < 35) last = int(i);
        }
    }
    if (last >= 0) freqs[last] *= 2;
    if (last != 14 && freqs[14]) freqs[14] += min(min(base[60], base[61]), freqs[14]);
    if (last != 15 && freqs[15]) freqs[15] += min(min(base[62], base[63]), freqs[15]);

    float features[44] = {};
    for (U i = 0; i < 8; ++i) features[i] = float(props[i]);
    for (U i = 0; i < 16; ++i) if (freqs[i]) features[ids[i] + 8] = float(freqs[i]);
    int32_t shadow = 0;
    if (features[32] > 0.0f) {
        atomicAdd(&output->shadows, 1u);
        box.ksa(chunks, 7);
        slot = 0;
        box.template collect<(Mode != 0)>(base, slot, 0, Mode == 0 ? 31 : 128);
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
