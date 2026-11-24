#pragma once
#include <array>
#include <tmmintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace pbb_cpu {
inline constexpr U front_width = 8;
static_assert(encoded_width == 15 && sizeof(Id) == 8);

// 只做确定性的整数 Q 门。收不齐属性的候选交回原版，不能按未过门处理。
class Front {
    static constexpr int limit = 96;
    using Box = std::array<u8_t, 256>;
    using Boxes = std::array<Box, front_width>;
    using Word = std::uint64_t;
    using Words = std::array<Word, 256>;
    using Fills = std::array<Word, name_limit>;

    struct Span { int i, end, offset; bool varying; };
    const Launch& input;
    std::vector<Span> first, head, tail;
    Words fixed_first{}, fixed_second{}, initial{};
    const bool ssse3;

    struct Packing {
        std::array<std::uint64_t, 256> indices{};
        std::array<u8_t, 256> counts{};
        constexpr Packing() {
            for (U mask = 0; mask < 256; ++mask) {
                std::uint64_t map = 0x8080808080808080ULL;
                int count = 0;
                for (U i = 0; i < 8; ++i) if (mask & (1u << i)) {
                    map ^= std::uint64_t(0x80u | i) << (8 * count);
                    ++count;
                }
                indices[mask] = map;
                counts[mask] = u8_t(count);
            }
        }
    };

    static bool can_shuffle_bytes() {
#if defined(_MSC_VER)
        int registers[4]{};
        __cpuid(registers, 1);
        return (registers[2] & (1 << 9)) != 0;
#else
        return __builtin_cpu_supports("ssse3");
#endif
    }

    void plan(std::vector<Span>& spans, Words& fixed, int i, int end, int k) {
        const int begin_fill = input.prefix_len;
        const int end_fill = begin_fill + input.fill_len;
        while (i < end) {
            const int begin = i;
            if (k >= begin_fill && k < end_fill) {
                const int count = std::min(end - i, end_fill - k);
                spans.push_back({i, i + count, k - begin_fill, true});
                i += count;
                k += count;
                continue;
            }
            do {
                if (k == input.name_len) { fixed[i++] = 0; k = 0; }
                else {
                    fixed[i++] = Word(u8_t(k < begin_fill ? input.prefix[k] : input.suffix[k - end_fill])) * 0x0101010101010101ULL;
                    ++k;
                }
            } while (i < end && (k < begin_fill || k >= end_fill));
            spans.push_back({begin, i, 0, false});
        }
    }

    static Word pack_bytes(__m128i left, __m128i right) {
        const auto words = _mm_packs_epi32(left, right);
        return Word(_mm_cvtsi128_si64(_mm_packus_epi16(words, _mm_setzero_si128())));
    }

    template<bool Last = false>
    static void utf8(Fills& fills, int at, __m128i a, __m128i b) {
        const auto base = _mm_set1_epi32(0x4e00), mask = _mm_set1_epi32(63);
        const auto lead = _mm_set1_epi32(0xe0), cont = _mm_set1_epi32(0x80);
        const auto ar = _mm_srli_epi32(_mm_add_epi32(a, base), 6);
        const auto br = _mm_srli_epi32(_mm_add_epi32(b, base), 6);
        fills[at] = pack_bytes(_mm_or_si128(_mm_srli_epi32(ar, 6), lead), _mm_or_si128(_mm_srli_epi32(br, 6), lead));
        fills[at + 1] = pack_bytes(_mm_or_si128(_mm_and_si128(ar, mask), cont), _mm_or_si128(_mm_and_si128(br, mask), cont));
        // U+4E00 对齐 64；最后一个字的低六位固定为 0。
        if constexpr (Last) fills[at + 2] = 0x8080808080808080ULL;
        else fills[at + 2] = pack_bytes(_mm_or_si128(_mm_and_si128(a, mask), cont), _mm_or_si128(_mm_and_si128(b, mask), cont));
    }

    // 直接从八个 ID 生成按字节位置交错的 15 个键字，不先构造再拆散 UTF-8 字符串。
    void base_fill(Id id, Fills& fills) const {
        Id value[8];
        for (U lane = 0; lane < 8; ++lane) value[lane] = shuffle_id<false>(input, id + lane);
        const auto low = [&](int n) {
            return _mm_set_epi32(int(U(value[n + 3])), int(U(value[n + 2])), int(U(value[n + 1])), int(U(value[n])));
        };
        const auto high = [&](int n) {
            return _mm_set_epi32(int(value[n + 3] >> 32), int(value[n + 2] >> 32),
                                 int(value[n + 1] >> 32), int(value[n] >> 32));
        };
        const auto al = low(0), bl = low(4), ah = high(0), bh = high(4);
        const auto mask = _mm_set1_epi32(0x3fff), tail = _mm_set1_epi32(255);
        utf8(fills, 0, _mm_srli_epi32(ah, 18), _mm_srli_epi32(bh, 18));
        utf8(fills, 3, _mm_and_si128(_mm_srli_epi32(ah, 4), mask), _mm_and_si128(_mm_srli_epi32(bh, 4), mask));
        const auto middle = [&](auto lo, auto hi) {
            return _mm_and_si128(_mm_or_si128(_mm_srli_epi32(lo, 22), _mm_slli_epi32(hi, 10)), mask);
        };
        utf8(fills, 6, middle(al, ah), middle(bl, bh));
        utf8(fills, 9, _mm_and_si128(_mm_srli_epi32(al, 8), mask), _mm_and_si128(_mm_srli_epi32(bl, 8), mask));
        utf8<true>(fills, 12, _mm_slli_epi32(_mm_and_si128(al, tail), 6), _mm_slli_epi32(_mm_and_si128(bl, tail), 6));
    }

    template<bool Crop>
    static inline void exchange(Words& __restrict box, __m128i& __restrict j, int i, Word key) {
        // 同一顺序槽的八个字节在一个字里；游标按 lane 独立回绕。
        const Word a = box[i];
        j = _mm_add_epi8(j, _mm_add_epi8(_mm_cvtsi64_si128(static_cast<long long>(a)), _mm_cvtsi64_si128(static_cast<long long>(key))));
        const Word x = Word(_mm_cvtsi128_si64(j));
        auto* raw = reinterpret_cast<u8_t*>(box.data());
        auto* p0 = raw + ((x & 255u) << 3);
        auto* p1 = raw + (((x >> 5) & 2040u) | 1u);
        auto* p2 = raw + (((x >> 13) & 2040u) | 2u);
        auto* p3 = raw + (((x >> 21) & 2040u) | 3u);
        auto* p4 = raw + (((x >> 29) & 2040u) | 4u);
        auto* p5 = raw + (((x >> 37) & 2040u) | 5u);
        auto* p6 = raw + (((x >> 45) & 2040u) | 6u);
        auto* p7 = raw + (((x >> 53) & 2040u) | 7u);
        // 随机地址可能指回当前槽，完整交换须先收齐八个旧值再写回。
        // 第二轮离开保留区后，当前槽的顺序写回不再影响前缀或未来游标。
        if constexpr (!Crop) {
            box[i] = Word(*p0) | (Word(*p1) << 8) | (Word(*p2) << 16) | (Word(*p3) << 24)
                   | (Word(*p4) << 32) | (Word(*p5) << 40) | (Word(*p6) << 48) | (Word(*p7) << 56);
        }
        *p0 = u8_t(a);
        *p1 = u8_t(a >> 8);
        *p2 = u8_t(a >> 16);
        *p3 = u8_t(a >> 24);
        *p4 = u8_t(a >> 32);
        *p5 = u8_t(a >> 40);
        *p6 = u8_t(a >> 48);
        *p7 = u8_t(a >> 56);
    }

    template<bool Crop>
    static __m128i apply(Words& __restrict box, __m128i j, const Fills& __restrict fills, const std::vector<Span>& __restrict spans,
                      const Words& __restrict fixed) {
        for (const auto& s : spans) {
            const int end = s.end;
            if (s.varying) {
                int k = s.offset;
                for (int i = s.i; i < end; ++i, ++k) exchange<Crop>(box, j, i, fills[k]);
            } else {
                for (int i = s.i; i < end; ++i) exchange<Crop>(box, j, i, fixed[i]);
            }
        }
        return j;
    }

    // 只拆回保留的 96 槽，按原顺序交给 Q 属性压缩。
    template<int Base, int Shift>
    static void unweave(Boxes& out, int i, __m128i a, __m128i b, __m128i c, __m128i d) {
        const auto mask = _mm_set1_epi32(255);
        const auto ab = _mm_packs_epi32(_mm_and_si128(_mm_srli_epi32(a, Shift), mask), _mm_and_si128(_mm_srli_epi32(b, Shift), mask));
        const auto cd = _mm_packs_epi32(_mm_and_si128(_mm_srli_epi32(c, Shift), mask), _mm_and_si128(_mm_srli_epi32(d, Shift), mask));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out[Base + Shift / 8].data() + i), _mm_packus_epi16(ab, cd));
    }

    static unsigned lowest_bit(unsigned bits) {
#ifdef _MSC_VER
        unsigned long at;
        _BitScanForward(&at, bits);
        return unsigned(at);
#else
        return unsigned(__builtin_ctz(bits));
#endif
    }

    static __m128i property_window(const Box& box, int i) {
        const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(box.data() + i));
        const auto mul = _mm_set1_epi16(181);
        const auto low = _mm_and_si128(_mm_mullo_epi16(raw, mul), _mm_set1_epi16(255));
        const auto high = _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_epi16(raw, 8), mul), 8);
        // (181*x + 160)∈[89,217) 等价于 (181*x+71) 的最高位为零。
        return _mm_add_epi8(_mm_or_si128(low, high), _mm_set1_epi8(71));
    }

    static __m128i property_bytes(__m128i window) {
        return _mm_and_si128(_mm_add_epi8(window, _mm_set1_epi8(89)), _mm_set1_epi8(63));
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((target("ssse3")))
#endif
    static int gate_packed(const Box& box) {
        static constexpr Packing table;
        alignas(16) u8_t attrs[48]; // 压缩只消费前 31 项；两次 8 字节存储可越过已消费位置。
        U n = 0;
        for (int i = 0; i < limit; i += 16) {
            const auto window = property_window(box, i);
            const U valid = U(_mm_movemask_epi8(window)) ^ 65535u;
            const U a = valid & 255u, b = valid >> 8;
            const auto selectors = _mm_set_epi64x(static_cast<long long>(table.indices[b] | 0x0808080808080808ULL),
                                                  static_cast<long long>(table.indices[a]));
            const auto bytes = _mm_shuffle_epi8(property_bytes(window), selectors);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(attrs + n), bytes);
            n += table.counts[a];
            _mm_storel_epi64(reinterpret_cast<__m128i*>(attrs + n), _mm_srli_si128(bytes, 8));
            n += table.counts[b];
            if (n >= 31) {
                int props[7];
                return gate_level_of(attrs, props);
            }
        }
        return -1;
    }

    static int gate(const Box& box) {
        u8_t attrs[31], packed[16];
        int n = 0;
        for (int i = 0; i < limit; i += 16) {
            const auto window = property_window(box, i);
            unsigned valid = unsigned(_mm_movemask_epi8(window)) ^ 65535u;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(packed), property_bytes(window));
            while (valid) {
                attrs[n++] = packed[lowest_bit(valid)];
                if (n == 31) {
                    int props[7];
                    return gate_level_of(attrs, props);
                }
                valid &= valid - 1;
            }
        }
        return -1;
    }

    U exact(Id id) const {
        Name state{};
        init_candidate(state, input);
        char name[name_limit + 1];
        make_name(input, id, name);
        state.load_name(name, input.name_len);
        int props[7];
        return U(gate_level_of(state.name_base, props));
    }

public:
    explicit Front(const Launch& launch) : input(launch), ssse3(can_shuffle_bytes()) {
        const auto cursor = [&](int i) { return (input.name_len + i) % (input.name_len + 1); };
        for (int i = 0; i < 256; ++i) initial[i] = Word(input.prefix_box[i]) * 0x0101010101010101ULL;
        plan(first, fixed_first, input.i_pre, 256, input.j_pre);
        plan(head, fixed_second, 0, limit, input.name_len);
        plan(tail, fixed_second, limit, 256, cursor(limit));
    }

    void levels(Id id, U (&out)[front_width]) const {
        Words box = initial;
        Fills fills;
        if (!input.fill_slots_count && input.fill_len == encoded_width) base_fill(id, fills);
        else {
            char names[front_width][name_limit];
            for (U lane = 0; lane < front_width; ++lane) make_fill(input, id + lane, names[lane]);
            for (int k = 0; k < input.fill_len; ++k) {
                fills[k] = Word(u8_t(names[0][k])) | (Word(u8_t(names[1][k])) << 8)
                         | (Word(u8_t(names[2][k])) << 16) | (Word(u8_t(names[3][k])) << 24)
                         | (Word(u8_t(names[4][k])) << 32) | (Word(u8_t(names[5][k])) << 40)
                         | (Word(u8_t(names[6][k])) << 48) | (Word(u8_t(names[7][k])) << 56);
            }
        }
        auto j = _mm_set1_epi8(static_cast<char>(input.s_pre));
        (void)apply<false>(box, j, fills, first, fixed_first);
        j = apply<false>(box, _mm_setzero_si128(), fills, head, fixed_second);
        (void)apply<true>(box, j, fills, tail, fixed_second);
        Boxes boxes;
        for (int i = 0; i < limit; i += 16) {
            const auto get = [&](int k) {
                return _mm_loadu_si128(reinterpret_cast<const __m128i*>(box.data() + i + 2 * k));
            };
            const auto x0 = get(0), x1 = get(1), x2 = get(2), x3 = get(3);
            const auto x4 = get(4), x5 = get(5), x6 = get(6), x7 = get(7);
            const auto lo = [](__m128i x) { return _mm_shuffle_epi32(x, 0x88); };
            const auto hi = [](__m128i x) { return _mm_shuffle_epi32(x, 0xdd); };
            const auto a = _mm_unpacklo_epi64(lo(x0), lo(x1)), b = _mm_unpacklo_epi64(lo(x2), lo(x3));
            const auto c = _mm_unpacklo_epi64(lo(x4), lo(x5)), d = _mm_unpacklo_epi64(lo(x6), lo(x7));
            const auto e = _mm_unpacklo_epi64(hi(x0), hi(x1)), f = _mm_unpacklo_epi64(hi(x2), hi(x3));
            const auto g = _mm_unpacklo_epi64(hi(x4), hi(x5)), h = _mm_unpacklo_epi64(hi(x6), hi(x7));
            unweave<0, 0>(boxes, i, a, b, c, d);
            unweave<0, 8>(boxes, i, a, b, c, d);
            unweave<0, 16>(boxes, i, a, b, c, d);
            unweave<0, 24>(boxes, i, a, b, c, d);
            unweave<4, 0>(boxes, i, e, f, g, h);
            unweave<4, 8>(boxes, i, e, f, g, h);
            unweave<4, 16>(boxes, i, e, f, g, h);
            unweave<4, 24>(boxes, i, e, f, g, h);
        }
        for (U lane = 0; lane < front_width; ++lane) {
            const int level = ssse3 ? gate_packed(boxes[lane]) : gate(boxes[lane]);
            out[lane] = level >= 0 ? U(level) : exact(id + lane);
        }
    }
};
} // namespace pbb_cpu
