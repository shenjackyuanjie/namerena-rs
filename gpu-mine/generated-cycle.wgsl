// 由宿主按当前模板字节展开，非特化路径仍走原着色器。
const DIRECT_CYCLE: bool = true;
const CYCLE_FIRST: u32 = /* KSA_FIRST */;
const CYCLE_LEN: u32 = /* KSA_LEN */;

fn segment_point(point: u32) -> u32 {
    return (0xe0u | (point >> 12u)) | ((0x80u | ((point >> 6u) & 63u)) << 8u)
        | ((0x80u | (point & 63u)) << 16u);
}

fn segment_step(lane: u32, i: u32, j: u32, key: u32) -> u32 {
    let word = values[(i >> 2u) * LANES + lane];
    let current = extractBits(word, (i & 3u) * 8u, 8u);
    let next = (j + key + current) & 255u;
    swap_packed(lane, i, next, word, current);
    return next;
}

fn cycle_suffix(lane: u32, i: u32, j: u32) -> u32 {
    var cursor = j;
/* KSA_SUFFIX */
    return cursor;
}

fn cycle_tail(p: u32, chunks: array<u32, 6>) -> u32 {
    // p 是去掉前导零后的键下标。若两种周期尾巴都不进入生成段，直接取字面。
    if ((256u % CYCLE_LEN) <= CYCLE_FIRST && (256u % (CYCLE_LEN + 7u)) <= CYCLE_FIRST) {
        return name_bytes[p];
    }
    let first = CYCLE_FIRST - 1u;
    if (p >= first && p < first + 18u) {
        let byte = p - first;
        return (chunks[byte / 3u] >> ((byte % 3u) * 8u)) & 255u;
    }
    if (p < CYCLE_LEN - 1u) { return name_bytes[p]; }
    return shadow_byte(p - CYCLE_LEN + 1u);
}

fn cycle_slot(lane: u32, i: u32, j: u32, chunks: array<u32, 6>) -> u32 {
    var cursor = j;
    cursor = segment_step(lane, i + 0u, cursor, (chunks[0] >> 0u) & 255u);
    cursor = segment_step(lane, i + 1u, cursor, (chunks[0] >> 8u) & 255u);
    cursor = segment_step(lane, i + 2u, cursor, (chunks[0] >> 16u) & 255u);
    cursor = segment_step(lane, i + 3u, cursor, (chunks[1] >> 0u) & 255u);
    cursor = segment_step(lane, i + 4u, cursor, (chunks[1] >> 8u) & 255u);
    cursor = segment_step(lane, i + 5u, cursor, (chunks[1] >> 16u) & 255u);
    cursor = segment_step(lane, i + 6u, cursor, (chunks[2] >> 0u) & 255u);
    cursor = segment_step(lane, i + 7u, cursor, (chunks[2] >> 8u) & 255u);
    cursor = segment_step(lane, i + 8u, cursor, (chunks[2] >> 16u) & 255u);
    cursor = segment_step(lane, i + 9u, cursor, (chunks[3] >> 0u) & 255u);
    cursor = segment_step(lane, i + 10u, cursor, (chunks[3] >> 8u) & 255u);
    cursor = segment_step(lane, i + 11u, cursor, (chunks[3] >> 16u) & 255u);
    cursor = segment_step(lane, i + 12u, cursor, (chunks[4] >> 0u) & 255u);
    cursor = segment_step(lane, i + 13u, cursor, (chunks[4] >> 8u) & 255u);
    cursor = segment_step(lane, i + 14u, cursor, (chunks[4] >> 16u) & 255u);
    cursor = segment_step(lane, i + 15u, cursor, (chunks[5] >> 0u) & 255u);
    cursor = segment_step(lane, i + 16u, cursor, (chunks[5] >> 8u) & 255u);
    cursor = segment_step(lane, i + 17u, cursor, (chunks[5] >> 16u) & 255u);
    return cursor;
}

fn cycle_shadow(lane: u32, i: u32, j: u32) -> u32 {
    var cursor = j;
    cursor = segment_step(lane, i + 0u, cursor, 63u);
    cursor = segment_step(lane, i + 1u, cursor, 115u);
    cursor = segment_step(lane, i + 2u, cursor, 104u);
    cursor = segment_step(lane, i + 3u, cursor, 97u);
    cursor = segment_step(lane, i + 4u, cursor, 100u);
    cursor = segment_step(lane, i + 5u, cursor, 111u);
    cursor = segment_step(lane, i + 6u, cursor, 119u);
    return cursor;
}

fn generated_cycle_ksa(lane: u32, index: u32, with_shadow: bool) {
    let low = params.pad0 + index;
    let high = params.pad1 + select(0u, 1u, low < params.pad0);
    let chunks = array<u32, 6>(
        segment_point(0x4e00u + ((high >> 18u) & 0x3fffu)),
        segment_point(0x4e00u + ((high >> 4u) & 0x3fffu)),
        segment_point(0x4e00u + (((high << 10u) | (low >> 22u)) & 0x3fffu)),
        segment_point(0x4e00u + ((low >> 8u) & 0x3fffu)),
        segment_point(0x4e00u + ((low & 255u) << 6u)),
        segment_point(0x3d01u),
    );
    let period = CYCLE_LEN + select(0u, 7u, with_shadow);
    for (var round = 0u; round < 2u; round += 1u) {
        var i = 0u;
        var cursor = 0u;
        if (round == 0u) {
            cursor = cycle_slot(lane, CYCLE_FIRST, names[1u].offset, chunks);
            cursor = cycle_suffix(lane, 0u, cursor);
            if (with_shadow) { cursor = cycle_shadow(lane, CYCLE_LEN, cursor); }
            i = period;
        }
        for (; i + period <= 256u; i += period) {
/* KSA_PREFIX */
            cursor = cycle_slot(lane, i + CYCLE_FIRST, cursor, chunks);
            cursor = cycle_suffix(lane, i, cursor);
            if (with_shadow) { cursor = cycle_shadow(lane, i + CYCLE_LEN, cursor); }
        }
        if (i < 256u) {
            cursor = segment_step(lane, i, cursor, 0u);
            i += 1u;
        }
        for (var p = 0u; i < 256u; p += 1u) {
            cursor = segment_step(lane, i, cursor, cycle_tail(p, chunks));
            i += 1u;
        }
    }
}
