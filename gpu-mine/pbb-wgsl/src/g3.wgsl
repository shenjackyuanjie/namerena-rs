// 输入来自原版整数 Q 门。GPU 只允许剔除不可能过 G3 的候选，最终仍由原版 CPU 精算。
const SCALE: u32 = 65536u;
const FREE: u32 = 0x80000000u;

fn get_byte(lane: u32, at: u32) -> u32 {
    return extractBits(sbox[(at >> 2u) * LANES + lane], (at & 3u) * 8u, 8u);
}

fn prga(lane: u32, i: u32, j: ptr<function, u32>) -> u32 {
    let ai = (i >> 2u) * LANES + lane;
    let ash = (i & 3u) * 8u;
    let aw = sbox[ai];
    let a = extractBits(aw, ash, 8u);
    let b_at = (*j + a) & 255u;
    *j = b_at;
    let bi = (b_at >> 2u) * LANES + lane;
    let bsh = (b_at & 3u) * 8u;
    let bw = sbox[bi];
    let b = extractBits(bw, bsh, 8u);
    // 两个槽在同一共享字时，第二次插入必须看到第一次的修改。
    let modified = insertBits(aw, b, ash, 8u);
    sbox[ai] = modified;
    sbox[bi] = insertBits(select(bw, modified, ai == bi), a, bsh, 8u);
    return get_byte(lane, (a + b) & 255u);
}

struct Skills {
    ids: array<u32, 40>,
    freq: array<u32, 16>,
}

fn skills(lane: u32) -> Skills {
    var result: Skills;
    // 181 与 256 互素，完整 S-box 必有 128 个有效项。逆扫只取技能频率需要的末 68 项。
    var left = 128u;
    var minimum = 63u;
    var boost: array<u32, 2>;
    for (var w = 64u; w > 0u && left > 60u; w -= 1u) {
        let word = sbox[(w - 1u) * LANES + lane];
        for (var b = 4u; b > 0u && left > 60u; b -= 1u) {
            let u = ((word >> ((b - 1u) * 8u)) * 181u + 71u) & 255u;
            if u < 128u {
                minimum = min(minimum, (u + 89u) & 63u);
                left -= 1u;
                if left >= 64u {
                    if (left & 3u) == 0u {
                        result.freq[(left - 64u) / 4u] = max(minimum, 10u) - 10u;
                        minimum = 63u;
                    }
                } else if (left & 1u) == 0u {
                    boost[(left - 60u) / 2u] = minimum;
                    minimum = 63u;
                }
            }
        }
    }
    for (var i = 0u; i < 40u; i += 1u) { result.ids[i] = i; }
    var j = 0u;
    var mix_cursor = 0u;
    for (var k = 0u; k < 80u; k += 1u) {
        let i = k % 40u;
        let hi = prga(lane, 2u * k + 1u, &j);
        let lo = prga(lane, 2u * k + 2u, &j);
        let old = result.ids[i];
        // ((raw % 40) + mix_cursor + old) % 40 与合并取模等价。
        mix_cursor = (mix_cursor + ((hi << 8u) | lo) + old) % 40u;
        result.ids[i] = result.ids[mix_cursor];
        result.ids[mix_cursor] = old;
    }
    var last = 16u;
    for (var i = 0u; i < 16u; i += 1u) {
        let skill = result.ids[i];
        if skill >= 35u { result.freq[i] = 0u; }
        else if skill < 25u && result.freq[i] > 0u { last = i; }
    }
    if last != 16u { result.freq[last] *= 2u; }
    for (var i = 14u; i < 16u; i += 1u) {
        if i != last { result.freq[i] += min(result.freq[i], boost[i - 14u]); }
    }
    return result;
}

fn weights(i: u32) -> vec3<u32> {
    let w = setup.g3_weights[i];
    return (vec3(w) >> vec3(0u, 5u, 10u)) & vec3(31u);
}

fn decay_percent(skill: u32) -> u32 {
    switch skill {
        case 9u, 16u: { return 30u; }
        case 18u: { return 35u; }
        case 19u, 23u: { return 60u; }
        case 20u, 22u: { return 70u; }
        default: { return 100u; }
    }
}

fn may_pass_g3(base: ptr<function, array<u32, 31>>, lane: u32) -> bool {
    // 原版两套 HP 比较网络的中间四项和在构建时经零一原理校验。
    var point = weights(0u) * sorted_hp(base);
    for (var i = 1u; i < 8u; i += 1u) {
        point += weights(i) * (u32(median(base, 7u + 3u * i)) + 36u);
    }
    point *= SCALE;
    var skill = skills(lane);
    var has_shield = false;
    for (var i = 0u; i < 16u; i += 1u) {
        has_shield = has_shield || (skill.ids[i] == 29u && skill.freq[i] != 0u);
    }
    var zd = SCALE;
    var kill = SCALE;
    for (var i = 0u; i < 16u; i += 1u) {
        let sk = skill.ids[i];
        let f = skill.freq[i];
        if f == 0u { continue; }
        var value = SCALE * f;
        if sk < 25u {
            value = zd * f;
            zd = (zd * (12800u - f * decay_percent(sk)) + 12799u) / 12800u;
        } else if sk == 31u || sk == 32u {
            value = kill * f;
            kill = (kill * (128u - f) + 127u) / 128u;
        } else if sk == 29u {
            // 将平方除法向上取整为 16.16；另一分支是精确整数。
            if f <= 70u { value = (SCALE * f * f + 69u) / 70u; }
            else { value = SCALE * (2u * f - 70u); }
        }
        point += weights(select(sk + 8u, 45u, sk == 18u && has_shield)) * value;
        if sk == 34u { point += weights(44u) * SCALE; }
    }
    // 所有系数非负、暗影权重为零。向上舍入状态给出数学上界，阈值还额外减一单位覆盖 CPU f64 舍入。
    return any(point >= vec3(setup.g3_minimum[0], setup.g3_minimum[1], setup.g3_minimum[2]));
}

@compute @workgroup_size(LANES)
fn secondary(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(num_workgroups) groups: vec3<u32>) {
    let total = atomicLoad(&output.count);
    for (var slot = id.x; slot < total; slot += groups.x * LANES) {
        let entry = output.entries[slot];
        let i = entry & ~FREE;
        var pass_gate = (entry & FREE) != 0u;
        if !pass_gate {
            var words = name_words(i);
            load_name(lane, &words, false);
            var base: array<u32, 31>;
            _ = collect_properties(lane, 64u, &base);
            pass_gate = may_pass_g3(&base, lane);
        }
        if pass_gate { selected.entries[atomicAdd(&selected.count, 1u)] = i; }
    }
}
