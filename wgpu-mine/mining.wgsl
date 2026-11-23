// 一个 invocation（GPU 并行调用）处理一个候选，lane 是组内编号；S-box 是 256 字节的排列状态。
// 本文件不含跨 lane barrier，因此尾批与属性筛选均可直接 return。
// host 每批清零 Output 的 16B header，records 至少分配 count 个 176B 槽。
// 队伍 KSA 状态或生成模式的公共前缀快照；packed 时四字节一字，wide 时一字节一字。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>;

struct NameInfo { // 上传模式记录名字在字节流中的偏移和 UTF-8 字节长度；生成模式复用这些字段。
    offset: u32,
    length: u32,
}
@group(0) @binding(1) var<storage, read> names: array<NameInfo>; // 绑定 1：只读候选元数据；由宿主按绑定号提供缓冲区。
// 上传模式为紧凑UTF8；GENERATED模式为逐u32模板映射，offset复用为prefix_steps。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>;

// 严格 44 个 32bit 字，align=4、size/stride=176；不增加 vec3 对齐填充。
struct Record {
    index: u32,                       // 字节偏移 0；本批候选下标，宿主据此还原名字。
    props: array<u32, 8>,              // 字节偏移 4；八个属性值。
    skill_ids: array<u32, 16>,         // 字节偏移 36；前 16 个技能编号。
    skill_freqs: array<u32, 16>,       // 字节偏移 100；对应技能频率。
    shadow_numerator: i32,            // 字节偏移 164；影子特征的整数分子。
    xp: f32,                          // 字节偏移 168；GPU 计算的 XP 近似分数。
    xd: f32,                          // 字节偏移 172；GPU 计算的 XD 近似分数。
}
struct Output { // 输出头的计数器供多个 invocation 并发更新，后面接变长记录数组。
    count: atomic<u32>, // 已领取的记录槽数。
    prop_pass: atomic<u32>, // 进入技能阶段的候选数。
    reserved: u32, // 保留原输出头布局；评分数与 prop_pass 完全相同。
    shadow_count: atomic<u32>, // 计算影子的候选数。
    records: array<Record>,           // 字节偏移 16
}
@group(0) @binding(3) var<storage, read_write> output: Output; // 绑定 3：可读写输出；每个命中候选占一个记录槽。

// 八个 u32，uniform 共 32B；host 将 prop_threshold 限制到 3*p 不溢出。
struct Params {
    count: u32, // 批内真实候选数，派发补齐的 invocation 不能越界。
    prop_threshold: u32, // 属性粗筛阈值。
    xp_threshold: u32, // XP 粗筛阈值。
    xd_threshold: u32, // XD 粗筛阈值。
    capture_all: u32, // 非零时保留所有候选，跳过两次淘汰。
    pad0: u32, // GENERATED: 起始ID低32bit
    pad1: u32, // GENERATED: 起始ID高32bit
    pad2: u32, // 填充，保持宿主和着色器布局一致。
}
@group(0) @binding(4) var<uniform> params: Params; // 绑定 4：宿主上传的小型只读参数块。
// weights[0] = vec2(BASE, BASE_QD)；weights[k+1] 对应第 k 项 XP/XD 系数。
// 一共 1035 个 vec2<f32>，storage stride=8，共 8280B。
@group(0) @binding(5) var<storage, read> weights: array<vec2<f32>>;

// 宿主替换编译期常量：LANES 为 32/64/128；packed 每候选 256B、wide 为 1024B，布局分支由常量决定。
const LANES: u32 = 32u; // 每个工作组的 invocation 数；此处是模板值，宿主按配置替换。
const WORDS: u32 = 8192u; // 工作组共享数组长度，以 u32 为单位。
const PACKED: bool = false; // true 时每个 u32 打包四个 S-box 字节；false 时每槽一个 u32。
const GENERATED: bool = false; // true 时从 ID 生成名字；false 时读取上传的名字。
const TILED: bool = false; // 大批次跨两行，组内布局和每候选计算不变。
const COMPACT_MODEL: bool = false; // true 时仅遍历非零特征；false 时遍历全部特征。
var<workgroup> values: array<u32, WORDS>; // 同组共享存储按槽转置，各 lane 独占一列，不依赖跨线程同步。

fn read_sbox(lane: u32, slot: u32) -> u32 { // 读取当前 lane 的 S-box 槽；lane 是组内编号，slot 是 0 至 255 的状态下标。
    if (PACKED) {
        return (values[(slot / 4u) * LANES + lane] >> ((slot % 4u) * 8u)) & 255u; // 先选打包字，再右移取出槽内字节；& 255 保留低八位。
    }
    return values[slot * LANES + lane]; // 宽字布局：相同槽的不同 lane 在内存中相邻。
}

fn write_sbox(lane: u32, slot: u32, value: u32) { // 写入一个状态字节，保留同一打包字中的其他槽。
    if (PACKED) {
        let word = (slot / 4u) * LANES + lane;
        let shift = (slot % 4u) * 8u;
        // 每 lane 独占一列。同一 word 的两个槽交换时，第二次必须读更新后的 word。
        values[word] = (values[word] & ~(255u << shift)) | (value << shift); // 先用掩码清空目标字节，再按位或写入新值。
    } else {
        values[slot * LANES + lane] = value;
    }
}

// 直接复用两次共享内存读取的打包字；若两个槽同字，第二次写入使用更新后的字。
fn swap_packed(lane: u32, a: u32, b: u32, word_a: u32, current: u32) -> u32 {
    let index_a = (a >> 2u) * LANES + lane;
    let index_b = (b >> 2u) * LANES + lane;
    let shift_a = (a & 3u) * 8u;
    let shift_b = (b & 3u) * 8u;
    let word_b = values[index_b];
    let other = (word_b >> shift_b) & 255u;
    // XOR 差量只改目标槽：current ^ delta == other，反过来也成立。
    let delta = current ^ other;
    let update_a = word_a ^ (delta << shift_a);
    values[index_a] = update_a;
    values[index_b] = select(word_b, update_a, index_a == index_b) ^ (delta << shift_b);
    return other;
}

fn restore_team(lane: u32) { // 把宿主准备的队伍状态或前缀快照复制到本 lane。
    if (PACKED) {
        // 宿主已经打包前缀；一次读 4 个字，到对应的四个槽行。
        for (var i = 0u; i < 16u; i += 1u) {
            let v = team_state[i];
            let row = i * 4u * LANES + lane;
            values[row] = v.x;
            values[row + LANES] = v.y;
            values[row + 2u * LANES] = v.z;
            values[row + 3u * LANES] = v.w;
        }
    } else {
        for (var i = 0u; i < 256u; i += 1u) {
            values[i * LANES + lane] = team_state[i / 4u][i % 4u];
        }
    }
}

// 以原始名字字节拼接 ?shadow；绝不解析名字内的 @，也不改 raw team。
fn shadow_byte(index: u32) -> u32 {
    switch index {
        case 0u: { return 63u; }   // ?
        case 1u: { return 115u; }  // s
        case 2u: { return 104u; }  // h
        case 3u: { return 97u; }   // a
        case 4u: { return 100u; }  // d
        case 5u: { return 111u; }  // o
        default: { return 119u; } // w，唯一剩余的合法索引为 6
    }
}

// KEY_BEGIN
fn prepare_generated(index: u32) {} // 上传模式无需生成名字；生成模式下宿主替换整个标记区间。

// k=0 是编码前导零；k>0 对应 name 或 name+?shadow 的第 k-1 个字节。
fn key_byte(info: NameInfo, k: u32) -> u32 {
    if (k == 0u) { return 0u; }
    let byte_index = k - 1u;
    if (byte_index < info.length) {
        let address = info.offset + byte_index; // 上传模式：偏移和下标都以字节为单位。
        return (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u; // 每个输入字含四字节，除以 4 选字、余数选字节。
    }
    return shadow_byte(byte_index - info.length);
}

// KEY_END

// KSA（密钥调度）依据名字反复交换 S-box；名字最长 256B，追加 shadow 后允许 263B。
// GENERATED_CYCLE_BEGIN
const DIRECT_CYCLE: bool = false;
fn generated_cycle_ksa(lane: u32, index: u32, with_shadow: bool) {}
// GENERATED_CYCLE_END

fn name_ksa(lane: u32, info: NameInfo, index: u32, with_shadow: bool) {
    if (DIRECT_CYCLE) {
        generated_cycle_ksa(lane, index, with_shadow);
        return;
    }
    let key_length = info.length + select(0u, 7u, with_shadow); // 名字字节长度；影子路径追加七字节 ?shadow。密钥还另有前导零。
    for (var round = 0u; round < 2u; round += 1u) { // 连续执行两轮 KSA，状态保留，游标每轮重置。
        var s = 0u;
        var k = 0u;
        var first = 0u;
        if (GENERATED && round == 0u) { first = info.offset; k = first; s = names[1u].offset; } // 生成模式第一轮从宿主预计算的公共前缀快照继续；第二轮从零开始。
        for (var i = first; i < 256u; i += 1u) {
            if (PACKED) {
                let word = values[(i >> 2u) * LANES + lane];
                let current = (word >> ((i & 3u) * 8u)) & 255u;
                s = (s + key_byte(info, k) + current) & 255u;
                swap_packed(lane, i, s, word, current);
            } else {
                let current = read_sbox(lane, i);
                s = (s + key_byte(info, k) + current) & 255u;
                let other = read_sbox(lane, s);
                write_sbox(lane, i, other);
                write_sbox(lane, s, current);
            }
            if (k == key_length) { k = 0u; } else { k += 1u; } // k 在前导零和全部名字字节间循环。
        }
    }
}

// Sbox 是完整排列，仿射映射亦为排列，[89,217) 恰好命中 128 项。
// 按原 Sbox 顺序稳定压缩，不排序，不修改 PRGA 使用的 Sbox。
fn collect_base(lane: u32, base: ptr<function, array<u32, 128>>) {
    var count = 0u;
    for (var i = 0u; i < 256u; i += 1u) {
        let mapped = (read_sbox(lane, i) * 181u + 160u) & 255u; // 对排列做模 256 仿射映射，供后续提取属性与技能数据。
        if (mapped >= 89u && mapped < 217u) {
            (*base)[count] = mapped & 63u; // 保留映射值低六位，按原槽顺序放入 base。
            count += 1u;
        }
    }
}

// 影子只消费八围所需的最早 31 项，技能仍依赖主号的完整 128 项。
fn collect_shadow(lane: u32, base: ptr<function, array<u32, 128>>) {
    var count = 0u;
    for (var i = 0u; i < 256u && count < 31u; i += 1u) {
        let mapped = (read_sbox(lane, i) * 181u + 160u) & 255u;
        if (mapped >= 89u && mapped < 217u) {
            (*base)[count] = mapped & 63u;
            count += 1u;
        }
    }
}

fn median3(x: u32, y: u32, z: u32) -> u32 { // 三个值的中位数，不需要对整个数组排序。
    return max(min(x, y), min(max(x, y), z));
}

fn properties(base: ptr<function, array<u32, 128>>) -> array<u32, 8> {
    // 借鉴 PBB 的固定 compare-exchange 网络，避免同组候选按不同次数执行插入排序。
    var h0 = (*base)[0];
    var h1 = (*base)[1];
    var h2 = (*base)[2];
    var h3 = (*base)[3];
    var h4 = (*base)[4];
    var h5 = (*base)[5];
    var h6 = (*base)[6];
    var h7 = (*base)[7];
    var h8 = (*base)[8];
    var h9 = (*base)[9];
    let low0 = min(h0, h5); h5 = max(h0, h5); h0 = low0;
    let low1 = min(h1, h6); h6 = max(h1, h6); h1 = low1;
    let low2 = min(h2, h7); h7 = max(h2, h7); h2 = low2;
    let low3 = min(h3, h8); h8 = max(h3, h8); h3 = low3;
    let low4 = min(h4, h9); h9 = max(h4, h9); h4 = low4;
    let low5 = min(h0, h3); h3 = max(h0, h3); h0 = low5;
    let low6 = min(h1, h4); h4 = max(h1, h4); h1 = low6;
    let low7 = min(h5, h8); h8 = max(h5, h8); h5 = low7;
    let low8 = min(h6, h9); h9 = max(h6, h9); h6 = low8;
    let low9 = min(h0, h2); h2 = max(h0, h2); h0 = low9;
    let low10 = min(h3, h6); h6 = max(h3, h6); h3 = low10;
    let low11 = min(h7, h9); h9 = max(h7, h9); h7 = low11;
    let low12 = min(h0, h1); h1 = max(h0, h1); h0 = low12;
    let low13 = min(h2, h4); h4 = max(h2, h4); h2 = low13;
    let low14 = min(h5, h7); h7 = max(h5, h7); h5 = low14;
    let low15 = min(h8, h9); h9 = max(h8, h9); h8 = low15;
    let low16 = min(h1, h2); h2 = max(h1, h2); h1 = low16;
    let low17 = min(h3, h5); h5 = max(h3, h5); h3 = low17;
    let low18 = min(h4, h6); h6 = max(h4, h6); h4 = low18;
    let low19 = min(h7, h8); h8 = max(h7, h8); h7 = low19;
    let low20 = min(h1, h3); h3 = max(h1, h3); h1 = low20;
    let low21 = min(h2, h5); h5 = max(h2, h5); h2 = low21;
    let low22 = min(h4, h7); h7 = max(h4, h7); h4 = low22;
    let low23 = min(h6, h8); h8 = max(h6, h8); h6 = low23;
    let low24 = min(h2, h3); h3 = max(h2, h3); h2 = low24;
    let low25 = min(h6, h7); h7 = max(h6, h7); h6 = low25;
    let low26 = min(h3, h4); h4 = max(h3, h4); h3 = low26;
    let low27 = min(h5, h6); h6 = max(h5, h6); h5 = low27;
    let low28 = min(h4, h5); h5 = max(h4, h5); h4 = low28;
    var props: array<u32, 8>;
    props[0] = 154u + h3 + h4 + h5 + h6;
    for (var i = 1u; i < 8u; i += 1u) {
        let start = 10u + (i - 1u) * 3u; // 其余七项每项使用连续三个 base 值的中位数。
        props[i] = median3((*base)[start], (*base)[start + 1u], (*base)[start + 2u]) + 36u;
    }
    return props;
}

fn prga_byte(lane: u32, a: ptr<function, u32>, b: ptr<function, u32>) -> u32 { // PRGA 每次更新同一 lane 的状态并返回一个伪随机字节；a、b 通过指针延续。
    *a = (*a + 1u) & 255u; // & 255 让状态游标在 0 至 255 间循环。
    if (PACKED) {
        let word = values[(*a >> 2u) * LANES + lane];
        let current = (word >> ((*a & 3u) * 8u)) & 255u;
        *b = (*b + current) & 255u;
        let other = swap_packed(lane, *a, *b, word, current);
        return read_sbox(lane, (current + other) & 255u);
    }
    let current = read_sbox(lane, *a);
    *b = (*b + current) & 255u;
    let other = read_sbox(lane, *b);
    write_sbox(lane, *a, other);
    write_sbox(lane, *b, current);
    // 交换前后两数之和不变，避免再次读取两个随机共享槽。
    return read_sbox(lane, (current + other) & 255u);
}

fn finite_score(value: f32) -> bool {
    // 位级检测同时覆盖 NaN 和正负 Inf，避免比较 NaN 时静默丢弃记录。
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

@compute @workgroup_size(LANES) // 计算入口：每组 LANES 个 invocation，由宿主按候选数向上取整派发。
fn mine(@builtin(global_invocation_id) position: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) { // 全局 x 区分候选，lane 是当前工作组内的编号。
    let index = select(position.x, position.x + position.y * grid.x * LANES, TILED);
    if (index >= params.count) { return; } // 尾组中多派发的 invocation 直接退出。
    let info = names[select(index, 0u, GENERATED)]; // 上传模式取当前候选；生成模式所有候选共用 names[0] 模板信息。
    if (GENERATED && !DIRECT_CYCLE) { prepare_generated(index); } // 仅生成模式按批内下标生成当前名字的编码缓存。

    restore_team(lane);
    name_ksa(lane, info, index, false);
    var base: array<u32, 128>;
    collect_base(lane, &base);
    var props = properties(&base);

    var prop_numerator = props[0]; // 用整数分子表示 HP/3 加其余属性之和，避免提前做浮点除法。
    for (var i = 1u; i < 8u; i += 1u) {
        prop_numerator += 3u * props[i];
    }
    if (params.capture_all == 0u && prop_numerator <= 3u * params.prop_threshold) { // 非 capture_all 时仅保留属性分子严格大于三倍阈值的候选。
        return;
    }
    // capture_all 模式统计所有进入技能阶段者，即本批实际候选数。
    atomicAdd(&output.prop_pass, 1u); // 原子加法让并发候选安全统计，不保证执行顺序。

    var skill_ids: array<u32, 40>; // 初始化 40 个技能编号；洗牌后只计算前 16 个的频率。
    var skill_freqs: array<u32, 16>;
    for (var i = 0u; i < 40u; i += 1u) {
        skill_ids[i] = i;
    }
    // 两轮共同保留 a/b/s；每轮 40 项，每项消耗两个 PRGA 字节，共 160B。
    var a = 0u;
    var b = 0u;
    var s = 0u;
    for (var round = 0u; round < 2u; round += 1u) {
        for (var i = 0u; i < 40u; i += 1u) {
            let high = prga_byte(lane, &a, &b);
            let low = prga_byte(lane, &a, &b);
            // s、余数、编号均小于 40；原先的和不会触发 u8 截断。
            s = (((high << 8u) | low) + s + skill_ids[i]) % 40u;
            let current = skill_ids[i];
            skill_ids[i] = skill_ids[s];
            skill_ids[s] = current;
        }
    }

    var last = -1i; // last 追踪最后一个满足附加条件的有效技能位置。
    for (var i = 0u; i < 16u; i += 1u) {
        let start = 64u + i * 4u;
        let p = min(min(base[start], base[start + 1u]), min(base[start + 2u], base[start + 3u])); // 四个 base 值取最小值，决定该技能的基础频率。
        if (p > 10u && skill_ids[i] < 35u) { // p 须大于 10 且技能编号小于 35 才有效。
            skill_freqs[i] = p - 10u;
            if (p < 35u) { last = i32(i); } // 记录最后一个 p 小于 35 的有效位置，供后续翻倍。
        } else {
            skill_freqs[i] = 0u;
        }
    }
    if (last >= 0i) { // 没有符合条件的技能时不做翻倍。
        skill_freqs[u32(last)] *= 2u;
    }
    // last=14/15 时该槽已翻倍，不再叠加自身的尾槽加成；另一槽照常。
    if (last != 14i && skill_freqs[14] != 0u) {
        skill_freqs[14] += min(min(base[60], base[61]), skill_freqs[14]);
    }
    if (last != 15i && skill_freqs[15] != 0u) {
        skill_freqs[15] += min(min(base[62], base[63]), skill_freqs[15]);
    }

    var features: array<f32, 44>; // 44 个特征：前 8 项属性、后续技能频率和影子特征；未赋值项为零。
    for (var i = 0u; i < 8u; i += 1u) {
        features[i] = f32(props[i]);
    }
    for (var i = 0u; i < 16u; i += 1u) {
        if (skill_freqs[i] != 0u) {
            features[skill_ids[i] + 8u] = f32(skill_freqs[i]); // 技能编号决定特征下标，而非洗牌后的位置。
        }
    }

    var shadow_numerator = 0i;
    if (features[32] > 0.0) { // 特征 32 对应技能编号 24，只有它有效时才计算影子。
        atomicAdd(&output.shadow_count, 1u);
        // 技能已结束，直接复用该 lane 的 Sbox/base，原 props/ids/freq 保留。
        restore_team(lane);
        name_ksa(lane, info, index, true);
        collect_shadow(lane, &base);
        var shadow_props = properties(&base);
        var raw_sum = 0u;
        for (var i = 1u; i < 8u; i += 1u) {
            raw_sum += shadow_props[i] - 36u;
        }
        shadow_numerator = i32(shadow_props[0] + 3u * raw_sum) // 以整数分子保存影子属性组合，宿主可据此重建精确值。
            - 9i * i32(shadow_props[7] - 36u) - 630i;
        features[43] = f32(shadow_numerator) / 3.0 * features[32] / 100.0; // 除以 3 并按技能频率缩放，得到影子特征。
    }
    if (features[42] > 0.0) { // 技能编号 34 有效时另加 20。
        features[42] += 20.0;
    }

    // 与 poly 完全相同的 44 个线性项及 i=0..43,j=i..43 二次项次序。
    // 不构造 1034 项多项式数组；零特征跳过乘加但仍推进权重索引。
    var score = weights[0]; // vec2 同时累加 XP 和 XD，从模型常数项开始。
    if (COMPACT_MODEL) {
        // 最多8属性+16非零技能+1影子=25项；保持原特征索引顺序，32槽留余量。
        var active_indices: array<u32, 32>;
        var active_count = 0u;
        for (var original = 0u; original < 44u; original += 1u) {
            let value = features[original];
            if (value != 0.0) {
                active_indices[active_count] = original; // 保存原始下标，压缩后仍可索引原模型权重。
                features[active_count] = value; // 原地压缩非零值；写入位置不超过当前读取位置。
                active_count += 1u;
            }
        }
        for (var i = 0u; i < active_count; i += 1u) {
            score += weights[1u + active_indices[i]] * features[i]; // 按原特征下标读取线性项权重。
        }
        for (var a = 0u; a < active_count; a += 1u) {
            let i = active_indices[a];
            let row = 45u + i * (89u - i) / 2u; // 跳过常数和 44 个线性项，再定位二次项上三角矩阵第 i 行。
            for (var b = a; b < active_count; b += 1u) {
                let j = active_indices[b];
                let term = features[a] * features[b];
                score += weights[row + j - i] * term; // 行内 j-i 偏移对应特征 i 与 j 的乘积系数。
            }
        }
    } else { // 稠密路径保持原索引顺序，跳过零项乘法但不跳过权重位置。
    var weight_index = 1u;
    for (var i = 0u; i < 44u; i += 1u) {
        if (features[i] != 0.0) {
            score += weights[weight_index] * features[i];
        }
        weight_index += 1u;
    }
    for (var i = 0u; i < 44u; i += 1u) {
        if (features[i] == 0.0) {
            weight_index += 44u - i; // 第 i 行含 44-i 个二次项，整行权重仍须跳过。
            continue;
        }
        for (var j = i; j < 44u; j += 1u) {
            if (features[j] != 0.0) {
                let term = features[i] * features[j];
                score += weights[weight_index] * term;
            }
            weight_index += 1u;
        }
    }
    }

    // 10分余量做候选粗筛；阈值附近由host基于整数记录重建f64精筛。
    // 非有限结果强制记录，交由 host validation 明确报告，不静默通过/丢弃。
    let invalid_score = !finite_score(score.x) || !finite_score(score.y); // 只要任一分数非有限，就交给宿主校验。
    // Rust as u32 将负分饱和为 0：阈值为 0 时不能丢弃负分候选。
    let possible_hit = (params.xp_threshold == 0u || score.x + 10.0 >= f32(params.xp_threshold))
        && (params.xd_threshold == 0u || score.y + 10.0 >= f32(params.xd_threshold));
    if (params.capture_all == 0u && !invalid_score && !possible_hit) { return; } // 非 capture_all 且分数有限、按 10 分余量粗筛判为不可能命中时才淘汰。

    // 每 invocation 最多写一次，故 count <= params.count；无需限容竞争或回退。
    let slot = atomicAdd(&output.count, 1u); // 原子领取唯一槽位；写入顺序不等于候选顺序。
    output.records[slot].index = index; // 记录原下标，方便宿主排序和还原候选。
    output.records[slot].props = props;
    for (var i = 0u; i < 16u; i += 1u) {
        output.records[slot].skill_ids[i] = skill_ids[i];
        output.records[slot].skill_freqs[i] = skill_freqs[i];
    }
    output.records[slot].shadow_numerator = shadow_numerator;
    output.records[slot].xp = score.x;
    output.records[slot].xd = score.y;
}
