// 32 个候选共享一块 32KiB workgroup 状态，每列属于一个 invocation。
// 转置布局让相同槽的 lane 相邻，意在减少共享内存 bank 冲突；实际效果取决于硬件。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>; // 绑定 0：宿主预计算的队伍状态，64 个四分量共 256 项。
struct NameInfo { offset: u32, length: u32, } // offset 和 length 是名字在连续 UTF-8 字节流中的偏移与长度。
@group(0) @binding(1) var<storage, read> names: array<NameInfo>; // 绑定 1：只读候选元数据。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>; // 绑定 2：每个 u32 低字节在前的名字数据。
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>; // 绑定 3：可写状态结果，每个候选 64 个 u32。
struct Params { count: u32, padding0: u32, padding1: u32, padding2: u32, } // count 是真实候选数，三个字段填充参数块到 16 字节。
@group(0) @binding(4) var<uniform> params: Params; // 绑定 4：宿主上传的只读批次参数。
var<workgroup> values: array<u32, 8192>; // 256 个槽乘 32 列，每列独占一个候选，共 32 KiB。

@compute @workgroup_size(32) // 每组 32 个 invocation；这里只回传两轮 KSA 状态。
fn rc4_name(@builtin(global_invocation_id) position: vec3<u32>, @builtin(local_invocation_index) lane: u32) { // position.x 是批内全局候选编号，lane 是组内编号。
    let index = position.x;
    if (index >= params.count) { return; } // 尾组补齐的 invocation 不处理候选，也不访问数组。
    let info = names[index];
    for (var i = 0u; i < 256u; i += 1u) {
        values[i * 32u + lane] = team_state[i / 4u][i % 4u]; // 将队伍状态复制到本 lane 的转置列：槽 i 的地址为 i*32+lane。
    }
    // 每列由同一个 invocation 初始化并读写，不跨列交换，无须轮内 barrier。
    for (var round = 0u; round < 2u; round += 1u) { // 两轮连续更新状态；每轮重新从零设置 j 和密钥游标 k。
        var j = 0u;
        var k = 0u;
        for (var i = 0u; i < 256u; i += 1u) {
            if (k != 0u) { // k=0 对应隐式前导零，其余位置读取名字字节。
                let address = info.offset + k - 1u;
                j += (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u; // 除以 4 选输入字，余数选字节，右移并掩码提取。
            }
            let address_i = i * 32u + lane;
            let current = values[address_i];
            j = (j + current) & 255u; // 加当前槽值后对 256 取余，得到交换目标 j。
            let address_j = j * 32u + lane; // 目标地址仍在同一 lane 的列中。
            values[address_i] = values[address_j]; // 交换两个槽；即使 i 与 j 相同也安全。
            values[address_j] = current;
            if (k == info.length) { k = 0u; } else { k += 1u; } // 密钥按前导零和全部名字字节循环。
        }
    }
    for (var i = 0u; i < 64u; i += 1u) { // 每四个状态槽打包成一个 u32，每候选输出 64 个字。
        let base = i * 4u;
        output_states[index * 64u + i] = values[base * 32u + lane] // 依次将四个槽放到结果字的低至高字节，按候选连续排列。
            | (values[(base + 1u) * 32u + lane] << 8u)
            | (values[(base + 2u) * 32u + lane] << 16u)
            | (values[(base + 3u) * 32u + lane] << 24u);
    }
}
