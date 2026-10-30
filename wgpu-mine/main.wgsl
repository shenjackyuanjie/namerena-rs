// 每个候选在独立 invocation 内执行两轮名字 KSA；队伍 KSA 在 CPU 预计算一次。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>;
struct NameInfo { offset: u32, length: u32, }
@group(0) @binding(1) var<storage, read> names: array<NameInfo>;
// UTF-8 字节连续打包，每个 u32 的低字节在前，避免将每个输入字节扩张成 u32。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>;
struct Params { count: u32, padding0: u32, padding1: u32, padding2: u32, }
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn rc4_name(@builtin(global_invocation_id) position: vec3<u32>) {
    let index = position.x;
    if (index >= params.count) { return; }
    let info = names[index];
    var values: array<u32, 256>;
    for (var i = 0u; i < 256u; i += 1u) {
        values[i] = team_state[i / 4u][i % 4u];
    }
    // 密钥为 [0, UTF-8(name)...]；每一轮都重置局部 j 和密钥游标。
    for (var round = 0u; round < 2u; round += 1u) {
        var j = 0u;
        var k = 0u;
        for (var i = 0u; i < 256u; i += 1u) {
            if (k != 0u) {
                let address = info.offset + k - 1u;
                j += (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u;
            }
            j = (j + values[i]) & 255u;
            let temporary = values[i];
            values[i] = values[j];
            values[j] = temporary;
            if (k == info.length) { k = 0u; } else { k += 1u; }
        }
    }
    // 保留完整置换但只回读 256 字节/名字，而不是原型的 1024 字节。
    for (var i = 0u; i < 64u; i += 1u) {
        let base = i * 4u;
        output_states[index * 64u + i] = values[base] | (values[base + 1u] << 8u)
            | (values[base + 2u] << 16u) | (values[base + 3u] << 24u);
    }
}
