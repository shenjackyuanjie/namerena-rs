// 32 个候选共享一块 32KiB workgroup 状态，每列属于一个 invocation。
// values[slot * 32 + lane] 使同一 warp 的不同候选落到不同共享内存 bank。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>;
struct NameInfo { offset: u32, length: u32, }
@group(0) @binding(1) var<storage, read> names: array<NameInfo>;
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>;
struct Params { count: u32, padding0: u32, padding1: u32, padding2: u32, }
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> values: array<u32, 8192>;

@compute @workgroup_size(32)
fn rc4_name(@builtin(global_invocation_id) position: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = position.x;
    if (index >= params.count) { return; }
    let info = names[index];
    for (var i = 0u; i < 256u; i += 1u) {
        values[i * 32u + lane] = team_state[i / 4u][i % 4u];
    }
    // 每列由同一个 invocation 初始化并读写，不跨列交换，无须轮内 barrier。
    for (var round = 0u; round < 2u; round += 1u) {
        var j = 0u;
        var k = 0u;
        for (var i = 0u; i < 256u; i += 1u) {
            if (k != 0u) {
                let address = info.offset + k - 1u;
                j += (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u;
            }
            let address_i = i * 32u + lane;
            let current = values[address_i];
            j = (j + current) & 255u;
            let address_j = j * 32u + lane;
            values[address_i] = values[address_j];
            values[address_j] = current;
            if (k == info.length) { k = 0u; } else { k += 1u; }
        }
    }
    for (var i = 0u; i < 64u; i += 1u) {
        let base = i * 4u;
        output_states[index * 64u + i] = values[base * 32u + lane]
            | (values[(base + 1u) * 32u + lane] << 8u)
            | (values[(base + 2u) * 32u + lane] << 16u)
            | (values[(base + 3u) * 32u + lane] << 24u);
    }
}
