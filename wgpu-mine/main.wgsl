// 每个候选在独立 invocation 内执行两轮名字 KSA；队伍 KSA 在 CPU 预计算一次。这是 RC4 对照实验，只回传状态，不计算完整属性、技能或模型。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>; // 绑定 0：宿主预计算的 256 个队伍状态值，四个一组放入 uniform。
struct NameInfo { offset: u32, length: u32, } // offset 和 length 均以 UTF-8 字节为单位。
@group(0) @binding(1) var<storage, read> names: array<NameInfo>; // 绑定 1：只读候选元数据数组。
// UTF-8 字节连续打包，每个 u32 的低字节在前，避免将每个输入字节扩张成 u32。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>; // 绑定 2：按低字节在前打包的只读名字字节。
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>; // 绑定 3：可写结果，每个候选 64 个 u32。
struct Params { count: u32, padding0: u32, padding1: u32, padding2: u32, } // count 是真实候选数，其他字段用于填充到 16 字节。
@group(0) @binding(4) var<uniform> params: Params; // 绑定 4：宿主上传的只读批次参数。

@compute @workgroup_size(64) // 每组 64 个 invocation；一个 invocation 处理一个候选。
fn rc4_name(@builtin(global_invocation_id) position: vec3<u32>) {
    let index = position.x; // 只使用三维全局编号的 x 分量作为候选下标。
    if (index >= params.count) { return; } // 派发按组数向上取整，超出真实数量的线程退出。
    let info = names[index];
    var values: array<u32, 256>; // 函数内局部数组属于当前 invocation，其他候选不能访问。
    for (var i = 0u; i < 256u; i += 1u) {
        values[i] = team_state[i / 4u][i % 4u]; // 从 64 个四分量向量中取出 256 项队伍状态。
    }
    // 密钥为 [0, UTF-8(name)...]；每一轮都重置局部 j 和密钥游标。
    for (var round = 0u; round < 2u; round += 1u) { // 两轮保留 S-box 状态，但每轮重置 j 和密钥游标。
        var j = 0u;
        var k = 0u;
        for (var i = 0u; i < 256u; i += 1u) {
            if (k != 0u) { // k 为零表示隐式前导零，否则读取名字的第 k-1 个字节。
                let address = info.offset + k - 1u;
                j += (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u; // 除以 4 选打包字，余数乘 8 选字节，& 255 截取低八位。
            }
            j = (j + values[i]) & 255u; // 累加当前状态并对 256 取余，得到交换目标 j。
            let temporary = values[i]; // 保存旧值，交换当前槽和目标槽；相同槽也安全。
            values[i] = values[j];
            values[j] = temporary;
            if (k == info.length) { k = 0u; } else { k += 1u; } // 循环使用前导零和全部名字字节，空名字也能循环。
        }
    }
    // 保留完整置换但只回读 256 字节/名字，而不是原型的 1024 字节。
    for (var i = 0u; i < 64u; i += 1u) {
        let base = i * 4u;
        output_states[index * 64u + i] = values[base] | (values[base + 1u] << 8u) // 四个状态字节依次放入结果字的低至高位，结果按候选连续排列。
            | (values[base + 2u] << 16u) | (values[base + 3u] << 24u);
    }
}
