@group(0) @binding(0) var<uniform> first_step: array<vec4<u32>, 64>;

// 处理第二步的两次 rc4 名称, 输入名字 bytes, 从第一步得到的 val
@compute
@workgroup_size(16, 16)
fn rc4_name(name_bytes: array<u32, 256>, name_len: u32) -> array<u32, 256> {
    // var val: array<u32, 256> = first_step;
    var val: array<u32, 256> = array<u32, 256>();
    // 把 first_step 的值复制到 val
    // first_step 内当成连续的内存即可
    for (var i: u32 = 0; i < 64; i = i + 1) {
        val[i * 4] = first_step[i].x;
        val[i * 4 + 1] = first_step[i].y;
        val[i * 4 + 2] = first_step[i].z;
        val[i * 4 + 3] = first_step[i].w;
    }
    // 上面的重复两次
    for (var n: u32 = 0; n < 2; n++) {    
        var s: u32 = 0u;
        var k: u32 = 0u;
        for (var i: u32 = 0; i < 256; i = i + 1) {
            if (k != 0) {
                s = s + val[k - 1];
            }
            s = s + val[i];
            s = s % 256u;
            var tmp = val[i];
            val[i] = val[s];
            val[s] = tmp;
            if (k == name_len - 1) {
                k = 0u;
            } else {
                k++;
            }
        }
    }
    return val;
}
