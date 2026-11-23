// 每个 invocation 独占缓存：Base16384为6个三字节UTF-8字，Base64为3个四字节字。
// 此片段替代原key_byte，不新增绑定；names[1].length存编码编号，params.pad0/1存起始ID。宿主会整体替换 KEY_BEGIN 与 KEY_END 间的函数定义；它不是独立计算入口。
var<private> generated_chunks: array<u32, 6>; // private 表示每个 invocation 拥有自己的缓存。

fn prepare_generated(index: u32) { // 根据批次起始 ID 和当前候选编号生成固定宽度编码。
    let low = params.pad0 + index;
    let high = params.pad1 + select(0u, 1u, low < params.pad0); // 低 32 位回绕时向高 32 位进一，拼成 64 位 ID。
    let encoding = names[1].length; // 0 为 Base16384，1 为标准 Base64，2 为 URL 安全 Base64。
    if (encoding == 0u) { // Base16384 把高位在前的 ID 位段映射为六个 Unicode 码点。
        // 64位大端输入：前56bit拆四组14bit，最后8bit左移6，末尾是固定㴁。
        for (var group = 0u; group < 6u; group += 1u) {
            var point = 0u;
            switch group {
                case 0u: { point = 0x4e00u + ((high >> 18u) & 0x3fffu); }
                case 1u: { point = 0x4e00u + ((high >> 4u) & 0x3fffu); }
                case 2u: { point = 0x4e00u + (((high << 10u) | (low >> 22u)) & 0x3fffu); } // 跨越高低字边界：高字低 4 位与低字高 10 位组成一组。
                case 3u: { point = 0x4e00u + ((low >> 8u) & 0x3fffu); }
                case 4u: { point = 0x4e00u + ((low & 255u) << 6u); } // 最后 8 位左移 6 位补零，形成第 5 组。
                default: { point = 0x3d01u; } // 第 6 个码点是固定结尾。
            }
            // U+4E00..U+8DFF和U+3D01均是三个字节的合法UTF-8，最高U+8DFF不排除。
            generated_chunks[group] = (0xe0u | (point >> 12u)) // 按 UTF-8 三字节格式编码，每字节依次放入 u32 的低到高位。
                | ((0x80u | ((point >> 6u) & 63u)) << 8u)
                | ((0x80u | (point & 63u)) << 16u);
        }
        return;
    }
    // Base64的前十字符消费60bit，第11字符保存剩余4bit<<2，第12字符固定'='。
    for (var group = 0u; group < 3u; group += 1u) { // Base64 共 12 个字符，每个 u32 打包四个 ASCII 字节。
        var packed = 0u;
        for (var offset = 0u; offset < 4u; offset += 1u) {
            let character = group * 4u + offset;
            var ascii = 61u; // 默认填充字符 =；只有前 11 个字符需要编码。
            if (character < 11u) {
                var value = 0u;
                if (character < 5u) {
                    value = (high >> (26u - character * 6u)) & 63u;
                } else if (character == 5u) {
                    value = ((high << 4u) | (low >> 28u)) & 63u; // 第 6 个字符跨越高低字边界，取高字末 2 位和低字前 4 位。
                } else if (character < 10u) {
                    value = (low >> (58u - character * 6u)) & 63u;
                } else {
                    value = (low << 2u) & 63u; // 最后剩余 4 位左移 2 位补零。
                }
                if (value < 26u) { // 六位值映射到 Base64 字母表：大写、小写、数字及两个符号。
                    ascii = 65u + value;
                } else if (value < 52u) {
                    ascii = 97u + value - 26u;
                } else if (value < 62u) {
                    ascii = 48u + value - 52u;
                } else if (value == 62u) {
                    ascii = select(43u, 45u, encoding == 2u); // 标准表用 +，URL 安全表用 -。
                } else {
                    ascii = select(47u, 95u, encoding == 2u); // 标准表用 /，URL 安全表用 _。
                }
            }
            packed |= ascii << (offset * 8u); // 按字符在组内的位置放入对应字节，低字节先输出。
        }
        generated_chunks[group] = packed;
    }
}

// info.offset是prefix_steps而非字节地址；template_map逐u32上传，不再是packed原名字。names[0].length 是名字字节长度；names[1] 保存前缀游标和编码编号。
fn key_byte(info: NameInfo, k: u32) -> u32 {
    if (k == 0u) { return 0u; } // KSA 的密钥有一个隐式前导零。
    let byte_index = k - 1u;
    if (byte_index < info.length) {
        let token = name_bytes[byte_index]; // 每个 token 占一个 u32：常量字节或生成字节的引用。
        if (token < 256u) { return token; } // 小于 256 的 token 直接是固定字节。
        let offset = token & 255u; // 其他 token 的低八位是编码结果中的字节偏移。
        if (names[1].length == 0u) {
            return (generated_chunks[offset / 3u] >> ((offset % 3u) * 8u)) & 255u; // Base16384 每槽三字节；除法选槽，余数选槽内字节。
        }
        return (generated_chunks[offset / 4u] >> ((offset % 4u) * 8u)) & 255u; // Base64 每槽四字节，同样右移并掩码提取。
    }
    return shadow_byte(byte_index - info.length); // 只有影子密钥越过名字长度后才读取 ?shadow。
}
