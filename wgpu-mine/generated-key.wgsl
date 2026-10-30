// 每个 invocation 独占缓存：Base16384为6个三字节UTF-8字，Base64为3个四字节字。
// 此片段替代原key_byte，不新增绑定；names[1].length存编码编号，params.pad0/1存起始ID。
var<private> generated_chunks: array<u32, 6>;

fn prepare_generated(index: u32) {
    let low = params.pad0 + index;
    let high = params.pad1 + select(0u, 1u, low < params.pad0);
    let encoding = names[1].length;
    if (encoding == 0u) {
        // 64位大端输入：前56bit拆四组14bit，最后8bit左移6，末尾是固定㴁。
        for (var group = 0u; group < 6u; group += 1u) {
            var point = 0u;
            switch group {
                case 0u: { point = 0x4e00u + ((high >> 18u) & 0x3fffu); }
                case 1u: { point = 0x4e00u + ((high >> 4u) & 0x3fffu); }
                case 2u: { point = 0x4e00u + (((high << 10u) | (low >> 22u)) & 0x3fffu); }
                case 3u: { point = 0x4e00u + ((low >> 8u) & 0x3fffu); }
                case 4u: { point = 0x4e00u + ((low & 255u) << 6u); }
                default: { point = 0x3d01u; }
            }
            // U+4E00..U+8DFF和U+3D01均是三个字节的合法UTF-8，最高U+8DFF不排除。
            generated_chunks[group] = (0xe0u | (point >> 12u))
                | ((0x80u | ((point >> 6u) & 63u)) << 8u)
                | ((0x80u | (point & 63u)) << 16u);
        }
        return;
    }
    // Base64的前十字符消费60bit，第11字符保存剩余4bit<<2，第12字符固定'='。
    for (var group = 0u; group < 3u; group += 1u) {
        var packed = 0u;
        for (var offset = 0u; offset < 4u; offset += 1u) {
            let character = group * 4u + offset;
            var ascii = 61u;
            if (character < 11u) {
                var value = 0u;
                if (character < 5u) {
                    value = (high >> (26u - character * 6u)) & 63u;
                } else if (character == 5u) {
                    value = ((high << 4u) | (low >> 28u)) & 63u;
                } else if (character < 10u) {
                    value = (low >> (58u - character * 6u)) & 63u;
                } else {
                    value = (low << 2u) & 63u;
                }
                if (value < 26u) {
                    ascii = 65u + value;
                } else if (value < 52u) {
                    ascii = 97u + value - 26u;
                } else if (value < 62u) {
                    ascii = 48u + value - 52u;
                } else if (value == 62u) {
                    ascii = select(43u, 45u, encoding == 2u);
                } else {
                    ascii = select(47u, 95u, encoding == 2u);
                }
            }
            packed |= ascii << (offset * 8u);
        }
        generated_chunks[group] = packed;
    }
}

// info.offset是prefix_steps而非字节地址；template_map逐u32上传，不再是packed原名字。
fn key_byte(info: NameInfo, k: u32) -> u32 {
    if (k == 0u) { return 0u; }
    let byte_index = k - 1u;
    if (byte_index < info.length) {
        let token = name_bytes[byte_index];
        if (token < 256u) { return token; }
        let offset = token & 255u;
        if (names[1].length == 0u) {
            return (generated_chunks[offset / 3u] >> ((offset % 3u) * 8u)) & 255u;
        }
        return (generated_chunks[offset / 4u] >> ((offset % 4u) * 8u)) & 255u;
    }
    return shadow_byte(byte_index - info.length);
}
