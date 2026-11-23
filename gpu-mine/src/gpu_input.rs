//! 固定宽度编码的 GPU 虚拟名字：只上传模板映射与第一轮 KSA 公共前缀。
use crate::input::Encoding;

#[derive(Clone, Debug)]
pub struct GpuInput {
    pub name_len: u32,
    pub prefix_steps: u32,
    pub prefix_j: u32,
    pub prefix_state: [u32; 256],
    pub template_map: Vec<u32>,
    pub encoding: u32,
}

impl GpuInput {
    /// 常量字节为0..255，生成编码的第n字节为0x100|n，替换全部[]。
    /// Number 长度可变，返回 None 交由原上传路径处理。
    pub fn new(team: &str, template: &str, encoding: Encoding) -> Result<Option<Self>, String> {
        let Some(first_placeholder) = template.find("[]") else {
            return Err("名字模板必须包含 []".to_owned());
        };
        let (encoding, width) = match encoding {
            Encoding::Base16384 => (0, 18_usize),
            Encoding::Base64Std => (1, 12),
            Encoding::Base64Url => (2, 12),
            Encoding::Number => return Ok(None),
        };
        let count = template.matches("[]").count();
        let name_len = (template.len() - count * 2).saturating_add(count.saturating_mul(width));
        if name_len > 256 {
            return Err(format!("固定编码名字为{name_len}个UTF-8字节，超过256"));
        }
        let mut template_map = Vec::with_capacity(name_len);
        for (part_index, part) in template.split("[]").enumerate() {
            if part_index != 0 {
                template_map.extend((0..width).map(|offset| 0x100 | offset as u32));
            }
            template_map.extend(part.bytes().map(u32::from));
        }
        let prefix = &template.as_bytes()[..first_placeholder];
        let prefix_steps = prefix.len() + 1;
        // 固定生成部分至少12B，所以合法名字的前缀最多244B，含前导零不超过245步。
        let mut state = crate::cpu::team_state(team);
        let mut j = 0_u8;
        for i in 0..prefix_steps {
            let byte = if i == 0 { 0 } else { prefix[i - 1] };
            j = j.wrapping_add(state[i]).wrapping_add(byte);
            state.swap(i, usize::from(j));
        }
        Ok(Some(Self {
            name_len: name_len as u32,
            prefix_steps: prefix_steps as u32,
            prefix_j: u32::from(j),
            prefix_state: state.map(u32::from),
            template_map,
            encoding,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Generator;

    // 对照着色器双u32切片，仅在测试里使用；金标准是原固定编码encode，不用decode。
    fn shader_encoding(encoding: u32, start: u64, index: u32) -> Vec<u8> {
        let (low, carry) = (start as u32).overflowing_add(index);
        let high = ((start >> 32) as u32).wrapping_add(u32::from(carry));
        if encoding == 0 {
            let groups = [
                (high >> 18) & 0x3fff,
                (high >> 4) & 0x3fff,
                ((high << 10) | (low >> 22)) & 0x3fff,
                (low >> 8) & 0x3fff,
                (low & 255) << 6,
            ];
            return groups
                .into_iter()
                .map(|value| value + 0x4e00)
                .chain(std::iter::once(0x3d01))
                .flat_map(|point| {
                    [
                        ((0xe0 | (point >> 12)) & 255) as u8,
                        (0x80 | ((point >> 6) & 63)) as u8,
                        (0x80 | (point & 63)) as u8,
                    ]
                })
                .collect();
        }
        let alphabet = if encoding == 1 {
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        } else {
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        };
        let mut result = Vec::with_capacity(12);
        for character in 0..11 {
            let value = if character < 5 {
                (high >> (26 - character * 6)) & 63
            } else if character == 5 {
                ((high << 4) | (low >> 28)) & 63
            } else if character < 10 {
                (low >> (58 - character * 6)) & 63
            } else {
                (low << 2) & 63
            };
            result.push(alphabet[value as usize]);
        }
        result.push(b'=');
        result
    }

    fn materialize(input: &GpuInput, encoded: &[u8]) -> String {
        let bytes = input
            .template_map
            .iter()
            .map(|&token| {
                if token < 256 {
                    token as u8
                } else {
                    encoded[(token & 255) as usize]
                }
            })
            .collect();
        String::from_utf8(bytes).unwrap()
    }

    fn reference_round(state: &mut [u8; 256], text: &str) {
        let key: Vec<u8> = std::iter::once(0).chain(text.bytes()).collect();
        let mut j = 0_usize;
        for i in 0..256 {
            j = (j + usize::from(state[i]) + usize::from(key[i % key.len()])) % 256;
            state.swap(i, j);
        }
    }

    // 从上传快照接完第一轮，再独立重置第二轮；同时检验第一轮中间结果。
    fn resume_and_check(input: &GpuInput, team: &str, name: &str) {
        let mut state = input.prefix_state.map(|value| u8::try_from(value).unwrap());
        let key: Vec<u8> = std::iter::once(0).chain(name.bytes()).collect();
        let mut j = input.prefix_j as usize;
        for i in input.prefix_steps as usize..256 {
            j = (j + usize::from(state[i]) + usize::from(key[i % key.len()])) % 256;
            state.swap(i, j);
        }
        let mut first = std::array::from_fn(|i| i as u8);
        reference_round(&mut first, team);
        reference_round(&mut first, name);
        assert_eq!(state, first, "接续第一轮：{name:?}@{team:?}");
        reference_round(&mut state, name);
        assert_eq!(state, crate::core::rc4_name(&crate::cpu::team_state(team), name));
        assert_eq!(state, crate::cpu::reference(team, name));
    }

    #[test]
    fn mapping_keeps_fixed_encoding_utf8_and_all_placeholders() {
        for (encoding, code, width) in [
            (Encoding::Base16384, 0, 18),
            (Encoding::Base64Std, 1, 12),
            (Encoding::Base64Url, 2, 12),
        ] {
            for template in ["[]", "前@缀\0[]中[]末", "[][]", "[[]]/[]/[]"] {
                let input = GpuInput::new("队@伍\0", template, encoding).unwrap().unwrap();
                let generator = Generator::new(template.to_owned(), encoding).unwrap();
                assert_eq!(input.encoding, code);
                assert_eq!(input.name_len as usize, input.template_map.len());
                assert_eq!(input.prefix_steps as usize, template.find("[]").unwrap() + 1);
                assert!(input
                    .template_map
                    .iter()
                    .all(|&token| token < 256 || (0x100..0x100 + width).contains(&token)));
                for id in [0, 1, u32::MAX as u64, 1_u64 << 32, 0x3132_3334_3536_3738, u64::MAX] {
                    let name = materialize(&input, &shader_encoding(code, id, 0));
                    assert_eq!(name, generator.generate(id).unwrap());
                    assert_eq!(name.len(), input.name_len as usize);
                    resume_and_check(&input, "队@伍\0", &name);
                    resume_and_check(&input, "队@伍\0", &format!("{name}?shadow"));
                }
            }
        }
    }

    #[test]
    fn double_u32_nonce_carry_matches_original_encoding_at_all_boundaries() {
        for (encoding, code) in [(Encoding::Base16384, 0), (Encoding::Base64Std, 1), (Encoding::Base64Url, 2)] {
            let generator = Generator::new("[]".to_owned(), encoding).unwrap();
            for start in [0, 1, 0xffff_fffc, 0xffff_ffff, 1_u64 << 32, 1_u64 << 63, u64::MAX - 9, u64::MAX] {
                for index in [0, 1, 2, 9, 31, u32::MAX] {
                    let expected = generator.generate(start.wrapping_add(u64::from(index))).unwrap();
                    assert_eq!(shader_encoding(code, start, index), expected.as_bytes());
                }
            }
            let mut random = 0x1234_5678_9abc_def0_u64;
            for _ in 0..1024 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let index = (random >> 32) as u32;
                assert_eq!(
                    shader_encoding(code, random, index),
                    generator.generate(random.wrapping_add(u64::from(index))).unwrap().as_bytes()
                );
            }
        }
        let maximum = shader_encoding(0, u64::MAX, 0);
        assert_eq!(
            String::from_utf8(maximum).unwrap(),
            "\u{8dff}\u{8dff}\u{8dff}\u{8dff}\u{8dc0}㴁"
        );
        assert_eq!(shader_encoding(1, u64::MAX, 0), b"//////////8=".to_vec());
        assert_eq!(shader_encoding(2, u64::MAX, 0), b"__________8=".to_vec());
    }

    #[test]
    fn length_boundary_and_number_fallback_are_explicit() {
        for (encoding, width) in [(Encoding::Base16384, 18), (Encoding::Base64Std, 12), (Encoding::Base64Url, 12)] {
            let template = format!("{}[]", "a".repeat(256 - width));
            let input = GpuInput::new("", &template, encoding).unwrap().unwrap();
            assert_eq!(input.name_len, 256);
            let name = Generator::new(template.clone(), encoding).unwrap().generate(u64::MAX).unwrap();
            resume_and_check(&input, "", &name);
            resume_and_check(&input, "", &format!("{name}?shadow"));
            assert!(GpuInput::new("", &(template + "a"), encoding).is_err());
            assert!(GpuInput::new("", "没有占位符", encoding).is_err());
        }
        assert!(GpuInput::new("", "[]", Encoding::Number).unwrap().is_none());
        assert!(GpuInput::new("", "没有占位符", Encoding::Number).is_err());
        assert!(GpuInput::new("", &"[]".repeat(15), Encoding::Base16384).is_err());
    }
}
