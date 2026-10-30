use crate::GenMethods;

use base64::Engine;
use base16384::Base16384Utf8;

impl GenMethods {
    /// 根据 u64 生成对应的 name
    /// 转换成 base 16384
    /// 禁用:
    /// U00 ~ U1F ，换行，制表符 等
    /// ? , 问号
    /// U2000 - U202F , unicode特殊空格 等
    /// 不可以空格开头
    #[inline(always)]
    pub fn gen_name(&self, template: &str, id: u64) -> String {
        template.replace(
            "[]",
            &match self {
                GenMethods::Base16384 => Base16384Utf8::encode(&id.to_be_bytes()),
                GenMethods::Base64Std => base64::engine::general_purpose::STANDARD.encode(id.to_be_bytes()),
                GenMethods::Base64Url => base64::engine::general_purpose::URL_SAFE.encode(id.to_be_bytes()),
                GenMethods::Number => id.to_string(),
            },
        )
    }
}

/// 每个 worker 编译一次模板，复用调用方的名字缓冲；每个候选不分配临时字符串。
pub struct NameGenerator<'a> {
    algorithm: &'a GenMethods,
    parts: Vec<&'a str>,
}

impl<'a> NameGenerator<'a> {
    pub fn new(algorithm: &'a GenMethods, template: &'a str) -> Self {
        Self {
            algorithm,
            parts: template.split("[]").collect(),
        }
    }

    pub fn prefix(&self) -> &str { self.parts[0] }

    pub fn render(&self, id: u64, output: &mut String) {
        use std::fmt::Write;
        let mut buffer = [b' '; 20];
        let encoded = match self.algorithm {
            GenMethods::Base16384 => {
                Base16384Utf8::encode_to_slice(&id.to_be_bytes(), std::str::from_utf8_mut(&mut buffer[..18]).unwrap())
            }
            GenMethods::Base64Std | GenMethods::Base64Url => {
                let engine = if matches!(self.algorithm, GenMethods::Base64Std) {
                    &base64::engine::general_purpose::STANDARD
                } else {
                    &base64::engine::general_purpose::URL_SAFE
                };
                let len = engine.encode_slice(id.to_be_bytes(), &mut buffer).unwrap();
                std::str::from_utf8(&buffer[..len]).unwrap()
            }
            GenMethods::Number => {
                // 最多 20 位，反向进位生成；保持十进制无前导零的原语义。
                let mut remaining = id;
                let mut cursor = buffer.len();
                loop {
                    cursor -= 1;
                    buffer[cursor] = b'0' + (remaining % 10) as u8;
                    remaining /= 10;
                    if remaining == 0 {
                        break;
                    }
                }
                std::str::from_utf8(&buffer[cursor..]).unwrap()
            }
        };
        output.clear();
        output.push_str(self.parts[0]);
        for part in &self.parts[1..] {
            // push_str 不重新解析模板，支持多个占位符和 UTF-8。
            write!(output, "{encoded}{part}").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_keeps_big_endian_and_padding() {
        assert_eq!(GenMethods::Base64Std.gen_name("[]", 0), "AAAAAAAAAAA=");
        assert_eq!(GenMethods::Base64Std.gen_name("[]", u64::MAX), "//////////8=");
        assert_eq!(GenMethods::Base64Url.gen_name("[]", u64::MAX), "__________8=");
        assert_eq!(GenMethods::Base64Std.gen_name("名字 []", 1), "名字 AAAAAAAAAAE=");
    }

    #[test]
    fn templates_replace_every_placeholder() {
        assert_eq!(
            GenMethods::Number.gen_name("我也不知道我要测多少 []-[]", 42),
            "我也不知道我要测多少 42-42"
        );
    }
}
#[cfg(test)]
mod buffer_tests {
    use super::*;
    #[test]
    fn reusable_generator_matches_original_encoder() {
        let mut output = String::with_capacity(256);
        let mut state = 0x1234_5678_9abc_def0_u64;
        for algorithm in [
            GenMethods::Base16384,
            GenMethods::Base64Std,
            GenMethods::Base64Url,
            GenMethods::Number,
        ] {
            for template in ["[]", "中文 [] 后缀 []", "a,@\"\n[]", "[]?shadow", "没有占位符"] {
                let generator = NameGenerator::new(&algorithm, template);
                for id in [0, 1, 9, 10, 99, 100, u64::MAX, 0xffff_ffff_ffff_ff00] {
                    generator.render(id, &mut output);
                    assert_eq!(output, algorithm.gen_name(template, id));
                }
                for _ in 0..1000 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    generator.render(state, &mut output);
                    assert_eq!(output, algorithm.gen_name(template, state));
                }
            }
        }
        // U+8DFF 是合法最高码点；仅依赖原 encode 对拍，不使用有已知边界缺陷的 decode。
        NameGenerator::new(&GenMethods::Base16384, "[]").render(u64::MAX, &mut output);
        assert!(output.contains('\u{8dff}'));
    }
}
