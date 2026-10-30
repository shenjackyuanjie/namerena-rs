use base64::Engine;
use clap::ValueEnum;
use std::ops::Range;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Encoding {
    Base16384,
    Base64Std,
    Base64Url,
    Number,
}
#[derive(Clone, Debug)]
pub struct Generator {
    template: String,
    encoding: Encoding,
}
impl Generator {
    pub fn new(template: String, encoding: Encoding) -> Result<Self, String> {
        if !template.contains("[]") {
            return Err("名字模板必须包含 []".to_owned());
        }
        Ok(Self { template, encoding })
    }
    /// 全部 u64 编号的 UTF-8 名字字节数上界，实际限制在 generate 中检查。
    pub fn max_name_bytes(&self) -> usize {
        let value_bytes = match self.encoding {
            Encoding::Base16384 => base16384::Base16384Utf8::encode_len(8),
            Encoding::Base64Std | Encoding::Base64Url => 12,
            Encoding::Number => 20,
        };
        let count = self.template.matches("[]").count();
        (self.template.len() - count * 2).saturating_add(count.saturating_mul(value_bytes))
    }
    pub fn generate(&self, id: u64) -> Result<String, String> {
        let bytes = id.to_be_bytes();
        let encoded = match self.encoding {
            Encoding::Base16384 => base16384::Base16384Utf8::encode(&bytes),
            Encoding::Base64Std => base64::engine::general_purpose::STANDARD.encode(bytes),
            Encoding::Base64Url => base64::engine::general_purpose::URL_SAFE.encode(bytes),
            Encoding::Number => id.to_string(),
        };
        let result = self.template.replace("[]", &encoded);
        if result.len() > 256 {
            return Err(format!("编号{id}的名字为{}个UTF-8字节，超过256", result.len()));
        }
        Ok(result)
    }
}

pub struct BatchRanges {
    cursor: u64,
    end: u64,
    batch: u64,
}
pub fn batch_ranges(start: u64, end: u64, batch_size: usize) -> Result<BatchRanges, String> {
    if start > end || batch_size == 0 {
        return Err("区间必须start<=end且batch_size>0".to_owned());
    }
    let batch = u64::try_from(batch_size).unwrap_or(u64::MAX);
    Ok(BatchRanges {
        cursor: start,
        end,
        batch,
    })
}
impl Iterator for BatchRanges {
    type Item = Range<u64>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor == self.end {
            return None;
        }
        let start = self.cursor;
        self.cursor += self.batch.min(self.end - self.cursor);
        Some(start..self.cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_big_endian_padding_and_full_u64_roundtrips() {
        let examples = [
            (0, "一一一一一㴁", "AAAAAAAAAAA=", "AAAAAAAAAAA="),
            (1, "一一一一乀㴁", "AAAAAAAAAAE=", "AAAAAAAAAAE="),
            (0x3132_3334_3536_3738, "婌焳廔萷尀㴁", "MTIzNDU2Nzg=", "MTIzNDU2Nzg="),
            (
                u64::MAX,
                "\u{8dff}\u{8dff}\u{8dff}\u{8dff}\u{8dc0}㴁",
                "//////////8=",
                "__________8=",
            ),
        ];
        for (id, b16384, standard, url) in examples {
            for (encoding, expected) in [
                (Encoding::Base16384, b16384),
                (Encoding::Base64Std, standard),
                (Encoding::Base64Url, url),
            ] {
                assert_eq!(
                    Generator::new("[]".to_owned(), encoding).unwrap().generate(id).unwrap(),
                    expected
                );
            }
        }
        for id in [
            0,
            1,
            255,
            256,
            u32::MAX as u64,
            1_u64 << 32,
            1_u64 << 63,
            u64::MAX - 1,
            u64::MAX,
        ] {
            for encoding in [Encoding::Base16384, Encoding::Base64Std, Encoding::Base64Url, Encoding::Number] {
                let generator = Generator::new("[]".to_owned(), encoding).unwrap();
                let name = generator.generate(id).unwrap();
                assert!(name.len() <= generator.max_name_bytes());
                let bytes = match encoding {
                    Encoding::Base16384 => {
                        // 独立恢复 8 字节：避免库的 UTF-8 解码器排除最高合法码点 U+8DFF 的边界缺陷。
                        let mut chars = name.chars();
                        let packed = chars.by_ref().take(5).fold(0_u128, |bits, c| {
                            let value = c as u32 - 0x4E00;
                            assert!(value <= 0x3FFF);
                            (bits << 14) | u128::from(value)
                        });
                        assert_eq!(chars.next(), Some('㴁'));
                        assert_eq!(chars.next(), None);
                        ((packed >> 6) as u64).to_be_bytes().to_vec()
                    }
                    Encoding::Base64Std => base64::engine::general_purpose::STANDARD.decode(&name).unwrap(),
                    Encoding::Base64Url => base64::engine::general_purpose::URL_SAFE.decode(&name).unwrap(),
                    Encoding::Number => {
                        assert_eq!(name.parse::<u64>().unwrap(), id);
                        continue;
                    }
                };
                assert_eq!(bytes, id.to_be_bytes());
            }
        }
        assert_eq!(
            Generator::new("[]".to_owned(), Encoding::Base64Std)
                .unwrap()
                .generate(0x0102_0304_0506_0708)
                .unwrap(),
            "AQIDBAUGBwg="
        );
    }

    #[test]
    fn all_placeholders_and_utf8_byte_limit_boundaries() {
        for (encoding, width) in [
            (Encoding::Base16384, 18),
            (Encoding::Base64Std, 12),
            (Encoding::Base64Url, 12),
            (Encoding::Number, 20),
        ] {
            let value = Generator::new("[]".to_owned(), encoding).unwrap().generate(1).unwrap();
            let generator = Generator::new("中文🙂[][]/[]/[ ]/\r\n\0".to_owned(), encoding).unwrap();
            assert_eq!(
                generator.generate(1).unwrap(),
                format!("中文🙂{value}{value}/{value}/[ ]/\r\n\0")
            );
            let exact = Generator::new(format!("{}[]", "a".repeat(256 - width)), encoding).unwrap();
            assert_eq!(exact.max_name_bytes(), 256);
            assert_eq!(exact.generate(u64::MAX).unwrap().len(), 256);
            assert!(Generator::new(format!("{}[]", "a".repeat(257 - width)), encoding)
                .unwrap()
                .generate(u64::MAX)
                .is_err());
        }
        let generator = Generator::new(format!("{}[]", "中".repeat(85)), Encoding::Number).unwrap();
        assert_eq!(generator.generate(0).unwrap().len(), 256);
        assert!(generator.generate(10).is_err());
        let generator = Generator::new("[]".repeat(256), Encoding::Number).unwrap();
        assert_eq!(generator.generate(0).unwrap().len(), 256);
        assert!(generator.generate(10).is_err());
        assert!(Generator::new(String::new(), Encoding::Number).is_err());
    }

    #[test]
    fn exhaustive_ranges_and_u64_max_clamping() {
        for count in [0, 1, 31, 32, 33, 255, 256, 257, 513] {
            for start in [0, 7, 257, u64::MAX - count] {
                let end = start + count;
                for batch in [1, 2, 31, 32, 33, 255, 256, 257, 513, usize::MAX] {
                    let ranges: Vec<_> = batch_ranges(start, end, batch).unwrap().collect();
                    let mut cursor = start;
                    for range in &ranges {
                        assert_eq!(range.start, cursor);
                        assert!(range.start < range.end && range.end <= end);
                        assert!(range.end - range.start <= u64::try_from(batch).unwrap_or(u64::MAX));
                        cursor = range.end;
                    }
                    assert_eq!(cursor, end);
                    assert_eq!(
                        ranges.into_iter().flatten().collect::<Vec<_>>(),
                        (start..end).collect::<Vec<_>>()
                    );
                }
            }
        }
        let mut ranges = batch_ranges(0, u64::MAX, usize::MAX).unwrap();
        assert_eq!(ranges.next(), Some(0..u64::try_from(usize::MAX).unwrap_or(u64::MAX)));
        let mut empty = batch_ranges(u64::MAX, u64::MAX, 1).unwrap();
        assert_eq!(empty.next(), None);
        assert_eq!(empty.next(), None);
        assert!(batch_ranges(0, 0, 0).is_err());
    }
    #[test]
    fn exact_encoding_and_utf8_templates() {
        assert_eq!(
            Generator::new("[]".to_owned(), Encoding::Base64Std).unwrap().generate(0).unwrap(),
            "AAAAAAAAAAA="
        );
        assert_eq!(
            Generator::new("[]".to_owned(), Encoding::Base64Std).unwrap().generate(u64::MAX).unwrap(),
            "//////////8="
        );
        assert_eq!(
            Generator::new("[]".to_owned(), Encoding::Base64Url).unwrap().generate(u64::MAX).unwrap(),
            "__________8="
        );
        assert_eq!(
            Generator::new("名字[]-[]".to_owned(), Encoding::Number).unwrap().generate(42).unwrap(),
            "名字42-42"
        );
        assert!(Generator::new("没有占位符".to_owned(), Encoding::Number).is_err());
        assert!(Generator::new(format!("{}[]", "中".repeat(86)), Encoding::Number)
            .unwrap()
            .generate(1)
            .is_err());
    }
    #[test]
    fn clipped_ranges_cover_every_id_once() {
        for count in [0, 1, 31, 32, 33, 255, 256, 257, 513] {
            for batch in [1, 32, 257, 1024] {
                let actual: Vec<_> = batch_ranges(7, 7 + count, batch).unwrap().flatten().collect();
                assert_eq!(actual, (7..7 + count).collect::<Vec<_>>());
            }
        }
        assert_eq!(
            batch_ranges(u64::MAX - 5, u64::MAX, 3).unwrap().collect::<Vec<_>>(),
            vec![u64::MAX - 5..u64::MAX - 2, u64::MAX - 2..u64::MAX]
        );
        assert!(batch_ranges(1, 0, 1).is_err());
        assert!(batch_ranges(0, 1, 0).is_err());
    }
}
