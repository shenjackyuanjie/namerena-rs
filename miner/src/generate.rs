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
