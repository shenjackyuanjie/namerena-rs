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
    pub fn gen_name(&self, template: &str, id: u64) -> String {
        template.replace("{}",
        &match self {
            GenMethods::Base16384 => {
                Base16384Utf8::encode(&id.to_be_bytes())
            }
            GenMethods::Base64Std => {
                base64::engine::general_purpose::STANDARD.encode(&id.to_be_bytes())
            }
            GenMethods::Base64Url => {
                base64::engine::general_purpose::URL_SAFE.encode(&id.to_be_bytes())
            }
            GenMethods::Number => {
                id.to_string()
            }
        })
    }
}
