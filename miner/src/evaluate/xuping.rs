mod eval;
mod model13;
#[allow(clippy::excessive_precision)]
mod model20;

use crate::evaluate::NamerEvaluater;
use crate::name::Namer;

pub struct XuPing1_3_1;

impl NamerEvaluater for XuPing1_3_1 {
    const NAME: &'static str = "虚评";
    const VERSION: &'static str = "1.3.1";
    fn evaluate(name: &Namer) -> f64 { eval::predict_13(name) }
    fn check(&self, name: &Namer) -> bool { eval::predict_13(name) > 0.0 }
}

pub struct XuPing2_0_1015;

impl NamerEvaluater for XuPing2_0_1015 {
    const NAME: &'static str = "虚评_强评";
    const VERSION: &'static str = "2.0-10.15";
    fn evaluate(name: &Namer) -> f64 { eval::predict_20(name) }
    fn check(&self, name: &Namer) -> bool { eval::predict_20(name) > 0.0 }
}

#[allow(non_camel_case_types)]
pub struct XuPing2_0_1015_QD;

impl NamerEvaluater for XuPing2_0_1015_QD {
    const NAME: &'static str = "虚评_强单";
    const VERSION: &'static str = "2.0-10.15-QD";
    fn evaluate(name: &Namer) -> f64 { eval::predict_20_qd(name) }
    fn check(&self, name: &Namer) -> bool { eval::predict_20_qd(name) > 0.0 }
}
