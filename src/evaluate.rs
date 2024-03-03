pub mod base;
pub mod xuping;

use crate::name::Namer;

/// 评价器
pub trait NamerEvaluater {
    const NAME: &'static str;
    const VERSION: &'static str;
    fn check(&self, name: &Namer) -> bool;
    fn evaluate(name: &Namer) -> f64;
}
