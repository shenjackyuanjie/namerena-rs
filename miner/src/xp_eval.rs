#![feature(portable_simd)]
#![feature(slice_swap_unchecked)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
mod evaluate;
mod name;

use std::env;

use evaluate::NamerEvaluater;

fn main() {
    // 获取 cli 参数
    let arg = env::args().nth(1).expect("need a name to evaluate");
    if arg == "-f" {
        let arg = env::args().nth(2).expect("need a file to evaluate");
        let file = std::fs::read_to_string(arg).expect("file not found");
        let names = file.lines().collect::<Vec<_>>();
        for name in names {
            let mut name = name::Namer::new(&name.to_string()).expect("your name have some thing wrong");
            name.update_skill();
            let xu = evaluate::xuping::XuPing2_0_1015::evaluate(&name);
            let xd = evaluate::xuping::XuPing2_0_1015_QD::evaluate(&name);
            println!("{xu}|{xd}")
        }
        return;
    } else if arg == "-h" {
        println!("Usage: xp_eval [name]");
        println!("       xp_eval -f [file]");
        return;
    }
    let mut name = name::Namer::new(&arg).expect("your name have some thing wrong");
    name.update_skill();

    let xu = evaluate::xuping::XuPing2_0_1015::evaluate(&name);
    let xd = evaluate::xuping::XuPing2_0_1015_QD::evaluate(&name);

    println!("{xu}\n{xd}")
}
