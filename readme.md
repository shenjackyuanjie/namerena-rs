# 名字竞技场 但是 rust

[https://deepmess.com/namerena/](https://deepmess.com/namerena/)

## 当前工程状态

- 依赖升级、API 适配、验证结果与后续优化/移植边界见 [依赖升级与 GPU MVP 迁移说明](docs/dependency-upgrade.md)。
- `gpu-mine` 提供 XP/XD 的 WGSL、CUDA 和 CPU 入口。PBB 有对应的 CUDA 与 WGSL 测号命令。见 [GPU 测号用法](gpu-mine/README.md)。
- CPU miner 已补完安全调度、范围、输出和中断，并做四路 KSA/前缀/生成与 f64 模型优化，见 [CPU 优化与验证报告](miner/docs/cpu-2026-10-30.md)。
- [maker-py](maker-py/README.md) 已弃用，不再维护。

## CPU 矿工

~~~powershell
cargo build -p tswn --release --locked
./target/release/tswn.exe --team shenjack -t 6 -p 550 -x 4000 -d 5000
~~~

通过 `--help` 查看当前参数。见 [历史效率摘要](bench/tswn.md)。
