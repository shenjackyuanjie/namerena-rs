# 名字竞技场 但是 rust

[https://deepmess.com/namerena/](https://deepmess.com/namerena/)

## 当前工程状态

- 依赖升级、API 适配、验证结果与后续优化/移植边界见 [依赖升级与 GPU MVP 迁移说明](docs/dependency-upgrade.md)。
- `gpu-mine` 提供 XP/XD 的 WGSL、CUDA 和 CPU 入口。PBB 另外支持 CPU、CUDA 和 WGSL 共用的测号计划。见 [CPU PBB](miner/docs/pbb-cpu.md)和 [GPU 测号用法](gpu-mine/README.md)。
- CPU miner 已补完安全调度、范围、输出和中断，并做四路 KSA/前缀/生成与 f64 模型优化，见 [CPU 优化与验证报告](miner/docs/cpu-2026-10-30.md)。
- [maker-py](maker-py/README.md) 已弃用，不再维护。

## CPU 矿工

~~~powershell
cargo build -p tswn --release --locked
./target/release/tswn.exe --team shenjack -t 6 -b 10000 -p 550 -x 4000 -d 5000
~~~

通过 `--help` 查看现有 XP/XD 参数。PBB 另走子命令，不改变以上参数和输出。

~~~powershell
cargo build -p tswn --release --locked --features native-pbb
./target/release/tswn.exe pbb --help
~~~

见 [CPU PBB 的构建、用法和一分钟对照](miner/docs/pbb-cpu.md)及 [XP/XD 历史效率摘要](bench/tswn.md)。
