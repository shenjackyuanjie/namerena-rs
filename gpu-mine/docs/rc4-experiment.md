# RC4 GPU 移植实验（历史阶段）

> 当前默认入口已是完整测号 CLI，见 [使用说明](../README.md)。本页保留纯 RC4 阶段的实验口径与吞吐摘要，使用 `--bench` 才进入旧实验模式。

## 当前结论

**已在本机 RTX 5060 Ti / Vulkan 上，将 RC4 两轮名字 KSA 的批量端到端吞吐提升到 5800X 六线程对照的约 2.3～2.4 倍。** 这比较的是相同输入、相同 RC4 运算及完整状态输出，不是完整矿工速度，也不是对最优化 PBB 多路交错 CPU 的上限结论。

本实验不计算属性、PRGA、技能、shadow 或虚评，不增加专用单线程模式。队伍 KSA 在 CPU 预计算一次；每个名字的两轮 KSA 在 GPU 内完整执行。CPU 对照同样缓存队伍状态。

## 直接复现

在 workspace 根目录运行：



```powershell
cargo build -p gpu-mine --release --locked -j 2
./target/release/gpu-mine.exe --bench --kernel shared -t 6 --batch-sizes 16384,65536,262144 --warmup 5 --rounds 15 --team shenjack -m "我也不知道我要测多少 []" --csv target/rc4-local.csv
```

推荐批大小 65,536；4,096 已可超过本次六线程基线，256/1,024 的小批量仍由 CPU 更合适。默认内核是 shared，CPU 默认 6 个持久 worker。

- `--kernel shared|private`：选择共享状态或私有状态内核。
- `--batch-sizes`：逗号分隔的批大小；`--warmup` / `--rounds`：预热次数和正式重复次数。
- `-t/--cpu-threads`：CPU 对照线程数，不是 GPU 工作组数。
- `--team`、`-m/--template`、`--start`、`--encoding base16384|number`：候选输入。
- `--csv`：可选的结果文件，覆盖指定路径；不指定时只输出控制台结果。
- 当前不带 `--bench` 时运行完整测号；本页命令均须保留 `--bench`。

## 运算与计时边界

1. 队伍：恒等 S-box → 一轮密钥为 `[0] + UTF-8(team)` 的 KSA，双方缓存该状态。
2. 每个名字：复制队伍 S-box → 两轮密钥为前导零加 UTF-8 字节的 KSA；每轮重置局部 j 和密钥游标。
3. 输出：完整的 256 字节置换状态，保持输入顺序。不是校验和，也不是仅返回命中计数。
4. GPU 端到端时间包含 CPU 输入打包、queue 上传、命令提交、完整状态回读、映射及复制到返回 Vec。
5. CPU 时间包含任务派发、6 个持久 worker 的计算、结果返回与归并；worker 缓冲复用，双方都预热。
6. **名字生成、设备/管线初始化、CPU 线程创建不计入双方吞吐。** 名字在 CPU 预生成；每个批大小单独输出生成耗时。不能把本表当作包含名字生成、属性和虚评的完整测速。
7. GPU 内核时间使用硬件 timestamp query，包含 pass 内初始化、KSA 和状态打包写出；不包含上传/回读。它用于定位瓶颈，不作为端到端加速比。
8. 默认每种批大小先全量 CPU/GPU 对照，再预热，再测多轮取中位数；每个正式计时轮结束后再次全量逐字节比较，验证成本不计入吞吐。

首轮正式实验每种批大小预热 3 次、测量 9 次；独立新进程复测预热 5 次、测量 15 次。两个内核串行执行，正式测速期间没有并行运行本项目其他 GPU 测试或构建。更早的并行诊断数据不用于下表。

初始化另行报告：本轮观察到约 3.8～14.5 秒，未计入稳态吞吐；短任务不一定能摊销冷启动成本。

## 本机实测（2026-10-30）

- CPU：AMD Ryzen 7 5800X，8 核 / 16 逻辑处理器；本次固定 6 个 worker。
- GPU：NVIDIA GeForce RTX 5060 Ti；Vulkan；驱动 32.0.16.1074。
- 编译：Windows x86_64，workspace Release，rustc 1.101.0-nightly；无额外 target-cpu/native 参数。
- 队伍：shenjack；模板：我也不知道我要测多少 []；u64 大端字节经 Base16384 编码。该样本每个完整名字为 49 个 UTF-8 字节。

### 共享 S-box 独立复测

| 批大小 | CPU 六线程耗时 | GPU 端到端耗时 | CPU 百万名字/秒 | GPU 百万名字/秒 | 端到端加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1.1701 ms | 0.6832 ms | 3.501 | 5.995 | 1.71× |
| 16,384 | 4.0344 ms | 1.7532 ms | 4.061 | 9.345 | 2.30× |
| 65,536 | 16.6606 ms | 7.1246 ms | 3.934 | 9.199 | 2.34× |
| 262,144 | 66.6737 ms | 27.5063 ms | 3.932 | 9.530 | 2.42× |

所有记录 verified=true，全部状态一致。

首轮 shared 的 256/1,024 批大小分别只有六线程 CPU 的 0.31× / 0.87×，因此不能声称 GPU 对任意任务都更快。

65,536 批的硬件内核时间约 0.9283 ms，即约 7,060 万名字/秒，但实际端到端只有约 920 万名字/秒：**上传、完整状态回读和 CPU 打包已成为主要剩余成本。** 本轮依然完整返回每个名字 256 字节，未用省略回读来制造速度。

### 为什么共享状态有效

- [private 内核](../main.wgsl) 每个 invocation 使用私有 u32 S-box，大批量时随机访问的局部状态容易受缓存/存储压力限制。本机 262,144 批仅约 0.98× CPU，65,536 批约 1.06×。
- [shared 内核](../rc4-shared.wgsl) 每工作组 32 个候选，使用 32KiB 转置共享状态：slot×32+lane。每列只属于一个 invocation，在当前设备的 bank 布局下减少独立候选间的冲突；候选内部仍严格串行执行 RC4 交换。
- 每个名字仍是独立状态，未跨候选共享可变状态，也未改变 KSA 算法。没有需要跨 lane 同步的数据依赖，轮内不加 barrier；默认 wgpu 工作组初始化策略保留。
- 多工作组一次派发，设备、管线、队伍状态、输入输出缓冲持续复用；不再一批 256 个就同步等待。
- 输入 UTF-8 连续紧凑打包；输出按四个状态字节打成一个 u32，使回读从旧原型的 1,024 字节/名字降到 256 字节/名字，信息完整保留。

shared 需要设备支持每工作组至少 32KiB 共享内存；不支持时会提示选择 private。CUDA warp/bank 解释只用于本机性能分析，不能外推所有 GPU 都有相同收益。

## 正确性与验证

[CPU 金标准及六线程对照](../src/cpu.rs) 包含独立朴素参考，CPU 热路径不逐候选分配密钥。默认测试覆盖空/中文/NUL/长名字、线程数与多批次顺序；GPU 硬件测试对 private/shared 两种内核、空/中文/ASCII 队伍、emoji、256 字节名字、31/32/33 与 63/64/65 等工作组边界、513 尾批、缩小再扩大批次、非法输入后恢复逐项对照。

```powershell
cargo test -p gpu-mine --locked -j 2
cargo test -p gpu-mine --locked -j 2 gpu_matches_independent_cpu_rc4 -- --ignored --nocapture --test-threads=1
cargo clippy -p gpu-mine --all-targets --locked -j 2 -- -D warnings
```

GPU 主机实现见 [gpu.rs](../src/gpu.rs)，实验 CLI 与计时统计见 [main.rs](../src/main.rs)。最终验证：默认 8 项测试通过；1 项默认忽略的硬件测试已手动执行并通过（同时覆盖 private/shared）；`cargo clippy -p gpu-mine --all-targets --locked -j 2 -- -D warnings` 无警告通过；Release 构建、workspace 全目标 check、本模块 rustfmt 与 `git diff --check` 均通过。另已完成只读语义/竞态/计时复核，无必要修正。原 CPU 矿工的历史虚评测试问题未在本轮改动。

## 下一步边界

本轮已经证明：**RC4 可以实际移到 GPU，并在足够大批量下超过本机六线程 CPU。** 后续若接回矿工，先保留同样的分层对拍，再考虑让属性计算/筛选紧随 GPU RC4，减少回传完整 S-box；CPU 留下 f64 虚评复核。那属于下一阶段，不能把当前 RC4 吞吐直接当成最终测号吞吐。
