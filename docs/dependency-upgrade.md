# 依赖升级与 GPU MVP 迁移边界

> 本文记录依赖升级阶段的实现与验证状态。后续已完成 [GPU E2E 测号 CLI、完整属性技能模型、误差验证与吞吐实测](../gpu-mine/README.md)；当前使用方式以该说明为准。早期 [纯 RC4 实验](../gpu-mine/docs/rc4-experiment.md) 单独保留。

## 本阶段范围

2026-10-30 按 crates.io 返回的最新稳定版本更新现存 Rust workspace，适配破坏性 API，并进行构建、回归及 GPU 验证。本阶段不实现完整测号 MVP，不新增专用单线程模式，不进行 GPU 性能调参。

- [workspace 清单](../Cargo.toml) 只保留 `miner` 和 `gpu-mine`。`openbox` 已被用户删除，本次移除失效成员引用，没有恢复目录。
- 保留用户对 [cacluate.rs](../miner/src/cacluate.rs) 的修改，以及已删除的其他历史目录。
- [maker-py](../maker-py/README.md) 按用户要求标记为 DEPRECATED，保留历史代码，不再升级或验证 Python 依赖。

## 依赖版本

下表旧版本为原清单约束；实际解析版本由 [Cargo.lock](../Cargo.lock) 固定。间接依赖同时通过 `cargo update` 刷新到当前约束允许的最新版本，不强行覆盖上游要求的主版本。

| 直接依赖 | 原约束 | 升级后最低版本 |
| --- | --- | --- |
| clap | 4.5 | 4.6.7 |
| base16384 | 0.1.0 | 0.1.0（已为最新稳定版） |
| tracing | 0.1.40 | 0.1.44 |
| tracing-subscriber | 0.3.18 | 0.3.23 |
| chrono | 0.4 | 0.4.45 |
| colored | 3 | 3.1.1 |
| crossbeam | 0.8.4 | 0.8.5 |
| base64 | 0.22.1 | 0.23.1 |
| windows-sys | * | 0.61.2（去掉通配符） |
| bytemuck | 1.15.0 | 1.25.2 |
| flume | 0.11.0 | 0.12.0 |
| pollster | 0.3.0 | 1.0.1 |
| wgpu | 0.19.4 | 30.0.1 |

依赖清单分别见 [miner](../miner/Cargo.toml) 和 [gpu-mine](../gpu-mine/Cargo.toml)。移除了没有任何源码调用的 `reqwest` 和 GPU 侧 `serde`、`serde_json` 直接依赖。`reqwest` 最初适配到 0.13.5 的 `rustls` feature 后，经代码扫描确认没有网络实现，因此按 YAGNI 移除，避免无用 HTTP/TLS、异步运行时及原生加密库构建成本。

## API 与兼容性修复

- [CPU CLI](../miner/src/main.rs) 原先将 `-d` 同时用于 XD 阈值和 debug，现保留 `-d/--xd-exp`，debug 改为仅 `--debug`；增加完整示例参数解析测试。
- [名字生成](../miner/src/generate.rs) 增加 Base64 标准/URL 字母表、大端字节、padding 和中文模板多占位符测试，确保依赖更新不改变名字编码。
- [GPU 宿主](../gpu-mine/src/main.rs) 适配 `request_adapter` 的 Result、单参数 `request_device` 与描述符默认值、计算管线的可选入口/编译选项/cache、ErrorScopeGuard、PollType 及映射视图的 Result。错误返回非零退出码。
- [GPU 着色器](../gpu-mine/main.wgsl) 与宿主统一使用 u32 输入，补齐名字长度/字节绑定、独立 storage 输出与 staging 回读，修复名字 RC4 的字节来源和前导零周期。仍只返回两轮 RC4 置换状态，不包含完整属性、技能或筛选。
- 内嵌 WGSL，运行不依赖当前目录；修复 `chunks_exact` 丢尾批，回读仅保留真实名字；增加 UTF-8 字节长度验证、5 项默认测试和 1 项手动 GPU/CPU 对照测试。保留 16×16 工作组形状，不做 GPU 调参。

## 验证方法

工程继续使用 [nightly 工具链声明](../rust-toolchain)，没有更改系统 Rust 配置。

```powershell
# 在 namerena-rs 目录运行。内存充足时可以自行提高 -j。
$env:RUSTC_WRAPPER = ""
cargo check --workspace --all-targets --locked -j 2
cargo test --workspace --all-targets --locked -j 2
cargo build --workspace --release --locked -j 2
```

本机默认高并发构建曾触发 Windows 页面文件不足（错误 1455，伴随无法映射标准库 metadata）；降低并发完成后续验证，不调整系统页面文件。清空 `RUSTC_WRAPPER` 只影响该次 PowerShell 进程，用于排除本地全局 sccache 对构建排障的影响。

### 实际验证结果

本轮实际工具链为 `rustc 1.101.0-nightly (822182db4 2026-10-13)`，验证平台为 Windows x86_64。

| 验证 | 结果 |
| --- | --- |
| `cargo check --workspace --all-targets --locked -j 2` | 通过 |
| `cargo build --workspace --release --locked -j 2` | 通过，生成三个 Release 二进制 |
| `cargo test --workspace --all-targets --no-fail-fast --locked -j 2` | 28 通过、6 失败、1 默认忽略；失败是同 3 个历史测试在两个 CPU 二进制中重复执行 |
| 手动 `gpu_matches_independent_cpu_rc4` | 通过；2 个队伍 × 513 个名字 × 256 项逐项对照 |
| 从工作区根直接运行 GPU Release 二进制 | 退出码 0；NVIDIA GeForce RTX 5060 Ti / Vulkan / DiscreteGpu，证明不依赖 shader 文件的运行目录 |
| CPU Release 帮助及完整示例参数的解析单测 | 通过 |
| 本轮修改文件的 rustfmt 检查及 `git diff --check` | 通过 |

测试完整日志分别保留为 [workspace 测试日志](../target/dependency-workspace-tests.log)、[GPU 对照日志](../target/dependency-gpu-runtime.log)、[Release 构建日志](../target/dependency-workspace-release.log) 和 [GPU 发布版冒烟日志](../target/dependency-gpu-release-smoke.log)。

结合删除失效 workspace 成员、去掉未使用直接依赖及更新依赖图，lock 中的包条目从 370 减少到 180。这反映依赖图精简，不等于已实测测号吞吐提升。

### 已确认的历史测试失败

完整 workspace 测试的 `tswn` 目标为 14 通过、3 失败，`xp2` 目标为 9 通过、同样 3 失败；GPU 目标默认 5 通过、1 项硬件测试默认忽略，该项已另外手动执行并通过。新增的 5 项 CLI/生成回归测试全部通过；失败的是 [虚评测试](../miner/src/evaluate/xuping/eval.rs#L298-L344) 中以下 3 项：

| 测试 | 当前实际值 | 原测试期望 |
| --- | --- | --- |
| xuping_13_test | 5820.122513651108 | 5799.586821819176 |
| xuping_20_1015_test | 3393.784908154375 | 3603.4389333619315 |
| xuping_20_1015_qd_test | 3438.2108464843195 | 3639.8920896689424 |

用 `git archive HEAD` 将升级前源码、依赖清单与 lock 导出到隔离的 target 子目录，只将基线 workspace 成员收窄为 `miner` 以便加载，复跑得到 **9 通过、同样 3 失败，实际值逐项一致**。因此它们不是本轮依赖升级引入的回归。没有修改模型、跳过原测试或改写期望值来制造全绿；后续应核对模型版本、shadow 逻辑与测试基准。

## 后续 MVP：哪些可移植，哪些要重写

| 模块 | 结论 | 下一步 |
| --- | --- | --- |
| CLI、队伍、模板、名字编码 | 可迁移到新入口；本轮已修复 CLI 短参数冲突 | 保留 `start/end/t/p/x/d/team/r/b/n/m`；新增范围、线程数、间隔和长度校验 |
| CPU 属性、技能、XP/XD 模型 | 可复用完整计算路径 | 作为 GPU 对照及命中候选复核，保持同一模型和阈值语义 |
| 旧 `replace_name` 快速预筛 | 不可当成无损阈值筛选复制 | 固定 `24/165/250/380` 早退会漏掉较低 `-p` 下的候选；采用无启发式完整构造作基准 |
| 多线程调度 | 需要重写 | 去掉跨线程可变别名与 busy-wait；调度器独占范围、worker 消息回传、关闭通道并 join |
| CSV 输出 | 保留字段含义，重写写入通道 | 单 writer、正确 CSV 转义、`write_all`、零命中也创建表头、错误向主线程传播 |
| RC4 GPU 原型 | 先建立正确性基线 | 完整属性/技能链路尚未实现；不能把依赖适配当成测号功能完成 |
| XP/XD 直接搬到 WGSL | 不能直接等价移植 | 当前 CPU 模型使用 f64 和 shadow 重算；WGSL 常规路径不等价支持，f32 会改变阈值边界。建议 GPU 粗筛后 CPU 同模型复核 |
| 网络、上传、服务端 | 当前无实现可移植，也不属于本地 MVP | 不新增网络服务或恢复已删除模块 |
| 专用单线程、绑核路径 | 不移植 | `-t` 保留 CPU 准备/复核 worker 数，不改解释为 GPU workgroup 数 |

### 筛选语义与已知历史问题

[现有计算链路](../miner/src/cacluate.rs#L363-L390) 是：预筛 → 八围严格 `> p` → 技能 → QD 转 `u32` 后 `>= d` → XP 转 `u32` 后 `>= x`，三个条件是 AND。`-r` 当前表示目标 batch 时长，而不是网络上报。

[旧预筛](../miner/src/name.rs#L395-L457) 的固定早退独立于命令行阈值。例如 [x@x 的已有测试](../miner/src/name.rs#L932-L939) 八围为 564.6667，虽然大于 550，仍可能被固定预筛提前排除。本轮没有悄悄改变此语义；MVP 阶段需明确采用“完整阈值筛选”还是“兼容旧候选集”。

[旧调度器](../miner/src/cacluate.rs#L136-L260) 有跨线程可变共享状态、任务归属/丢弃、预热范围不钳制及结束不 join 等问题；[主入口](../miner/src/main.rs) 还存在范围调整可能溢出的逻辑。不要以现有矿工的短范围运行结果作为 ID 恰好覆盖一次的证明。

### 推荐后续验收

1. 先抽出共享 CPU 核心与 CLI/生成，不复制旧调度器；完整属性/模型计算作为金标准。
2. 建立安全的多线程准备/复核和单输出器，再接 GPU 属性粗筛。不要另写专用单线程模式。
3. 覆盖候选数 `0/1/255/256/257/513`、非零 start、中文及空名字、末尾短批和接近 `u64::MAX` 的范围，证明每个 ID 恰好一次。
4. 覆盖阈值边界、shadow、CSV 逗号/双引号及零命中输出，再验收用户给出的长命令。
5. 当前 `tswn.exe` 属于 CPU 包，`gpu-mine.exe` 属于 GPU 原型。后续决定替换入口或独立发布，避免同一 target 下两个包输出同名二进制。

## 对照 tswn-core 与 PBB 的优化候选

已只读分析用户指定的两个参考工程。本阶段没有改动参考仓库，也没有把其评分模型、经验门控或战斗运行时整体引入矿工；以下列为后续性能阶段的候选，尚无测号吞吐基准。

| 优先级 | 可迁移点与参考 | 预期作用与正确性前提 |
| --- | --- | --- |
| 中 | 连续编号的进位式名字生成：[PBB advance_buf](../../PBB/source/pbb/native/src/engine.hpp#L697-L762) | 重写发生进位的位置、固定前后缀复用；先为本项目各编码定义精确的编号到名字映射，不能照搬 PBB 的固定字宽/固定容量假设。GPU 可保留编号，命中后 CPU 重建名字 |
| 高 | 队伍 KSA 和固定模板前缀检查点：[tswn-core RC4](../../tswn-core/crates/tswn_core/src/rc4.rs#L271-L299)、[PBB load_team/load_prefix](../../PBB/source/pbb/native/src/name.hpp#L111-L158) | 同队伍、同模板批次只计算一次固定前缀。必须验证前缀字节与 KSA 起始状态一致；不能跨不同队伍或不同位置的可变字节复用 |
| 高 | 独立候选的 4 路交错 KSA：[tswn-core update_interleaved](../../tswn-core/crates/tswn_core/src/rc4.rs#L407)、[PBB batch4](../../PBB/source/pbb/native/src/name.hpp#L280-L344) | 用多个独立状态隐藏单条 RC4 的串行访存依赖；不是把同一个 RC4 状态的步骤强行并行。长度相同可走专门快路径，短尾/不同长度保留通用处理 |
| 高 | 仅提取前 31 个属性候选：[PBB finish_load_name](../../PBB/source/pbb/native/src/name.hpp#L432-L443) | 先筛八围，避免为绝大多数落选者构造完整名字/技能状态；命中者需要的后续状态必须保留或可等价重建。只能迁移提取与压缩，不复制其固定 V 早退 |
| 高 | 技能与完整模型按需求计算：[PBB 流水线](../../PBB/source/pbb/native/src/name.hpp#L8-L11)、[紧凑/完整属性构建对拍](../../tswn-core/crates/tswn_core/src/runtime/prepared_init/init.rs#L782-L877) | 属性先过滤，命中后再付出技能、shadow 和 f64 模型计算成本；复用 worker 本地状态，不为全部候选创建战斗对象或做无用堆分配 |
| 高 | 不可变任务块、关闭后排空和完成归并：[PBB 任务队列](../../PBB/source/pbb/native/src/engine.hpp#L141-L194)、[结果归并](../../PBB/source/pbb/native/src/engine.hpp#L912-L971) | 借鉴块所有权和按完成计数，修复旧调度器漏任务；不复制长期溢出的 int 任务编号或随机模式允许重复/尾块跑满的范围语义。MVP 先不扩展 UI、复杂恢复协议 |
| 中 | 由剩余项理论上界推导的提前退出：[PBB 上界说明](../../PBB/source/pbb/native/src/scoring_1124.hpp#L903-L925) | 借鉴“即使剩余贡献取最大值也达不到阈值才淘汰”的方法，针对本项目 p/x/d 重新证明。负权重、浮点舍入与边界须保守处理，不能复制其他模型的常量 |
| 必做 | 先移植独立对拍测试：[安全 KSA/PRGA](../../tswn-core/crates/tswn_core/src/rc4_tests.rs#L3-L19)、[前缀/交错对拍](../../tswn-core/crates/tswn_core/src/rc4_tests.rs#L128-L199)、[属性 filter 对拍](../../tswn-core/crates/tswn_core/src/runtime/prepared_init/init.rs#L740-L744) | 给后续 unsafe/SIMD/缓存和 GPU 改写提供共同金标准；只复用匹配本项目语义的中间状态，不将不同版本的最终分数互当真值 |
| 必做 | 范围、跨 chunk 尾批及线程数一致性：[PBB flex 测试](../../PBB/source/tests/test_flex.py#L298-L348) | 在优化前证明没有漏号、重复号或线程数改变结果；借鉴测试结构，不引入专用单线程运行模式 |

补充正确性约束：Rust 的 `Rc4KeySchedulePrefix` 从恒等排列开始，而矿工名字的前缀缓存起点是已经完成队伍 KSA 的排列，因此不能不改初始化就直接替换。CPU 的多路交错也不应未经实测搬成“一个 GPU invocation 保存 4 份 S-box”，否则可能抬高私有存储压力。

属性/技能对拍还可借用 [PBB 独立过滤验证](../../PBB/source/pbb/native/autotune_filter_verify.cpp#L27-L71)：先比完整置换，再比稳定压缩数组、八围、技能，最后才比旧虚评。注意 PBB 的 SIMD 技能过滤会重写完整 name_base，不能依赖之前排序的 HP 头仍保留；使用独立小数组或明确重建。

**明确不直接移植：**

- [PBB 的固定 V 早退](../../PBB/source/pbb/native/src/name.hpp#L445-L465) 返回的是可能尚未算完的部分和，不是对任意八围阈值都无损的剪枝。
- [PBB Q 门控](../../PBB/source/pbb/native/src/scoring_1124.hpp#L859-L867) 及 [技能/模型族门控](../../PBB/source/pbb/native/src/scoring_1124.hpp#L927-L939) 明确由样本 Pareto 拟合并允许漏召回，不可作为严格阈值测号的默认路径。
- PBB 的 1124 项模型及 tswn-core 的新版 SQP/DS4 输出与本项目旧 XP/XD 口径不同。它们适合验证共享的 RC4、属性/技能中间数据，不足以直接修正旧虚评测试期望。
- 战斗调度、目标选择、战斗 PRGA 等优化不属于候选枚举热路径，不为测号 MVP 引入完整战斗模拟。
- CPU 的 AVX2/AVX-512 内在函数、指针别名提示与 f64 模型不能原样搬成 WGSL；需按 GPU 数据布局与数值精度重新实现并逐层对拍。
