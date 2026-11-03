# GPU 测号学习路线：从零开始

这套文档以仓库在编写时的实现为准；源码会变化，行号是方便查找的快照。它面向会读一点 Rust、但从没写过 GPU 程序的人。先读这里，不需要先掌握图形渲染。

## 怎么读

1. [技术入门与整个测号流程](gpu-learning-01-basics.md)：先了解 GPU、wgpu、WGSL、内存和 CPU/GPU 分工，再看本项目如何计算、筛选与验证。
2. [正式测号内核：完整代码与中文注释](gpu-learning-02-mining-wgsl.md)：文档直接贴出完整 `mining.wgsl`，代码旁就有中文解释，不必反复跳转源码。
3. [GPU 名字生成及 RC4 实验内核：完整代码与中文注释](gpu-learning-03-other-wgsl.md)：同样内嵌 `generated-key.wgsl`、`main.wgsl` 和 `rc4-shared.wgsl` 的完整代码。前者是正式内核的一段替换代码，后两者服务于 RC4 实验，不是完整测号。
4. 能读懂后再看 [使用说明与精度契约](../README.md)、[极限优化报告](gpu-extreme.md) 和 [native 对照](native.md)。性能数字是该报告记录的硬件和负载下的实测，不是所有机器、所有命中率都能达到的保证。

## 先抓住一张全景图

`ID 区间 + 模板 + 队伍 + 阈值` 由 CPU 接收；CPU 创建 GPU 资源并提交计算。固定宽度编码可以在 GPU 上从 ID 生成名字字节，Number 等回退 CPU 上传。然后每个有效 invocation（一次内核调用实例）处理一个候选：

`名字字节 → 队伍/名字 RC4 状态 → 属性 → 属性筛选 → 技能 → shadow（按需）→ XP/XD → GPU 粗筛 → 原子分配结果槽`

GPU 完成后 CPU 先回读计数，再只回读候选记录；CPU 做必要的 f64 边界精算和最终筛选，为命中恢复名字并写 CSV。调用位置可从 [GPU 主机端](../src/mining_gpu.rs) 的 `GpuMiner::create` 和 `execute` 开始追踪；命令行和批次管理见 [main.rs](../src/main.rs)。

## 手算一次 GPU 怎么分工

假设一个批次有 70 个候选、每工作组 32 个 invocation。CPU 发起 `ceil(70 / 32) = 3` 个工作组，GPU 一共启动 96 个 invocation：全局编号 0～69 各处理一个候选，70～95 在入口判断中直接返回。每组内的 `lane` 编号是 0～31；不同组可以重复使用相同 lane 编号，因为各工作组有独立的 `workgroup` 存储。编号规则和尾批返回见 [mining.wgsl:196–202](../mining.wgsl#L196-L202)。

同一个工作组可以并发处理不同名字，但**同一个名字的 RC4 步骤仍有先后依赖**，不能简单把它的 256 次交换拆成 256 个互不相关的线程。本实现通过大量候选之间的并行来利用 GPU；实际执行的硬件线程编组和调度仍由硬件/驱动决定。

## 用一小段数据理解紧凑 S-box

RC4 的 S-box 是 256 个字节的排列。WGSL 这里以 `u32` 数组承载共享数据：wide 模式每个值占一个 32 位字，每名 1024 字节；packed 模式一个 32 位字装四个 8 位槽，每名 256 字节。`LANES=32` 时分别占 32 KiB 与 8 KiB 工作组共享内存。

例如同一 lane 的四个槽值是十六进制 `11 22 33 44`，按代码中的位位置放在一个数里就是 `0x44332211`：槽 0 在最低 8 位，槽 3 在最高 8 位。读取槽 2 相当于右移 16 位，再用 `& 255` 取低 8 位，得到 `0x33`。相邻 lane 的字交错存放；每个 lane 只改自己那一列。有关具体公式与交换同一字时的注意事项，见 [read_sbox / write_sbox](../mining.wgsl#L58-L74)。

## 一次最小上手实验

在 `namerena-rs` 目录，先构建，再用有限范围和容易命中的阈值观察结果；输出文件必须不存在：

```powershell
cargo build -p wgpu-mine --release --locked -j 2
./target/release/wgpu-mine.exe --team shenjack -m "测试[]" -p 0 -x 0 -d 0 --count 32 --output target/gpu-learning-results.csv
```

这不是测速命令。`-p 0 -x 0 -d 0` 让读者更容易看到记录；名字、队伍、属性和评分的列含义以 [CLI 说明](../README.md) 为准。想比较 CPU 和 GPU，使用 `--validate` 捕获指定数量的全部候选，与 CPU 完整对照，而不是只看命中的名字：

```powershell
./target/release/wgpu-mine.exe --team shenjack -m "测试[]" --validate 1000 --metrics target/gpu-learning-validation.csv
```

`--verify` 复核返回命中，`--exact-scores` 精算回传候选，但它们都不自动证明粗筛之前的所有候选绝无漏报。当前 GPU f32 使用 10 分余量，样本误差结果和设备范围见 [精度说明](../README.md#默认评分策略与误差预算)。换设备或权重时重新验证；需要完整 CPU 金标准可使用 `--backend cpu`。

## 看完后试着回答

- 把批次改成 33 个候选时，32-lane 工作组需要启动几组？为什么多余 invocation 能提前退出？
- 为什么工作组共享内存虽然名为“共享”，这里的每列却只由一个 lane 访问？如果引入跨 lane 协作，是否仍能随意提前返回？
- 上传模式和 GPU 生成模式的 `name_bytes` 分别存什么？为什么 `NameInfo.offset` 在两种模式含义不同？
- 为什么一个结果只由一个 invocation 写，但多个 invocation 仍要用原子计数器分配槽？
- 为什么对拍少量样本得出的最大误差，不能直接证明所有名字和所有显卡都不会误筛？

先自己画出数据流，再对照后两篇导读核对答案。遇到 `ptr<function, ...>`、`select`、`atomicAdd` 或位运算时，不妨逐步记录一个 invocation 的局部变量和内存状态，而不是同时跟踪全部线程。
