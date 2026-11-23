# PBB 长跑测号与性能

PBB CUDA 在 GPU 上运行完整评分。PBB WGSL 在 Vulkan 上生成候选并运行整数一级 Q 门，只回读过门 ID，后续由原版 PBB 标量 CPU 完成。两条路线共用队伍、Base16384 编号、计划、CSV 和停止逻辑。

## 日常使用

CUDA 版需要 CUDA Toolkit 12.8、MSVC x64、Python 3。WGSL 版需要可用的 Vulkan 计算驱动、Rust 和 C++17 工具链，支持 Windows x86_64 和 Linux x86_64。两条路线从同一份 PBB 头编译。默认查找仓库旁边的 `PBB/source/pbb/native/src`。自定义位置用 CUDA 的 `-PbbSource` 参数或 WGSL 的 `PBB_SOURCE_DIR` 环境变量。

在仓库根目录构建并用已知有命中的有限区间验收。

~~~powershell
& ./gpu-mine/benchmarks/pbb-cuda/build.ps1 -OutputDir target/pbb-cuda
cargo build -p pbb-wgsl --features native-pbb --release --locked
$plan = './gpu-mine/benchmarks/pbb-cuda/fixtures/football-real.kv'
./target/pbb-cuda/pbb-cuda.exe --mine --out target/pbb-cuda-demo --plan $plan --start 15118306 --count 65537
./target/release/pbb-wgsl.exe --mine --out target/pbb-wgsl-demo --plan $plan --start 15118306 --count 65537
~~~

每次用 `--out` 指定新目录，产物为 `results.csv`。区间是 `[start,end)`，可用 `--count` 或 `--end` 指定结束位置。省略两者后可以持续跑，按 Ctrl+C 会写完当前批次、刷新 CSV 并报告 `next_id`。续跑时用该值作为 `--start`，沿用评分计划、队伍、模板和新的输出目录。

可用 `--team` 和 `--name-template` 覆盖文件或默认输入。模板含一个 `[]`，生成后最多 252 个 UTF-8 字节。复制足球计划可编辑 `min_*`、`ptt_*`、`score_list`、`score_mode`、`collect_mode` 和自定义收集阈值。主命中与蓝号共用一份 CSV，由 `keep` 和 `blue` 两列区分。带引号的名字按 CSV 规则转义。

## 实测

在 RTX 5060 Ti（驱动 621.24，CUDA 12.8 / `sm_120`）上，从 ID 0 开始跑同一 [足球计划](../benchmarks/pbb-cuda/fixtures/football-real.kv)：十族 all、五个 PTT 族、collect_mode=1，批次均为 1,048,576。前核 block 128；新后核按 8 × SM 数派发，这张卡为 288 个 block，每 block 64 线程。

| 同口径实测 | 实际秒数 | 完成候选 | E2E 候选/秒 | T/天 | T/周 |
|---|---:|---:|---:|---:|---:|
| 旧两阶段复测（16 × 32 后核） | 60.001805 | 2,991,587,328 | 49,858,289 | 4.31 | 30.15 |
| **新两阶段（三轮中位数）** | **60.005002** | **14,532,214,784** | **242,183,388** | **20.92** | **146.47** |

三轮分别为 **21.11、20.89、20.92 T/天**，中位数是同机旧版复测的 **4.86 倍**。三轮与旧版在共同 ID 区间的 16 条主命中逐行相同；三轮彼此在各自公共区间的 78 条命中也逐行相同。中位数轮共 78 条主命中、0 条蓝号、274,283 个候选产生非零潜力分。

提速来自三处：

- **稀疏模型点积**：普通分与 PTT 同样按原权重下标顺序，只跳过为零的特征及其交互项，不再物化每份 1,123 项的稠密数组。生成器检查原表一阶、二阶顺序并从原表生成交互下标（包含原有缺项）。QP/QD 仍逐项收回 float，其余仍 double，编译保留 `--fmad=false`。后核每线程编译栈帧由 21,648 B 降到 4,080 B。
- **后核铺满 GPU**：取代旧的 16 个 block，按 SM 数铺开，让各 SM 都能处理压缩候选；多批 block 分摊潜力路径的长尾。
- **前核双步 KSA**：在原有每 lane 独立的转置共享内存上做字节交换，相邻两步提前读取，用首步新值处理地址重叠，保留原写入顺序、V 和首道门控。

### 长跑入口完成后复测

RTX 5060 Ti / Ryzen 7 5800X，足球计划，从 ID 0 开始。WGSL 前筛用 128 线程工作组和 524,288 个候选的批次，后评分用六个 CPU 线程。CUDA 前核为 128 线程，批次为 1,048,576。

| 后端 | 实际秒数 | 完成候选 | 候选/秒 | T/天 | T/周 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WGSL + CPU 评分 | 60.003329 | 3,683,123,200 | 61,381,981 | 5.30 | 37.12 |
| CUDA | 60.004952 | 14,437,842,944 | 240,610,859 | 20.79 | 145.52 |

公共区间的 18 条命中逐行一致。WGSL 共回读 46,995,215 个过 Q 门的候选，其余候选留在 GPU 上筛除。

### 首次移植基线（历史）

同一队伍和枚举口径，RTX 5060 Ti / Ryzen 7 5800X，批次 1,048,576，各实测 60 秒。

| 后端 | 实际秒数 | 完成候选 | E2E 候选/秒 | T/天 | T/周 |
|---|---:|---:|---:|---:|---:|
| 原版标量 CPU，6 线程 | 60.035892 | 232,783,872 | 3,877,412 | 0.34 | 2.35 |
| CUDA 单核移植，block 64 | 60.028109 | 1,763,704,832 | 29,381,316 | 2.54 | 17.77 |
| **CUDA 两阶段，block 128** | **60.007113** | **2,977,955,840** | **49,626,714** | **4.29** | **30.01** |

两阶段比单核快 **68.91%**，约为本次六线程标量 CPU 的 **12.80 倍**。轻量前核只做打包 S-box、原属性序列与同一首道 Q 门；**1.277%** 候选进入后核，后核依然跑完整的 Q/PRE/FREE、各族评分和 PTT。两阶段的 16 条正式主命中内，与单核 CUDA 公共区间的 8 条、与 CPU 公共区间的 1 条均逐行一致。两阶段另有 **56,543** 个候选产生非零潜力分。

计时包括 GPU 生成、全部按原版顺序计算的门控/评分、计数、候选回读、CPU 名字恢复和写 CSV / flush；初始化及预热不计入。每轮按 60 秒起计，截止后完成已提交批次。T = 10¹²，日周是低命中负载下的满载外推，不能套到其它评分计划或高输出条件。CPU 比较项是六线程的原版严格标量评分基准，不代表 PBB 本机 native/AVX 构建的最快速度。

## 支持的评分语义

- 十个族：QP、QD、PP、PD、CQD、CP、FP、WC、FS、PJ；按原 plan 族门顺序、模型权重、SP1/SP2 的 46 个特征与多项式索引。GPUs 不降双精度模型为 f32；QP/QD 仍按原先逐项收回 f32，QP 仍按原规则扣分。
- 原 PBB 的状态生成、前置属性逻辑和技能，影子、守护、护盾、隐匿衍生维不变。PBB 与旧 XP/XD 的翻倍槽条件、HP 排序路径以及 sum 语义不同，不共用旧模型。
- all / any、评分族别名和精确潜力规则；任一已启用族普通达标时跳过所有 PTT。collect_mode 0 / 1 / 2，特殊收集独立于主输出但仍要求原评分门存活；mode 2 保留 sum−HP/3 的整数除法。
- 数据来源由 [生成器](../benchmarks/pbb-cuda/generate_pbb_device_scoring.py) 在构建时读取本机 PBB 的评分头；保留 11 张模型表（CQD 有两张）、19 张门控权重表。PBB 第三方头和生成后的模型数据不入库；构建目录保留原文件哈希。

## 对拍和一分钟基准

构建后的二进制可运行 `--validate` 捕获逐字段对拍，或用 `--seconds 60` 测端到端吞吐。测速产生的临时结果无需入库。

~~~powershell
$exe = './target/pbb-cuda/pbb-cuda.exe'
$wgsl = './target/release/pbb-wgsl.exe'
& ./gpu-mine/benchmarks/pbb-cuda/verify.ps1 -Binary $exe -OutputDir target/pbb-cuda/validation -Path split -Block 128
& ./gpu-mine/benchmarks/pbb-cuda/verify.ps1 -Binary $exe -OutputDir target/pbb-cuda/validation-direct -Path direct -Block 64
& ./gpu-mine/benchmarks/pbb-cuda/verify.ps1 -Binary $wgsl -OutputDir target/pbb-wgsl-validation -Path wgsl -Block 128
$plan = './gpu-mine/benchmarks/pbb-cuda/fixtures/football-real.kv'
& $exe --out target/pbb-cuda/validation-extended --plan $plan --validate 4194305 --block 128 --batch 1048576
& $exe --out target/pbb-cuda/football-60s --plan $plan --backend cuda --seconds 60 --block 128 --batch 1048576
& $wgsl --out target/pbb-wgsl-60s --plan $plan --backend wgsl --seconds 60 --block 128 --batch 524288 --threads 6
~~~

可用 [football-real.kv](../benchmarks/pbb-cuda/fixtures/football-real.kv) 或其它计划覆盖评分阈值、潜力、任意/全量、收集条件和枚举输入；--plan 不传时使用足球评分参数。输入为带一个 [] 的 UTF-8 模板、单槽 Base16384；名字上限 252 字节以适配原 PBB 影子缓冲。--cuda-path direct 用完整单核作对照；split 先运行同一原版一级门，幸存者再执行完整评分。两条 CUDA 路径都使用同序稀疏点积，CPU 保留未修改的原版稠密评分。

### 验证覆盖

- 稀疏版真机对拍覆盖完整验证矩阵。单核和两阶段各 **16 组 / 549,910 个候选**，两阶段另在正式 block、batch 上额外对拍 **4,194,305 个候选**。每组既全捕获逐字段和计数，又按正式筛选验证主结果及特殊集合的并集。十族整数分、潜力、属性、技能、频次、门、集合均精确相等，双精度诊断量用明确误差界比较。覆盖 32 位进位、u64 末端、空前后缀、固定后缀和 252 字节名字上限。
- 双步 KSA 前端在 96,287 项 CPU 对拍上逐值匹配完整 S-box、31 个属性字节、V 和原一级门档位（含 10 种附加模板和 u32/u64 边界）。
- 原足球计划有真实普通命中；诊断计划有仅潜力救回的命中以及三个单样本覆盖五族 PTT。all 与 any 的命中集合一致，但后续未计算列为零。自定义收集覆盖主结果为空、两集合相交和整数除法的边界；正式足球窗口没有自然出现 mode 1 的特殊号，不用诊断阈值冒充生产成绩。对应的四个 mode 1 判据另用原版 CPU 流程的合成边界核对。
- WGSL 同样跑过 16 组、549,910 个候选的完整矩阵，逐 ID 核对淘汰、PRE 和 FREE 三档，再对照原版完整评分与实际收集。已在 RTX 5060 Ti 验证，RX 580 的运行仍待其 Vulkan 驱动环境可用。
