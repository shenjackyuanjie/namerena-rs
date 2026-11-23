# PBB 完整评分 CUDA 基准

这是独立的 **PBB 评分后端**，不是把原 XP/XD 换权重，也不替换现有 wgpu-mine。GPU 从 ID 生成同一批 Base16384 名字；CPU 金标准直接编译未修改的本机 PBB 原版标量头。没有移植 PBB GUI、排行榜、词库或其它字符集枚举；基准 CSV 集中保留主结果、特殊收集和所有评分字段，不是 PBB 原文本文件格式。

## 实测

在 RTX 5060 Ti（驱动 621.24，CUDA 12.8 / `sm_120`）上，从 ID 0 开始跑同一 [足球计划](../benchmarks/pbb-cuda/fixtures/football-real.kv)：十族 all、五个 PTT 族、collect_mode=1，批次均为 1,048,576。前核 block 128；新后核按 8 × SM 数派发，这张卡为 288 个 block，每 block 64 线程。

| 同口径实测 | 实际秒数 | 完成候选 | E2E 候选/秒 | T/天 | T/周 |
|---|---:|---:|---:|---:|---:|
| 旧两阶段复测（16 × 32 后核） | 60.001805 | 2,991,587,328 | 49,858,289 | 4.31 | 30.15 |
| **新两阶段（三轮中位数）** | **60.005002** | **14,532,214,784** | **242,183,388** | **20.92** | **146.47** |

三轮分别为 **21.11、20.89、20.92 T/天**，中位数是同机旧版复测的 **4.86 倍**。[60 秒原始数据](../benchmarks/pbb-cuda/football-sparse-60s.csv)保留各轮完整计数与时钟。三轮与旧版在共同 ID 区间的 16 条主命中逐行相同；三轮彼此在各自公共区间的 78 条命中也逐行相同。中位数轮共 78 条主命中、0 条蓝号、274,283 个候选产生非零潜力分。

提速来自三处：

- **稀疏模型点积**：普通分与 PTT 同样按原权重下标顺序，只跳过为零的特征及其交互项，不再物化每份 1,123 项的稠密数组。生成器检查原表一阶、二阶顺序并从原表生成交互下标（包含原有缺项）。QP/QD 仍逐项收回 float，其余仍 double，编译保留 `--fmad=false`。后核每线程编译栈帧由 21,648 B 降到 4,080 B。
- **后核铺满 GPU**：取代旧的 16 个 block，按 SM 数铺开，让各 SM 都能处理压缩候选；多批 block 分摊潜力路径的长尾。
- **前核双步 KSA**：在原有每 lane 独立的转置共享内存上做字节交换，相邻两步提前读取，用首步新值处理地址重叠，保留原写入顺序、V 和首道门控。

### 首次移植基线（历史）

同一队伍和枚举口径，RTX 5060 Ti / Ryzen 7 5800X，批次 1,048,576，各实测 60 秒。

| 后端 | 实际秒数 | 完成候选 | E2E 候选/秒 | T/天 | T/周 |
|---|---:|---:|---:|---:|---:|
| 原版标量 CPU，6 线程 | 60.035892 | 232,783,872 | 3,877,412 | 0.34 | 2.35 |
| CUDA 单核移植，block 64 | 60.028109 | 1,763,704,832 | 29,381,316 | 2.54 | 17.77 |
| **CUDA 两阶段，block 128** | **60.007113** | **2,977,955,840** | **49,626,714** | **4.29** | **30.01** |

两阶段比单核快 **68.91%**，约为本次六线程标量 CPU 的 **12.80 倍**。轻量前核只做打包 S-box、原属性序列与同一首道 Q 门；**1.277%** 候选进入后核，后核依然跑完整的 Q/PRE/FREE、各族评分和 PTT。两阶段的 16 条正式主命中内，与单核 CUDA 公共区间的 8 条、与 CPU 公共区间的 1 条均逐行一致。两阶段另有 **56,543** 个候选产生非零潜力分。见 [60 秒原始数据](../benchmarks/pbb-cuda/football-60s.csv)。

计时包括 GPU 生成、全部按原版顺序计算的门控/评分、计数、候选回读、CPU 名字恢复和写 CSV / flush；初始化及预热不计入。每轮按 60 秒起计，截止后完成已提交批次。T = 10¹²，日周是低命中负载下的满载外推，不能套到其它评分计划或高输出条件。CPU 比较项是六线程的原版严格标量评分基准，不代表 PBB 本机 native/AVX 构建的最快速度。

## 支持的评分语义

- 十个族：QP、QD、PP、PD、CQD、CP、FP、WC、FS、PJ；按原 plan 族门顺序、模型权重、SP1/SP2 的 46 个特征与多项式索引。GPUs 不降双精度模型为 f32；QP/QD 仍按原先逐项收回 f32，QP 仍按原规则扣分。
- 原 PBB 的状态生成、前置属性逻辑和技能，影子、守护、护盾、隐匿衍生维不变。PBB 与旧 XP/XD 的翻倍槽条件、HP 排序路径以及 sum 语义不同，不共用旧模型。
- all / any、评分族别名和精确潜力规则；任一已启用族普通达标时跳过所有 PTT。collect_mode 0 / 1 / 2，特殊收集独立于主输出但仍要求原评分门存活；mode 2 保留 sum−HP/3 的整数除法。
- 数据来源由 [生成器](../benchmarks/pbb-cuda/generate_pbb_device_scoring.py) 在构建时读取本机 PBB 的评分头；保留 11 张模型表（CQD 有两张）、19 张门控权重表。PBB 第三方头和生成后的模型数据不入库；构建目录保留原文件哈希。

## 构建、对拍、复现

在 namerena-rs 根目录用 PowerShell 7 运行，已装好 CUDA Toolkit 12.8、MSVC x64 和 Python 3。默认查找旁边的 PBB/source/pbb/native/src，也可显式传 -PbbSource。所有输出都用新目录。

~~~powershell
& ./wgpu-mine/benchmarks/pbb-cuda/build.ps1 -OutputDir target/pbb-cuda
$exe = './target/pbb-cuda/pbb-cuda.exe'
& ./wgpu-mine/benchmarks/pbb-cuda/verify.ps1 -Binary $exe -OutputDir target/pbb-cuda/validation -Path split -Block 128
& ./wgpu-mine/benchmarks/pbb-cuda/verify.ps1 -Binary $exe -OutputDir target/pbb-cuda/validation-direct -Path direct -Block 64
$plan = './wgpu-mine/benchmarks/pbb-cuda/fixtures/football-real.kv'
& $exe --out target/pbb-cuda/validation-extended --plan $plan --validate 4194305 --block 128 --batch 1048576
foreach ($i in 1..3) {
    & $exe --out "target/pbb-cuda/football-60s-$i" --plan $plan --backend cuda --seconds 60 --block 128 --batch 1048576
    if ($LASTEXITCODE) { throw "正式测速失败：$i" }
}
~~~

可用 [football-real.kv](../benchmarks/pbb-cuda/fixtures/football-real.kv) 或其它计划覆盖评分阈值、潜力、任意/全量、收集条件和枚举输入；--plan 不传时使用足球评分参数。输入为带一个 [] 的 UTF-8 模板、单槽 Base16384；名字上限 252 字节以适配原 PBB 影子缓冲。--cuda-path direct 用完整单核作对照；split 先运行同一原版一级门，幸存者再执行完整评分。两条 CUDA 路径都使用同序稀疏点积，CPU 保留未修改的原版稠密评分。

### 验证覆盖

- [稀疏版真机矩阵](../benchmarks/pbb-cuda/verification-sparse.csv)：单核和两阶段各 **16 组 / 549,910 个候选**，两阶段另在正式 block、batch 上额外对拍 **4,194,305 个候选**。每组既全捕获逐字段和计数，又按正式筛选验证主结果及特殊集合的并集。十族整数分、潜力、属性、技能、频次、门、集合均精确相等，双精度诊断量用明确误差界比较。覆盖 32 位进位、u64 末端、空前后缀、固定后缀和 252 字节名字上限。[初版矩阵](../benchmarks/pbb-cuda/verification.csv)留作历史对照。
- 双步 KSA 前端在 96,287 项 CPU 对拍上逐值匹配完整 S-box、31 个属性字节、V 和原一级门档位（含 10 种附加模板和 u32/u64 边界）。
- 原足球计划有真实普通命中；诊断计划有仅潜力救回的命中以及三个单样本覆盖五族 PTT。all 与 any 的命中集合一致，但后续未计算列为零。自定义收集覆盖主结果为空、两集合相交和整数除法的边界；正式足球窗口没有自然出现 mode 1 的特殊号，不用诊断阈值冒充生产成绩。对应的四个 mode 1 判据另用原版 CPU 流程的合成边界核对。
- 这是 **CUDA 的 PBB 完整评分模式**。WGSL 尚未移植 PBB 的双精度模型和取整，因此不把另一个 XP/XD 负载的 10.48 T/天当作 PBB 数据。
