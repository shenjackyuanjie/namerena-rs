# gpu-mine

| 评分 | WGSL | CUDA |
| --- | --- | --- |
| XP/XD 2.0 | `gpu-mine --backend wgsl` | `gpu-mine --backend cuda` |
| PBB | `pbb-wgsl --mine`，GPU 完成 Q 门、技能和保守 G3，CPU 原版精算 | `pbb-cuda --mine`，完整 GPU 评分 |

四条路线支持按 ID 测号、CSV 写入和 Ctrl+C 结束。XP/XD 使用八围、XP、XD 阈值。PBB 支持十族评分、PTT 潜力和蓝号收集。两种 PBB 后端共用评分计划、范围参数和输出。见 [PBB 构建与用法](docs/pbb-cuda.md)。

## 一分钟实测

| 路线 | 候选/秒 | T/天 | T/周 |
| --- | ---: | ---: | ---: |
| XP/XD WGSL | 86,244,621 | 7.45 | 52.16 |
| XP/XD CUDA | 122,815,187 | 10.61 | 74.28 |
| PBB WGSL 双门控 + 原版 CPU 评分，三轮中位数 | **453,637,845** | **39.19** | **274.36** |
| PBB CUDA，三轮中位数 | **751,467,998** | **64.93** | **454.49** |
| XP/XD CPU，6 线程 | 2,636,930 | 0.23 | 1.59 |
| PBB CPU，6 线程原版标量 | 3,877,412 | 0.34 | 2.35 |

RTX 5060 Ti，Ryzen 7 5800X。PBB 的 WGSL 和 CUDA 均取各自三轮 60 秒的中位数；其余各项沿用同机的一分钟结果。XP/XD 使用 p550/x4000/d5000，PBB 使用十族足球计划和五族 PTT。日、周数字按实测端到端速度折算，T = 10¹²。仓库只记录基准摘要与复现办法。另见 [XP/XD CPU 实测](docs/wgsl-pbb-2026-11-23.md) 和 [PBB 实测](docs/pbb-cuda.md#实测)。

用新入口复测 XP/XD 吞吐时，定时脚本会在写出 CSV 表头后计时，收完批次并检查命中行数。脚本退出时会清理本轮的临时 CSV 和日志。

~~~powershell
$env:WGPU_BACKEND = 'vulkan'
py -3 -B ./gpu-mine/benchmarks/timed-mine.py --mode wgsl --binary target/release/gpu-mine.exe --seconds 60 --batch-size 3145728 --gpu-input generated
py -3 -B ./gpu-mine/benchmarks/timed-mine.py --mode cuda --binary target/release/gpu-mine.exe --seconds 60 --batch-size 2097120 --gpu-input generated
~~~

## 想学习 GPU 实现？

从 [GPU 测号学习路线](docs/gpu-learning-00-start.md) 开始：包含零基础技术入门；四份 WGSL 的完整代码直接写在文档中，并配有详细中文注释。

## XP/XD 测号

在仓库根目录运行以下命令。CUDA 构建需要 Windows x64、CUDA Toolkit 和 MSVC。默认 CUDA 目标为 Blackwell `sm_120`，其它 NVIDIA 架构通过 `-Arch` 选择。

```powershell
cargo build -p gpu-mine --release --locked
./target/release/gpu-mine.exe --backend wgsl --team shenjack -p 550 -x 4000 -d 5000 -m "我也不知道我要测多少 []"

& ./gpu-mine/benchmarks/build-cuda-ab.ps1 -CudaOnly
./target/release/gpu-mine.exe --backend cuda --team shenjack -p 550 -x 4000 -d 5000 -m "我也不知道我要测多少 []"
```

CUDA 使用 GPU 生成的 Base16384、packed S-box 和稀疏模型，回传的候选由 CPU 精算。若构建时指定了输出目录，使用 `--cuda-dll` 传入该 DLL 路径。WGSL 支持下方的全部名字编码。

不指定终点时枚举至 u64::MAX（不包含），Ctrl+C 在当前完整批次写入、flush 后停止，并报告下一 ID；可用 --start 从该 ID 继续到新文件。

```powershell
# 先跑有限范围；输出路径必须尚不存在，避免覆盖结果。
./target/release/gpu-mine.exe -t 6 -p 550 -x 4000 -d 5000 --team shenjack -r 10 -m "我也不知道我要测多少 []" --start 0 --count 1000000 --output target/gpu-results.csv
# 同一完整计算语义的 CPU 对照，不走旧有损预筛。
./target/release/gpu-mine.exe --backend cpu -t 6 -p 550 -x 4000 -d 5000 --team shenjack -m "我也不知道我要测多少 []" --count 1000000 --output target/cpu-results.csv
```

### 常用参数

| 参数 | 含义 |
| --- | --- |
| --backend | wgsl（默认，别名 gpu）、cuda、cpu |
| --start / --end | 严格半开区间 [start,end)，支持空范围和接近 u64::MAX 的末尾 |
| --count | 从 start 开始的有限数量，与 end 互斥；溢出明确拒绝 |
| -t / --thread-count | CPU对照/验证/上传路径worker数，默认6；GPU直接生成时不参与逐名计算 |
| -p / -x / -d | 八围严格 >p，XP/XD 按 Rust as u32 饱和截断后 >=x/d，条件为 AND |
| -m / --name-template | 替换全部 []，最终名字最多 256 个 UTF-8 字节 |
| -n / --name-gen | base16384（默认）、base64-std、base64-url、number；二进制编码使用 u64 大端字节 |
| -b / --batch-size | GPU/CPU 批大小，默认 65536，不会为整除线程数修改终点 |
| -r / --report-interval | 进度报告秒数，默认 10，不是网络上报 |
| --output / -o | 新 CSV 文件；禁止覆盖；- 表示 CSV 写 stdout，日志仍在 stderr |
| --output-dir | 未指定文件时创建带时间戳的新结果文件，默认 namerena |
| --gpu-input auto/uploaded/generated | 固定宽编码auto默认GPU直接生成；Number回退上传 |
| --sbox packed/wide | 默认字节packed，保留wide对照 |
| --workgroup-size | 32（默认）、64、128；wide仅32 |
| --dense-model | 使用旧稠密遍历作对照；默认压缩非零特征，同一模型和权重 |
| --exact-scores | 对全部回传候选重算 f64；默认只在非零评分阈值 10 分附近重算 |
| --verify | 对命中做完整 CPU 属性/技能/模型复核，并对回传候选精算，适合验收而非测速 |
| --validate N | 捕获全部 N 个候选做完整对拍并统计推理误差，不执行常规筛选输出 |
| --e2e-bench | 同范围、同阈值交替运行 GPU/CPU，并校验 CSV（整数/名字精确，评分误差<10） |
| --metrics | validate/e2e-bench 的新指标 CSV 文件；禁止覆盖 |

CSV 固定 16 列：full_name,id,name,team,hp,attack,defense,speed,agility,magic,resistance,wisdom,property,xp,xd,skills。full_name 与 CPU miner 一致，为 `name@team`（team 为空时退化为 `name@name`）；技能为排序后的 id:freq 分号列表，包含在单列；逗号、引号、CRLF、NUL 和 Unicode 交给 csv 库正确处理。零命中仍有表头，写入/flush 失败返回非零退出码。

## 默认评分策略与误差预算

- GPU 使用当前同源 2.0 模型权重，以 f32 同时计算 44 个特征的 1034 项 XP/XD。队伍、属性和技能仍是精确整数运算，没有模型简化、权重替换或经验门控。
- GPU 粗筛保留阈值以下 10 分余量；距任意非零阈值不超过 10 的候选用整数属性/技能/shadow 分子在 CPU 上重算 f64 再筛选。远离阈值的命中直接输出 GPU 评分，避免每个命中都付出 CPU 推理成本。
- --exact-scores 对全部回传候选重算；--verify 还核对完整 CPU 原始计算。但二者仍有 GPU 粗筛，不是对未知平台全域无漏号的形式证明；完全 CPU 金标准可使用 --backend cpu。
- 阈值为零时遵循 Rust 负分转 u32 饱和为零的语义，不因负分误删候选。
- 原矿工 replace_name 中固定 24/165/250/380 早退没有移植；低 -p 不会继承该有损预筛。名字/队伍按原始 UTF-8 处理，不将其中 @ 当作分隔符。
- 评分余量可靠性依赖当前模型和实际浮点误差；下面是全量样本证据，不是所有名字/队伍/模型/硬件后端的全域误差证明。换模型或设备应重跑 validate。运行时 refined_*_max_error 指标只统计实际精算的回传候选，0 不代表所有候选误差为 0。

### 已执行的完整误差验证

| 数据集 | 样本 | shadow | XP 最大绝对误差 | XD 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| shenjack + 原中文模板 + Base16384 | 100000 | 18808 | 0.00514504 | 0.05530066 |
| 中文@队伍 + 原始@名字 + 数字，非零 start | 50000 | 9495 | 0.00507856 | 0.04769579 |

全部样本的 8 属性、16 技能 ID、16 技能频率及 shadow 整数分子逐项相同；推理误差严格 <10。捕获模式没有只测最终命中。

```powershell
./target/release/gpu-mine.exe --team shenjack -t 6 -m "我也不知道我要测多少 []" --validate 100000 --metrics target/validation-new.csv
```

## 首轮 E2E 实测（极限优化前）

2026-10-30，Ryzen 7 5800X（6 worker）+ NVIDIA GeForce RTX 5060 Ti / Vulkan，Release，批 65536。先跑 1 轮完整预热，再跑 3 轮正式测量取中位数；GPU/CPU 顺序交替，测速期间没有并行构建或其他本项目性能测试。

另有用户原来的旧无限测号进程自当天 01:27 一直运行，占约两核 CPU。本任务没有终止或修改它；GPU 和 CPU 对照在同一背景负载下串行运行，因此下面是该桌面负载下的实测，不是独占整机的峰值。需要纯净峰值时，由用户先自行结束原后台任务，再用相同命令复测。

| 负载 | 数量/命中 | GPU E2E | 内置 CPU 六线程 E2E | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 原参数 p550/x4000/d5000 | 5000000 / 0 | 414.2 万名/s（1.2071s） | 118.4 万名/s（4.2235s） | 3.50× |
| 有实际输出 p550/x1500/d3000 | 1000000 / 173579 | 137.4 万名/s（0.7278s） | 76.9 万名/s（1.2999s） | 1.79× |
| 大量输出 p550/x0/d0 | 100000 / 66707 | 47.8 万名/s（0.2091s） | 38.6 万名/s（0.2589s） | 1.24× |

每一轮输出的全部 ID、名字、属性、技能均一致，两个评分逐记录误差 <10。该 CPU 对照是 GPU 包内的完整金标准池，不等同另行优化后的 tswn.exe 性能；CPU miner 队友的独立结果另记。

### 另行优化后的 CPU miner

上表是输出格式相同的包内 CPU 金标准对照。另行优化后的 CPU miner 使用四路 KSA、前缀复用与 f64 流式模型，在相同五百万原阈值下最终复核为 **167.2 万名/s**，含启动和 sync_all 的墙钟为 **166.1 万名/s**，相对正确性修复基线提升 **4.46×**。与 GPU 的 414.2 万名/s 两组独立实测相比约为 **2.48×**，不应继续把包内基准的 3.50× 当成对这个优化版 CPU 的加速比。

两边批大小及输出格式不同，且高输出时优化版 CPU 序列化可更快；详细负载与限制见 [CPU 报告](../miner/docs/cpu-2026-10-30.md)。零命中原参数不存在 CSV 正文大小差异，但初始化仍应另外考虑。

### 计时解释

- 表中为稳态 E2E：从创建本轮 CSV 开始，包含并行名字生成、CPU 输入打包、GPU 上传/计算/压缩/候选回读、必要边界精算、CSV 格式化/写入/flush。不是只报 GPU 内核。
- GPU 设备/管线和 CPU worker 初始化、进程启动/析构另列，不混入每轮吞吐；本次进程初始化约 2.7～8.8 秒。短范围加上冷启动后不一定胜过 CPU，持续测号才能摊销。基准 setup_s 是同进程共享初始化，不是独立 CPU 进程的冷启动耗时。
- 原参数五百万范围是零命中，结果并未隐藏；中、高命中率另外实测。不能拿零命中的 414 万名/s 承诺任意筛选条件。
- 大量输出时主要瓶颈是 CSV 格式化/写入。此前对所有候选都做 f64 精算时，该场景仅约 0.96×；改为边界精算后提高到约 1.24×，仍受输出吞吐限制。
- 文件 flush 计入，但没有每条结果 fsync；这衡量正常缓冲文件写入，不是断电持久化带宽。

```powershell
./target/release/gpu-mine.exe --e2e-bench --team shenjack -t 6 -p 550 -x 4000 -d 5000 -m "我也不知道我要测多少 []" --count 5000000 --rounds 3 --warmup 1 --metrics target/e2e-new.csv --output-dir target/e2e-new-results
```

## 测试与实现

主要实现：[CPU 完整金标准](src/core.rs)、[GPU 内核](mining.wgsl)、[GPU Host 与压缩回读](src/mining_gpu.rs)、[并行工作池](src/pool.rs)、[输入与区间](src/input.rs)、[输出](src/output.rs)、[真实 CLI 测试](tests/cli.rs)。

当前 GPU 包默认测试 **43 项通过、5 项硬件测试默认忽略**；这5项已另外串行启用，均通过。严格 Clippy（-D warnings）和 Release 构建通过。CPU miner 的独立测试、历史模型断言与优化结果单独报告。

已覆盖：原始固定数组金标准、完整稳定过滤、技能所有分支与连续 PRGA、1034 项多项式索引、空/中文/emoji/NUL/@、256 字节名字和 263 字节内部 shadow、属性分数/评分阈值、31/32/33/63/64/65/257/513 等尾批、u64 边界、线程数改变不漏不重、工作池错误恢复、零命中表头、CSV 全字段往返及 writer 写入/flush 错误、拒绝覆盖、标准输出无日志污染、取消后的完成游标与表头刷新、总轮数溢出拒绝。

Base16384 的最高合法码点 U+8DFF 暴露了第三方 UTF8 decode 的上界排除缺陷。测号只使用 encode，兼容原行为；测试改用独立 14bit 恢复作为边界金标准，没有修改全局依赖代码，也没有跳过高值编号。

```powershell
cargo test -p gpu-mine --locked -j 2
# 硬件测试串行执行，不与性能测量并行。
cargo test -p gpu-mine --locked -j 2 -- --ignored --nocapture --test-threads=1
cargo clippy -p gpu-mine --all-targets --locked -j 2 -- -D warnings
```

纯 RC4 的旧基准仍通过 --bench 使用，历史数据与口径见 [RC4 实验记录](docs/rc4-experiment.md)。

## 本地提交

后续极限优化提交1881569、2444285、79283d4及其完整验收见 [极限报告](docs/gpu-extreme.md)。

- 6aff41b：依赖升级与已验证 RC4 基线。
- d786074：完整 CPU/GPU 属性、技能、shadow 与容差模型链。
- 878e440：E2E CLI、安全工作池、名字范围、CSV 和集成测试。
- 773beb7：仅边界精算、完整精算选项、取消/轮数边界验证。

CPU 矿工早期的正确性和优化分别见 edbf51e、e8d3525。详见 [CPU 优化与验收报告](../miner/docs/cpu-2026-10-30.md)。
