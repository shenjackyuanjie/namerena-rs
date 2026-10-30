# GPU 极限 E2E 优化实测（2026-10-30）

本轮搜索的是完整测号吞吐：ID生成名字、RC4、属性、技能、shadow、同一1034项XP/XD模型、阈值边界精算、命中回读和CSV写入均保留。不是纯RC4测速，没有更换权重或使用有损经验门控。

## 最终结果

RTX 5060 Ti / Vulkan，原模板、shenjack、p550/x4000/d5000，packed S-box、工作组32、批524287、GPU直接生成、非零特征压缩模型。没有超频或修改功耗限制。

| 三轮各10亿、同一区间 | 每轮命中 | E2E秒 | E2E名/秒 | 整进程墙钟秒 | 墙钟名/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第1轮 | 33 | 23.458 | 42629769 | 24.2321 | 41267536 |
| 第2轮 | 33 | 23.554 | 42455683 | 24.2479 | 41240603 |
| 第3轮 | 33 | 23.570 | 42426082 | 24.2306 | 41270184 |

**稳态中位数约4246万名/秒，包含启动/初始化/退出的整进程中位数约4127万名/秒。** 每轮均开启--verify，全部命中完整CPU复核，三轮完整结果CSV SHA256相同。三轮重复同一十亿区间，总计算30亿次，不是覆盖30亿个不同ID。

原始数据：[三轮十亿持续测量](../benchmarks/extreme-sustained-1b.csv)。另有[一亿完整CPU/GPU对照](../benchmarks/extreme-100m-verified.csv)：GPU2.394758s、CPU金标准81.577156s，双方均1个命中，整数/名字逐字段一致、评分误差<10。

## 直接复现

```powershell
$env:RUSTC_WRAPPER = ""
cargo build -p wgpu-mine --release --locked -j 2
./target/release/wgpu-mine.exe -t 2 -p 550 -x 4000 -d 5000 --team shenjack -r 10 -m "我也不知道我要测多少 []" --gpu-input generated --sbox packed --workgroup-size 32 --batch-size 524287 --count 1000000000 --verify --output target/gpu-extreme-new.csv
```

输出路径必须尚不存在。去掉count即可持续测号；Ctrl+C在完整批次写入并flush后停止。普通持续使用不必开verify，本表仍把该复核开销算入。默认输入auto已经对固定宽编码选择GPU生成，默认packed/工作组32/压缩模型；默认批仍保留65536以节约缓冲内存，最快配置显式使用524287。

```powershell
# 三次持续测量，逐样本落盘、包含墙钟、结果哈希一致性检查；不构建，不覆盖旧文件。
& ./wgpu-mine/benchmarks/measure-gpu-extreme.ps1 -Candidates 1000000000 -Repetitions 3 -Output wgpu-mine/benchmarks/gpu-local-new.csv -OutputDir target/gpu-local-new
```

脚本：[measure-gpu-extreme.ps1](../benchmarks/measure-gpu-extreme.ps1)。不要使用GPU的旧--bench纯RC4模式来替代这里的E2E测试。

## 哪些改动有效

1. **共享S-box压成字节。** 每名由256个u32降到64个u32，即1024B→256B。32候选组共享内存从32KiB降到8KiB，提高驻留并行度。每lane独占一列；同一word内交换两个字节时，第二次RMW读取更新后的word，保持精确语义。
2. **GPU直接从u64 ID生成名字。** 上传模板映射一次，每批只传起始ID和数量，不再CPU构建全部String、metadata和UTF8输入。只为实际命中在CPU恢复字符串。Base16384与两种带padding Base64均使用原来的u64大端语义；Number长度可变，auto自动回退上传。
3. **公共KSA前缀缓存。** CPU只预计算一次前导零+模板字面前缀状态，GPU续完第一轮并执行完整第二轮。shadow恢复同一前缀快照，仍完整计算name+?shadow，不解析raw名字/队伍中的@。
4. **压缩非零特征执行同一多项式。** 收集原顺序非零特征，按闭式索引访问原1034项权重，保留非零乘加顺序，减少GPU在零特征分支上的SIMT空转。不是简化模型或少算非零项。
5. **实测工作组和批大小。** 32/64/128均正确，32在本机略快；大批减少同步和头回读开销。极限批输出+回读缓冲总计约176MiB，默认小批更节省资源。

实现：[S-box/模型内核](../mining.wgsl)、[GPU字节编码](../generated-key.wgsl)、[模板与前缀输入](../src/gpu_input.rs)、[Host](../src/mining_gpu.rs)。

### 分阶段实测

| 阶段 | 代表吞吐 | 证据 |
| --- | ---: | --- |
| 本轮原实现重测 | 436万名/s | [基线](../benchmarks/extreme-baseline.csv) |
| packed32、仍CPU生成上传 | 638万名/s | [上传路径](../benchmarks/extreme-uploaded-packed32.csv) |
| GPU直接生成、原稠密遍历 | 2089万名/s | [直接生成](../benchmarks/extreme-generated-packed32-b524287.csv) |
| GPU生成+压缩非零特征，500万对照 | 4133万名/s | [模型对照](../benchmarks/extreme-model-compact-wg32.csv) |
| 三轮各十亿持续运行 | **4246万名/s** | [持续结果](../benchmarks/extreme-sustained-1b.csv) |

阶段数据的数量/批次见各CSV，不应当作同长度的单次精确比值。相对上一轮约414万的实现约10.25倍，相对本轮基线约9.7倍。

--gpu-input uploaded保留原输入对照；--sbox wide保留u32状态；--workgroup-size可选32/64/128（wide仅32）；--dense-model保留原稠密遍历。CSV的sbox/workgroup/model_execution列记录同次A/B的GPU配置，backend=cpu仍始终是f64 CPU金标准。

## 与优化CPU比较

本轮检查时未见旧CPU矿工进程，未停止用户进程；性能测试均串行，期间没有本项目的并行构建或其它性能实验。与前一阶段保留旧矿工占两核的环境不同，因此重新测了原GPU基线及优化CPU。桌面环境结果仍不是硬件全域物理上限。

优化CPU六线程、原条件、固定-b10000、500万正常写盘三次，中位约181.1万名/s，墙钟约179.9万名/s，见[优化CPU新对照](../benchmarks/extreme-cpu-optimized-6t.csv)。GPU持续稳态对它约23.4倍，墙钟比值约22.9倍；二者是独立重复实验、批大小和运行长度不同，不是同进程严格A/B。

指标中相对包内CPU金标准的约34倍，不能冒充对已优化CPU矿工的加速比。CPU矿工另见[优化报告](../../miner/docs/cpu-2026-10-30.md)。

用户旧日志约374.607亿ID、17小时53分44秒：按本轮稳态外推约14分42秒，按整进程速度外推约15分08秒，实际可按**约15分钟**预算。这是相同ID区间估算，不等于旧有损预筛已完整无损计算了所有ID。

## 精度与正确性

| 全量捕获数据集 | 样本 | shadow | XP最大绝对误差 | XD最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| 原模板+Base16384 | 200000 | 37510 | 0.00526540 | 0.05541708 |
| 中文@队伍、多占位符、Base64Url、高位u64 | 50000 | 9349 | 0.00640809 | 0.05410976 |

全部25万样本的属性、技能ID/频率、shadow整数分子与独立CPU金标准逐项一致，评分均<10。捕获没有只验证最终命中。原始数据：[原模板](../benchmarks/extreme-validation-main.csv)、[Unicode/高位](../benchmarks/extreme-validation-unicode.csv)。

额外硬件测试覆盖wide/packed、工作组32/64/128、三种固定编码、重复占位符、raw @/NUL、256B名字与263B内部shadow、u32进位、随机高位、u64最大值、尾批和重复调用。CPU生成助手另有随机1024×3编码对照。当前43项默认测试通过，5项硬件测试另行全部启用通过。

模型仍为同一权重的f32计算，阈值10分余量+边界f64重算策略不变。上述是当前模型/设备的强样本证据，不是所有名字/权重/后端的全域误差证明；换设备或模型请重跑validate。exact-scores/verify仍不绕过GPU粗筛。

```powershell
./target/release/wgpu-mine.exe --team shenjack -m "我也不知道我要测多少 []" --gpu-input generated --batch-size 524287 --validate 200000 --metrics target/extreme-validation-new.csv
cargo test -p wgpu-mine --locked -j 2
cargo test -p wgpu-mine --locked -j 2 -- --ignored --nocapture --test-threads=1
cargo clippy -p wgpu-mine --all-targets --locked -j 2 -- -D warnings
```

## 高输出仍由CSV限制

| 条件 | 候选/命中 | GPU E2E | 包内CPU金标准 |
| --- | ---: | ---: | ---: |
| p550/x1500/d3000 | 1000000 / 173579 | 207.6万名/s | 78.8万名/s |
| p550/x0/d0 | 100000 / 66707 | 56.1万名/s | 40.2万名/s |

见[中命中](../benchmarks/extreme-matches.csv)、[大量输出](../benchmarks/extreme-output.csv)。此时约87%～89%的GPU E2E时间落在CPU名字恢复、CSV格式化/写入等输出阶段，不能把低命中的4246万名/s承诺给任意筛选条件。文件flush计入，没有逐行fsync；持续测试另含正常进程关闭墙钟。

## 本地提交

- 1881569：共享S-box压缩及工作组对照。
- 2444285：GPU按ID生成、模板映射、公共前缀缓存。
- 79283d4：非零特征压缩执行同一完整模型。

全部只在本地提交，没有推送。用户其它删除与原有改动保留。
