# 从零理解本项目的 GPU 测号链路

本文面向第一次接触 GPU 计算的读者。目标是看懂一个 ID 如何变成 CSV 的一行，而不是先学完图形学。代码链接相对于本文；行号、依赖版本和默认值可能随代码变动，请以当前文件为准。核对时包版本为 0.1.0、wgpu 依赖为 30.0.1，见 [Cargo.toml:1–18](../Cargo.toml#L1-L18)；实际锁定版本以 [Cargo.lock](../../Cargo.lock) 为准。

## 1. 全景：一批 ID 怎样变成结果

```text
主机 CPU：解析参数、选设备、分批、准备资源
       ├─ 固定宽编码：上传模板映射一次，每批给起始 ID 和数量
       └─ 上传路径：CPU 生成本批名字，打包 UTF-8 字节及偏移后上传
              ↓
GPU：每个 invocation 处理一个候选
 ID/名字 → RC4 状态 → 八围预筛 → 技能 → 必要时 shadow
         → f32 XP/XD → 留有余量的粗筛 → 原子追加候选记录
              ↓
主机 CPU：读回计数，再只读回候选；必要时 f64 精算并最终筛选
         → 生成/取回命中名字 → 可选 CPU 复核 → 写 CSV 并 flush
```

这是数据流，不是三个程序。“主机”指运行 Rust 程序并控制 GPU 的 CPU 一侧；“设备”指选中的 GPU 计算设备。GPU 不会自动调用 Rust 函数：主机必须创建资源、提交计算命令并取回结果。主线见 [main.rs:208–304](../src/main.rs#L208-L304)、[mining_gpu.rs:296–489](../src/mining_gpu.rs#L296-L489) 和 [mining.wgsl:196–364](../mining.wgsl#L196-L364)。

## 2. CPU、GPU、WebGPU、wgpu、WGSL

CPU 擅长通用控制、文件处理和分支较多的任务；GPU 通常用大量执行单元批量处理相似任务，重视吞吐。两者都能并行，也都能计算；小任务不一定在 GPU 上更快。本项目的许多候选互不依赖，适合批量处理；控制流程、必要的精算和 CSV 仍在 CPU。

- **WebGPU** 是跨平台 GPU 接口规范，包括计算和图形，并不是某块显卡。
- **wgpu** 是 Rust 的 GPU 库，提供与 WebGPU 概念对应的接口，可使用本机后端。本项目请求高性能 adapter，并记录设备和后端，见 [mining_gpu.rs:131–177](../src/mining_gpu.rs#L131-L177)。实际可用性取决于设备、驱动和后端，不能保证总选中独立显卡。
- **WGSL** 是 shader 使用的语言。这里的 shader 是计算程序，不是画图程序：主机创建 compute pipeline，指定 `mine` 入口，之后派发，见 [mining.wgsl:196–200](../mining.wgsl#L196-L200)。

可把 pipeline 想成“准备好的工序”，bind group 是“工序使用的资源清单”，dispatch 是“按批下单”；它们本身不是计算结果。

## 3. 调度：workgroup 和 invocation

一个 **invocation** 是执行一次 shader 入口；若干 invocation 组成一个 **workgroup**。项目把一维全局编号 `position.x` 当批内候选下标，每个有效 invocation 处理一个候选；局部编号 `lane` 用来选择自己的 S-box 列。默认组大小为 32，另支持 64、128；wide S-box 只支持 32，见 [mining_gpu.rs:21–35](../src/mining_gpu.rs#L21-L35)。这里的 lane 是代码对组内位置的称呼，不保证等于硬件 warp 或 wave 的宽度。

例如 65 个候选、每组 32：主机派发 ceil(65/32)=3 组，共 96 个 invocation。下标 0～64 有效；65～95 由入口的 `index >= params.count` 检查直接返回。因此尾批不会漏算或越界，见 [mining_gpu.rs:379–400](../src/mining_gpu.rs#L379-L400)、[mining.wgsl:196–205](../mining.wgsl#L196-L205)。不同组的完成顺序不确定，也不一定全部同时驻留。

同组可以使用 workgroup 共享内存，但本实现按 lane 分列，每个候选独占自己的 S-box，不依赖跨 lane barrier，见 [mining.wgsl:1–86](../mining.wgsl#L1-L86)。若以后引入组内协作或 barrier，必须重新检查早退和同步规则。

## 4. 内存与绑定：两边讲同一种布局

buffer 可理解为一段带用途标记的字节存储。主机用 bind group 将 buffer 对应到 shader 的 `@group(0) @binding(n)`；GPU 输出并不是普通 Rust 数组，结果要经过复制和映射。定义见 [mining.wgsl:5–48](../mining.wgsl#L5-L48)，创建及绑定见 [mining_gpu.rs:181–247](../src/mining_gpu.rs#L181-L247)：

| binding | 资源 | 用途 |
| --- | --- | --- |
| 0 | uniform team_state | 队伍 KSA 状态；生成路径为公共名字前缀快照 |
| 1 | 只读 storage names | 上传路径为偏移/长度；生成路径复用为前缀及编码元数据 |
| 2 | 只读 storage name_bytes | 上传路径为紧凑名字字节；生成路径为逐 u32 的模板映射 |
| 3 | 可读写 storage output | 计数头与紧凑候选记录 |
| 4 | uniform params | 数量、阈值、捕获开关、生成起始 ID 的高低位 |
| 5 | 只读 storage weights | XP/XD 权重对 |

uniform 用于这里的小型只读参数，storage 用于较大数组和可写数据；这不承诺某种物理内存。`var<private>` 是 invocation 独有数据，例如编码缓存；`var<workgroup>` 在同组共享但本项目按列隔离各候选，见 [generated-key.wgsl:1–8](../generated-key.wgsl#L1-L8)、[mining.wgsl:50–73](../mining.wgsl#L50-L73)。

S-box 有 256 个字节状态。wide 的每项占一个 u32，即每候选 1024 字节；packed 每四个字节装入一个 u32，即 256 字节。32 候选一组分别需要 32 KiB 或 8 KiB 工作组存储。packed 的同 word 字节交换要注意更新顺序，见 [mining.wgsl:58–86](../mining.wgsl#L58-L86)。packed 不保证在所有设备上最快；创建时还检查设备共享内存、buffer 和派发限制，见 [mining_gpu.rs:117–150](../src/mining_gpu.rs#L117-L150)。

输出头是四个 u32 原子计数，共 16 字节；每条记录 44 个 32 位字，共 176 字节，记录批内下标、八围、技能、shadow 分子和两个 f32 分数。shader 原子加法领取不同槽位，避免同时写同一槽；领取顺序不等于 ID 顺序。主机排序并检查重复，见 [mining.wgsl:15–30](../mining.wgsl#L15-L30)、[mining.wgsl:343–364](../mining.wgsl#L343-L364)、[mining_gpu.rs:423–477](../src/mining_gpu.rs#L423-L477)。修改结构时须同时核对两端字段、偏移、对齐和长度。

## 5. 两条输入路径

模板必须含 `[]`，会替换**所有**占位符。示例：Number 编码时 ID 42、模板 `名字[]-[]` 得到 `名字42-42`；Base64Std 编码 ID 0 时，先把 u64 表示成 8 字节大端，再编码为 `AAAAAAAAAAA=`。名字最多 256 个 **UTF-8 字节**，不是 256 个汉字，见 [input.rs:17–46](../src/input.rs#L17-L46)。ID 范围是半开区间 `[start, end)`，例如 `[7, 10)` 只有 7、8、9；`--count` 与 `--end` 互斥，范围加法溢出报错，见 [input.rs:50–75](../src/input.rs#L50-L75)、[main.rs:65–71](../src/main.rs#L65-L71)、[main.rs:162–167](../src/main.rs#L162-L167)。

| 路径 | 操作 | 适用情况 |
| --- | --- | --- |
| generated | CPU 预先生成并上传模板映射和公共前缀 KSA 快照；每批传起始 ID、数量。GPU 用两个 u32 处理高低位及进位，生成编码字节；正常输出时 CPU 只为最终命中恢复名字。 | Base16384、Base64Std、Base64Url |
| uploaded | CPU 工作池逐 ID 生成字符串；主机打包 UTF-8 字节及各名字的偏移/长度并上传。后续计算仍由 GPU 做。 | 所有编码；Number 在 auto 下回退到此路径 |

依据：[gpu_input.rs:14–56](../src/gpu_input.rs#L14-L56)、[generated-key.wgsl:1–28](../generated-key.wgsl#L1-L28)、[mining_gpu.rs:296–340](../src/mining_gpu.rs#L296-L340)。默认 `--gpu-input auto` 选择可用的生成路径；显式 `generated` 配 Number 会报错，不会悄悄回退，见 [main.rs:435–458](../src/main.rs#L435-L458)。模板映射区分字面字节和编码字节位置，故支持重复占位符；生成路径减少逐名主机生成和传输，不是省掉最终命中的名字。

## 6. 一名候选：RC4、属性、技能、shadow

RC4 在这里用于兼容项目的确定性名字计算，不是在传输中加密 CSV，也不应据此选用 RC4 保护新系统。先把 0～255 的排列当 S-box。**KSA** 根据输入字节反复交换 S-box；项目每轮密钥字节有隐式前导零，先按队伍执行一轮，再按名字执行两轮。队伍状态可以复用；生成模式还缓存队伍后的名字公共前缀，接续第一轮并完整执行第二轮。CPU 金标准见 [core.rs:42–67](../src/core.rs#L42-L67)，快照见 [gpu_input.rs:39–53](../src/gpu_input.rs#L39-L53)，GPU 见 [mining.wgsl:118–135](../mining.wgsl#L118-L135)。原始名字和队伍按 UTF-8 字节分别处理，其中的 `@` 不被拆为分隔符。

接着对排列做固定映射并保持原顺序抽出 128 项，算八围：HP 是前 10 项排序后中间四项之和加 154；其余七围是各组三数中位数加 36，见 [core.rs:70–102](../src/core.rs#L70-L102)。属性不合格可提早结束，节省后续计算。

对于通过属性筛的候选，**PRGA** 从 S-box 连续生成字节，驱动两轮技能洗牌，再计算 16 个技能频率；两轮共享状态，不各自重置，见 [mining.wgsl:220–262](../mining.wgsl#L220-L262)。如果有效技能 ID 24 存在，程序以原名字追加 `?shadow` 再计算属性并构造 shadow 特征，否则该值为零，见 [mining.wgsl:264–288](../mining.wgsl#L264-L288)。内部 shadow 名字可比 256 字节原名字多 7 字节。

模型用 44 个特征，包括八围、技能和 shadow；计算 44 个线性项及 990 个二次项，共 1034 项，加上常数，得到 XP 和 XD 评分。CPU 用原权重 f64，GPU 上传 f32 权重并计算；默认压缩非零特征后按原索引访问同一模型，`--dense-model` 是遍历方式对照，不是更换模型，见 [core.rs:162–230](../src/core.rs#L162-L230)、[mining.wgsl:294–342](../mining.wgsl#L294-L342)。

## 7. 过滤、回读与 CSV

1. **八围预筛**：总值按 `HP / 3 + 其余七围之和` 理解；代码比较分子 `HP + 3 × 其余七围之和`，避免提前整除。必须严格大于 `-p`，即分子 `> 3*p`，见 [core.rs:29–40](../src/core.rs#L29-L40)。
2. **GPU 评分粗筛**：XP 和 XD 均需可能达到各自阈值；非零阈值给 GPU 分数加 10 分余量，避免边界附近因 f32 误差过早丢弃。阈值为零时不因负分淘汰；非有限分数送主机报错，见 [mining.wgsl:343–354](../mining.wgsl#L343-L354)。
3. **两段回读**：每批清空输出头，提交计算并复制 16 字节计数头；知道候选数后，若非零，再将相应记录复制到可映射 staging buffer。`map_async`、设备等待、读取、unmap 是必要步骤，不能把 submit 当作完成。设备支持时间戳时还读取计时，见 [mining_gpu.rs:363–422](../src/mining_gpu.rs#L363-L422)、[mining_gpu.rs:493–513](../src/mining_gpu.rs#L493-L513)。
4. **CPU 精筛**：候选距任一非零评分阈值不超过 10 时，用回传的整数属性、技能和 shadow 分子重算 f64；其余默认沿用 GPU f32 分数。最终条件是 AND：八围严格大于 p、XP 和 XD 分别按 Rust `as u32` 饱和截断后大于等于阈值。例如 4000.9 转成 4000，负分转成 0。见 [mining_gpu.rs:432–475](../src/mining_gpu.rs#L432-L475)、[core.rs:29–35](../src/core.rs#L29-L35)。
5. **写出**：主机用批起点加记录下标恢复 ID，生成或查找名字；`--verify` 对最终 GPU 命中做完整 CPU 复核，再写 CSV。零命中仍写表头，输出文件必须是新文件。16 列依次为 `full_name,id,name,team,hp,attack,defense,speed,agility,magic,resistance,wisdom,property,xp,xd,skills`；skills 是按 ID 排序的 `id:freq` 列表。CSV 库处理逗号和引号转义，见 [main.rs:148–152](../src/main.rs#L148-L152)、[main.rs:266–304](../src/main.rs#L266-L304)、[output.rs:8–54](../src/output.rs#L8-L54)。

粗筛通过的“候选”不一定是最终命中。GPU 原子追加顺序不稳定，主机排序后按 ID 顺序写出；写入或 flush 失败会返回错误。缓冲并 flush 不等于逐行 fsync 或断电持久化保证。

## 8. 精度与验收：样本证据不是全域证明

f32 和 f64 位宽、表示精度不同，浮点计算还会受顺序和硬件实现影响；整数属性和技能对照一致，不代表评分逐位一致。10 分余量是本工程策略，不是 WebGPU 对所有模型、设备和数据的误差上界保证。

- `--exact-scores` 对**已回传候选**全部重算 f64；`--verify` 也启用该精算，并对**最终命中**用 CPU 从名字重新算完整属性、技能和分数。二者仍经过 GPU 粗筛，无法检查已被粗筛丢掉的名字。
- `--validate N` 使用 capture-all，跳过常规属性与分数筛选，逐条对照 N 个候选的 CPU 结果，测 GPU 近似评分绝对误差并要求均小于 10；它不是常规输出模式。见 [main.rs:320–397](../src/main.rs#L320-L397)、[mining_gpu.rs:443–467](../src/mining_gpu.rs#L443-L467)。
- `--backend cpu` 是不依赖 GPU 粗筛的完整 CPU 对照；`--e2e-bench` 对同范围交替运行并比较 CSV：非评分字段精确一致，两个评分误差均小于 10，见 [main.rs:400–432](../src/main.rs#L400-L432)。

已有报告记录特定设备、模型和数据上的 25 万全量样本；这是历史测量，不是本文重新运行的结果，见 [gpu-extreme.md:70–81](gpu-extreme.md#L70-L81)。更换 GPU、驱动、后端、模型或数据分布时重新运行全量捕获和同范围 CPU 对照。运行时只对实际重算的回传候选统计误差，显示 0 不证明被筛掉的候选误差为 0。

### 可执行的入门验收

在仓库根目录 PowerShell 运行；示例不保证本机一定有可用 GPU。指定输出和指标路径必须**尚不存在**；先确认路径。命令会构建并生成目标文件：

```powershell
cargo build -p gpu-mine --release --locked -j 2
# 全量捕获 65 个候选，覆盖 32 线程组尾批；对照 CPU，不生成普通结果 CSV。
./target/release/gpu-mine.exe --team test -m "[]" --start 0 --validate 65 --metrics target/gpu-learn-validation.csv
# GPU 与 CPU 同范围比较：生成两侧 CSV 和指标，输出目录使用新目录。
./target/release/gpu-mine.exe --team test -m "[]" -p 0 -x 0 -d 0 --start 0 --count 65 --e2e-bench --warmup 0 --rounds 1 --metrics target/gpu-learn-e2e.csv --output-dir target/gpu-learn-e2e
# 常规测试；硬件测试默认忽略，另行串行执行。
cargo test -p gpu-mine --locked -j 2
cargo test -p gpu-mine --locked -j 2 -- --ignored --nocapture --test-threads=1
```

验收看退出码、全量样本数、XP/XD 最大误差是否严格小于 10、E2E CSV 对比是否成功；也要检查最终 CSV 表头、ID 范围和命中数。65 只是学习样本，不代表生产规模或所有输入已验证。CLI 边界与 CSV 测试见 [cli.rs:60–195](../tests/cli.rs#L60-L195)，GPU 相关测试见 [mining_gpu.rs:516–729](../src/mining_gpu.rs#L516-L729)。

## 9. 性能：测完整链路，不只测 GPU

吞吐量是候选数除以时间。内核时间只覆盖 GPU 计算阶段；E2E 还包含本轮 CSV 创建、输入生成/上传、计算、回读、必要精算、名字恢复、输出和 flush。设备/管线初始化与进程启动退出要单独计时；短任务可能被初始化淹没。见 [main.rs:208–304](../src/main.rs#L208-L304)、[mining_gpu.rs:379–410](../src/mining_gpu.rs#L379-L410)。

可调因素包括批大小、工作组大小、S-box 布局、生成/上传和模型遍历；大批可能减少同步开销，也会增加缓冲和设备限制压力。更多命中会增加回读、CPU 处理和 CSV I/O，不能拿低命中数字承诺高命中吞吐。历史极限报告的约 4246 万名/秒是 RTX 5060 Ti / Vulkan、指定参数和批大小、三轮各十亿且每轮 33 命中的稳态中位数；整进程墙钟中位约 4127 万名/秒，重复的是同一区间，见 [gpu-extreme.md:7–17](gpu-extreme.md#L7-L17)。这不是本次实测，也不是任意硬件的保证。比较时记录设备/驱动、编码、阈值、范围和命中数、批大小、预热、是否 verify、输出介质及计时口径；独立重复和不同 CPU 实现的数字不应直接当严格 A/B。

## 10. 逐步阅读路线

1. 看 [main.rs:45–127](../src/main.rs#L45-L127) 的参数和 [input.rs:17–46](../src/input.rs#L17-L46) 的名字编码；手算 ID 0 的 Base64 和 65 个候选的三组派发。
2. 看 [main.rs:208–280](../src/main.rs#L208-L280)：找到输入路径、GPU 调用、批内下标恢复 ID 和写出的位置。
3. 看 [mining_gpu.rs:131–177](../src/mining_gpu.rs#L131-L177) 如何创建设备及 pipeline，再在 [正式内核的内嵌代码](gpu-learning-02-mining-wgsl.md) 中找到六个绑定及旁边的解释。
4. 在同一份内嵌代码中找到 `main`，跟踪一个 invocation；不懂业务计算时对照 [core.rs:42–112](../src/core.rs#L42-L112) 的 CPU 实现。
5. 继续读内嵌代码中的模型与候选压缩，再看 [mining_gpu.rs:363–489](../src/mining_gpu.rs#L363-L489) 的两段回读和精筛。
6. 最后看 [output.rs:8–54](../src/output.rs#L8-L54)、[main.rs:320–432](../src/main.rs#L320-L432) 和 [gpu-extreme.md](gpu-extreme.md)，区分输出正确性、样本精度与性能数据的证据范围。
