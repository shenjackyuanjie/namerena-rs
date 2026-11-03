# GPU 入门 03：名字生成片段与两个 RC4 实验着色器

本文根据当前工作树逐项阅读 [generated-key.wgsl](../generated-key.wgsl)、[main.wgsl](../main.wgsl) 和 [rc4-shared.wgsl](../rc4-shared.wgsl)。下面的行号按当前文件从 1 开始；文件版本、行号和行为都可能变化，阅读时以相对链接打开的实际代码为准。空行、独立注释和单独的花括号不逐行列出；花括号只表示结束相应的分支、循环或函数。其余有效代码均按行或相邻同义行解释。这里的“生产”指当前完整测号路径，不等于已经证明所有设备和输入都正确。

## 先认识环境

CPU 是运行 Rust 宿主程序的处理器；GPU 上运行的 WGSL 程序称为着色器。一次 compute 派发包含若干工作组（workgroup），每组包含多个 invocation，类似多名各自处理一个候选的工人。`global_invocation_id.x` 是这次一维派发内的全局编号；`local_invocation_index` 是组内编号 lane。一个组大小为 32 时，前两组的全局编号分别为 0～31、32～63，lane 都是 0～31。末组通常补齐，所以必须检查真实候选数。

`u32` 是 32 位无符号整数；Rust 的 `u64` 是 64 位无符号整数，本着色器没有直接使用 `u64`，而用两个 `u32` 表示 ID 的高低位。`u` 后缀表示无符号常量。一个字节是 8 位，取值 0～255。`array<T, N>` 是 N 项数组，下标从 0 开始；`vec4<u32>` 是含四个 u32 的向量，可用下标取分量。`struct` 把多个字段组成记录，`let` 定义不可重新赋值的局部量，`var` 定义可写变量。`fn` 定义函数，`return` 返回，`if/else` 分支，`for` 逐次循环，`switch/case/default` 选分支，`+=` 与 `|=` 分别是加后赋值和按位或后赋值。

`>>` 右移、`<<` 左移；`&` 是按位与、`|` 是按位或。`& 255` 保留最低 8 位，等价于对非负数取模 256；`& 63`、`& 0x3fff` 分别保留 6、14 位。`/` 对这里的无符号整数取整除，`%` 取余。比如 `13 / 4 = 3`、`13 % 4 = 1`。`0x` 是十六进制；`0x100 = 256`。码点是 Unicode 分配给字符的数值，如“一”是 U+4E00；大端表示高位字节先于低位字节。32 位无符号加法按 2³² 回绕。`select(a, b, condition)` 条件为假取 a，真取 b。

字节打包例子：四字节依次为 `11 22 33 44`（十六进制），按低字节先放得到 `0x44332211`。取第 2 个字节（从 0 数）用 `(word >> (2 * 8)) & 255`，结果是 `0x33`。低字节先放描述的是一个数值内部的排列，与后面 ID 输入使用的大端顺序是两件事。UTF-8 是将文本表示为字节的编码，中文通常占三个字节；这里计算的是字节数而不是字符数。

RC4 的状态盒 S-box 在这里是 256 个 0～255 的排列。KSA 是密钥调度：每步根据当前状态和密钥更新游标，再交换状态的两个槽；交换不丢失任何值。举例，初始盒为 0,1,2,…，密钥第一个字节为 0，初始 j=0：i=0 时 j=(0+0+0)%256=0，交换第 0 与第 0 槽；第二步若密钥字节为 65，i=1，则 j=(0+1+65)%256=66，交换槽 1 与槽 66。实际项目先计算队伍状态，再从该状态对名字执行两轮 KSA；这不是通用加密教程。

## 三者与 Rust 的实际关系

- [main.wgsl:2–42](../main.wgsl#L2-L42) 和 [rc4-shared.wgsl:3–45](../rc4-shared.wgsl#L3-L45) 是两个完整、互斥选择的 RC4 实验内核，入口都是 `rc4_name`；只返回两轮名字 KSA 后的完整状态，不做完整测号。`Kernel::Private/Shared` 分别通过 `include_str!` 选择它们，工作组大小 64/32，见 [gpu.rs:6–24](../src/gpu.rs#L6-L24)；Rust 创建对应模块和入口见 [gpu.rs:105–117](../src/gpu.rs#L105-L117)。CLI 遇到 `--bench` 才转入实验入口（[main.rs:613–617](../src/main.rs#L613-L617)），实验参数选择 `--kernel private|shared`，默认 shared（[rc4_bench.rs:20–28](../src/rc4_bench.rs#L20-L28)、[rc4_bench.rs:125–141](../src/rc4_bench.rs#L125-L141)）。普通完整测号不是从这两个文件拼装而成。
- 完整测号使用 [mining.wgsl](../mining.wgsl)，包含后续属性、技能、shadow、模型与筛选，入口为 `mine`（[mining_gpu.rs:165–176](../src/mining_gpu.rs#L165-L176)）。其默认配置和可选布局见 [mining_gpu.rs:21–35](../src/mining_gpu.rs#L21-L35)。两个实验内核与它有相似的 KSA 和转置思路，但不是同一着色器，也不要把实验吞吐当成完整测号吞吐。
- [generated-key.wgsl](../generated-key.wgsl) **不是独立内核**：没有绑定或入口，引用正式内核定义的类型、缓冲和函数。Rust 先加载 mining.wgsl、替换工作组、S-box、模型常量；生成模式将 `GENERATED` 改为 true，以 `KEY_BEGIN` 和 `KEY_END` 分割，整个替换区间插入该片段（[mining_gpu.rs:37–59](../src/mining_gpu.rs#L37-L59)，原区间 [mining.wgsl:102–116](../mining.wgsl#L102-L116)）。这不是 WGSL 自带 import；缺少标记时宿主返回错误。上传模式保留原空 `prepare_generated` 和读取打包名字的 `key_byte`。生成模式的 `mine` 选择 `names[0]` 并先准备编码，随后 KSA 使用替换后的函数（[mining.wgsl:196–205](../mining.wgsl#L196-L205)）。
- CLI 的 `auto` 在固定宽 Base16384、标准 Base64、URL Base64 上使用生成模式；`uploaded` 强制上传；`generated` 遇到 Number 则报错，`auto` 遇到 Number 回退上传（[main.rs:93–95](../src/main.rs#L93-L95)、[main.rs:440–458](../src/main.rs#L440-L458)、[gpu_input.rs:17–26](../src/gpu_input.rs#L17-L26)）。CPU 参考生成器将 ID 转为 8 字节大端编码，替换模板中的全部 `[]`（[input.rs:34–46](../src/input.rs#L34-L46)）。

## 生成片段的输入约定

Rust 为模板中每个最终名字字节制作一个 u32 token：固定 UTF-8 字节取 0～255；每个占位符的第 n 个生成字节记作 `0x100 | n`。例如模板 `A[]!` 的 Base64 映射是 `[65, 256, 257, …, 267, 33]`；多个 `[]` 都引用同一候选的编码。总名字最多 256 字节，Base16384 占 18 字节、两种 Base64 均占 12 字节（[gpu_input.rs:14–38](../src/gpu_input.rs#L14-L38)）。

Rust 在 CPU 上从队伍 S-box 接着预跑第一轮的公共名字前缀（含开头零），保存执行步数、j 和状态（[gpu_input.rs:39–55](../src/gpu_input.rs#L39-L55)）。生成模式将状态放入 `team_state`，将 `[prefix_steps, name_len]` 放入 `names[0]`，将 `[prefix_j, encoding]` 放入 `names[1]`，将每个 token 各占一个 u32 上传到 `name_bytes`（[mining_gpu.rs:190–196](../src/mining_gpu.rs#L190-L196)、[mining_gpu.rs:265–271](../src/mining_gpu.rs#L265-L271)）。因此此模式的 `info.offset` **不是字节地址**。第一轮从快照步数、j 和 k 继续，第二轮重新从零开始；shadow 也复用同一个公共前缀（[mining.wgsl:118–135](../mining.wgsl#L118-L135)、[mining.wgsl:274–280](../mining.wgsl#L274-L280)）。

每批起始 ID 分别写入 `pad0` 低 32 位与 `pad1` 高 32 位（[mining_gpu.rs:363–377](../src/mining_gpu.rs#L363-L377)）；`run_range` 检查末 ID 溢出（[mining_gpu.rs:323–340](../src/mining_gpu.rs#L323-L340)）。派发按工作组大小向上取整（[mining_gpu.rs:390–393](../src/mining_gpu.rs#L390-L393)）。

### generated-key.wgsl 逐段说明

以下行号均链接到该文件；每个编号范围包含该段实际语句和必要的语法边界。

| 行 | 做什么、为什么 |
| --- | --- |
| [3](../generated-key.wgsl#L3) | `var<private> generated_chunks: array<u32, 6>`：每个 invocation 私有的六个 32 位槽；不在不同候选间共享。Base16384 每槽存三个 UTF-8 字节，Base64 只用前三槽、每槽四字节。 |
| [5–8](../generated-key.wgsl#L5-L8) | `prepare_generated(index)` 计算本候选 ID：低字加批内 index；若结果小于原低字说明发生回绕，`select` 给高字加 1，否则加 0；编码编号从 `names[1].length` 取。例：低字 `0xffffffff` 加 1 得 0，高字进 1。 |
| [9–13](../generated-key.wgsl#L9-L13) | 编码 0 进入 Base16384，循环六组，先将码点 `point` 置 0，再按组号分支。 |
| [14–19](../generated-key.wgsl#L14-L19) | 把 64 位大端输入（高位先于低位，位号从最低位 0 起）按高位在前分组：组 0 是 high 的第 31～18 位；组 1 是 high 的 17～4 位；组 2 拼 high 的低 4 位与 low 的高 10 位；组 3 是 low 的 21～8 位。每组取 14 位并加 `0x4e00`。组 4 将 low 最低 8 位左移 6 位，再加 `0x4e00`；组 5 用固定码点 `0x3d01`（㴁）；普通 14 位组的最大码点是 U+8DFF。`0x3fff` 是 14 个 1，掩掉其他位。 |
| [22–26](../generated-key.wgsl#L22-L26) | 把码点编码为三字节 UTF-8：分别是 `0xe0 \| point>>12`、`0x80 \| ((point>>6)&63)`、`0x80 \| (point&63)`；按低字节在前放进一个 u32，例如 U+4E00 的三个字节是 `e4 b8 80`，打包结果为 `0x0080b8e4`。循环结束后返回，不再执行 Base64。例：ID 0 的结果为 `一一一一一㴁`，每字三字节（另见 [input.rs:82–87](../src/input.rs#L82-L87)）。 |
| [29–34](../generated-key.wgsl#L29-L34) | 非零编码走 Base64：外层三组，每组四字符，`packed` 清零；`character=group*4+offset` 是全局字符序号；先令 ASCII 为 61，即 `=`，仅前 11 个字符另算。 |
| [35–44](../generated-key.wgsl#L35-L44) | 每个编码值占 6 位：字符 0～4 从 high 各取 6 位；字符 5 拼 high 最低 2 位与 low 最高 4 位；字符 6～9 从 low 取 6 位；字符 10 使用 low 最低 4 位左移 2 位补零。具体右移位数分别为 `26-character*6`、`58-character*6`；跨界分支用 `(high<<4) \| (low>>28)`。每次 `&63` 留低 6 位；序号 11 的字符保持 `=`。 |
| [45–55](../generated-key.wgsl#L45-L55) | 0～25 映射 ASCII 65～90，即 A～Z；26～51 映射 97～122，即 a～z；52～61 映射 48～57，即 0～9。62 对应标准 `+` 或编码 2 的 `-`；63 对应标准 `/` 或编码 2 的 `_`。宿主只提供编号 0、1、2，不应把其他非零编号当作已支持的独立编码。 |
| [57–61](../generated-key.wgsl#L57-L61) | 将 ASCII 左移 `offset*8` 后或入对应字节位置；内层结束保存一个四字节 u32；外层结束函数返回。例：ID 1 的标准/URL Base64 都是 `AAAAAAAAAAE=`（[input.rs:82–87](../src/input.rs#L82-L87)）。 |
| [64–67](../generated-key.wgsl#L64-L67) | `key_byte(info,k)` 返回一字节数值：k=0 返回密钥的前导 0；否则 `byte_index=k-1`。当它小于名字长度时读取名字部分。 |
| [68–74](../generated-key.wgsl#L68-L74) | 每个 token 一整个 u32：小于 256 就是固定字节；否则 `token & 255` 是生成字节偏移。Base16384 用 `offset/3` 选槽、`offset%3` 选槽内字节；Base64 用 4 代替 3。右移 `槽内序号*8` 再 `&255` 提取。例如 token 257 表示生成结果第 1 字节。 |
| [76–77](../generated-key.wgsl#L76-L77) | 名字之后调用 `shadow_byte(byte_index-info.length)` 返回 `?shadow` 的对应字节；正常名字 KSA 的游标不会走到这里，shadow 路径才扩展长度 7（[mining.wgsl:89–100](../mining.wgsl#L89-L100)、[mining.wgsl:119–132](../mining.wgsl#L119-L132)）。 |

## 两个完整实验内核：绑定和入口

实验的 `@group(0) @binding(n)` 是宿主提供缓冲的编号，`uniform` 用于只读参数，`storage, read` 只读较大数据，`storage, read_write` 可写输出。Rust 按相同顺序创建绑定并上传（[gpu.rs:121–169](../src/gpu.rs#L121-L169)、[gpu.rs:211–238](../src/gpu.rs#L211-L238)）。下面将两个文件中相邻、语义相同的行并列；不同行单独说明。

| private 行 | shared 行 | 解释 |
| --- | --- | --- |
| [2](../main.wgsl#L2) | [3](../rc4-shared.wgsl#L3) | `team_state` 是 64 个四分量向量，共 256 个 u32；CPU 算好的队伍 KSA 状态每字节扩成 u32。 |
| [3–4](../main.wgsl#L3-L4) | [4–5](../rc4-shared.wgsl#L4-L5) | `NameInfo` 保存名字在连续输入中的字节偏移 `offset` 和字节长度 `length`；`names` 是候选元数据数组。与生成模式复用的字段含义不同。 |
| [6–9](../main.wgsl#L6-L9) | [6–9](../rc4-shared.wgsl#L6-L9) | `name_bytes` 是四字节打包的 u32 数组；`output_states` 是可写结果；`Params` 有候选数和三个填充字，`params` 为 uniform。填充使宿主参数块为 16 字节；不是额外名字信息。 |
| — | [10](../rc4-shared.wgsl#L10) | 工作组共享 `values[8192]`：256 槽 × 32 lane × 4 字节 = 32768 字节（32 KiB）。每个候选的 256 槽都在该候选独占的列里。 |
| [11–12](../main.wgsl#L11-L12) | [12–13](../rc4-shared.wgsl#L12-L13) | 声明 compute 入口，分别每组 64 或 32 个 invocation。`position` 是三维全局 ID；shared 额外接收组内 lane。实验只用 x 维。 |
| [13–15](../main.wgsl#L13-L15) | [14–16](../rc4-shared.wgsl#L14-L16) | 取全局 x 为候选编号，超出真实 `count` 的补齐 invocation 立即返回；否则读取该候选元数据。 |
| [16–19](../main.wgsl#L16-L19) | [17–19](../rc4-shared.wgsl#L17-L19) | 遍历 0～255 槽，把队伍状态 `team_state[i/4][i%4]` 复制到本候选。private 在函数内建独立的 256 项 `values`；shared 写 `values[i*32+lane]`。比如 slot 5、lane 2 的地址是 162。 |

shared 的列属于不同 invocation，任何一次随机交换都只访问自己的 lane；其他候选不会读写该列。因此这里无需跨 lane 同步，也没有 `barrier`（[rc4-shared.wgsl:17–20](../rc4-shared.wgsl#L17-L20)）。转置布局意在使同槽不同 lane 分布到不同共享内存 bank；bank、warp 的实际组织及速度依赖硬件，不能保证所有设备都加速。宿主要求至少 32 KiB 工作组存储，否则提示使用 private（[gpu.rs:76–83](../src/gpu.rs#L76-L83)）。

### 两轮名字 KSA：每条语句

密钥的概念字节序列是 `[0] + UTF-8(name)`，循环使用，长度为 `info.length + 1`。代码没有实际创建这个数组：k=0 时跳过名字加法，相当于加 0；k>0 时读第 k-1 个名字字节。即使空名字，密钥也有一个零，不会出现长度为零的取模。

| private 行 | shared 行 | 解释 |
| --- | --- | --- |
| [21–24](../main.wgsl#L21-L24) | [21–24](../rc4-shared.wgsl#L21-L24) | 外层两轮；每轮 j（交换目标游标）和 k（密钥游标）都归零；内层 i 从 0 到 255，每轮 256 次。状态本身不在轮间重置。 |
| [25–28](../main.wgsl#L25-L28) | [25–28](../rc4-shared.wgsl#L25-L28) | k 非零时，`address=info.offset+k-1` 是输入字节绝对位置；`address/4` 选择 u32，`address%4` 选择该字内位置，右移并掩码得到字节，加到 j。k 为零则隐式加零。 |
| [29](../main.wgsl#L29) | [29–32](../rc4-shared.wgsl#L29-L32) | j 加当前槽值后 `&255`，得到 0～255 的交换槽。shared 先算 `address_i=i*32+lane` 并将当前值存到 `current`，再算 `address_j=j*32+lane`。 |
| [30–32](../main.wgsl#L30-L32) | [33–34](../rc4-shared.wgsl#L33-L34) | 保存 i 位置旧值，先将 j 位置值放入 i，再将旧值放入 j，完成交换；i=j 时也安全。shared 的两个地址都使用同一 lane。 |
| [33–35](../main.wgsl#L33-L35) | [35–37](../rc4-shared.wgsl#L35-L37) | k 等于名字长度时归零，否则加 1：因 k 从 0 到 length，恰好循环前导零和全部名字字节。结束内层、外层循环。 |
| [37–42](../main.wgsl#L37-L42) | [38–45](../rc4-shared.wgsl#L38-L45) | 64 次循环，每次 base=i*4，读取连续四个状态槽，分别左移 0、8、16、24 位后按位或，写入 `output_states[index*64+i]`。shared 对每槽再乘 32 加 lane。每候选 64 个 u32，即 256 字节；保留完整排列并按候选顺序排列。 |

举例：名字 `A` 的 UTF-8 为一个字节 65，密钥游标依次循环 k=0（字节 0）、k=1（字节 65）、k=0……；若名字为 `中`，它是三个 UTF-8 字节，游标依次为 0、1、2、3、0，不是只走一个汉字。

宿主把每个名字的原始 UTF-8 字节连续拼接，记录偏移与长度，补零到四字节对齐；派发后把输出复制到 staging 缓冲并映射回 CPU（[gpu.rs:211–256](../src/gpu.rs#L211-L256)、[gpu.rs:261–279](../src/gpu.rs#L261-L279)）。补零只是缓冲对齐，不是名字的一部分；尾部 invocation 的边界检查防止读到它们当候选。

## 如何核实和继续学习

- 对照 [gpu_input.rs:154–210](../src/gpu_input.rs#L154-L210) 的固定编码、跨 32 位进位和极值测试，以及 [gpu.rs:320–340](../src/gpu.rs#L320-L340) 中默认忽略、需要 GPU 的 private/shared 对照测试。测试代码存在不等于本文写作时实际运行通过。
- 可以按 [RC4 实验记录:78–86](rc4-experiment.md#L78-L86) 自行选择运行测试；硬件测试需要可用 GPU。完整路径的浮点粗筛与 CPU 精算边界、平台误差风险见 [README:53–60](../README.md#L53-L60)。本文是静态代码核对，不是 GPU 实测、性能承诺或跨设备正确性证明。
