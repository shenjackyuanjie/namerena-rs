# 从零读懂 mining.wgsl：逐行计算与筛选

本文按当前 [mining.wgsl](../mining.wgsl) 的源码顺序解释全部非空有效代码；连续且作用相同的行合并讲解，花括号表示所在结构、函数、循环或分支结束。行号以当前文件为准，版本变化或插入行后可能失效，请以链接中的源码为准。文中引用的行号均为该文件的行号，其他文件单独链接。

## 先认识执行环境

GPU 的计算着色器是由 CPU（这里称 host）交给 GPU 执行的小程序，使用 WGSL 语言；一次执行称 invocation。多个 invocation 属于一个 workgroup（工作组），组内可使用共享的 workgroup 内存。本程序每个 invocation 处理一个候选名字；它的组内编号称 lane。同一组各 lane 的状态在共享数组中占不同列，所以本文件没有跨 lane 的 barrier（同步屏障），尾批和筛选失败可直接返回。

基础记号：`u32` 是 32 位无符号整数，`i32` 是有符号整数，`f32` 是 32 位浮点数；数字后缀 `u`/`i` 表示类型。`array<T, N>` 是 N 项数组，`vec2`/`vec4` 分别是 2/4 分量向量；下标从 0 开始。`let` 不可重新赋值，`var` 可修改；函数局部未显式初始化的变量按类型零初始化。`&` 在表达式中取地址，在整数之间是按位与；`*` 可解引用指针，也可表示乘法。`|` 是按位或，`~` 按位取反，`<<`/`>>` 是移位，`%` 取余；例如 `x & 255u` 保留低 8 位，相当于非负整数对 256 取余。`select(a, b, 条件)` 在条件为假时取 a、为真时取 b。

整体流程：恢复队伍/前缀的 256 项状态 → 加入名字做两轮 KSA → 得到属性并初筛 → 从状态产生随机字节安排技能 → 必要时计算影子 → 两个模型评分 → 压缩回传。KSA 是密钥调度，PRGA 是状态更新并产生字节的过程；这里的 Sbox 是 0..255 的排列。这个项目借用类似 RC4 的状态操作来确定性地计算候选，不是随机试验保证。

## 资源、布局和编译常量

- **[1–5 行](../mining.wgsl#L1-L5)**，`@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>;`：组 0 的第 0 号绑定是只读 uniform（统一参数缓冲），64 个四分量共 256 个 u32，每个数保存一个状态字节。上传名字模式使用原始队伍 KSA 状态；生成模式使用已做完名字公共前缀的第一轮 KSA 快照。host 的选择见 [mining_gpu.rs:190–196](../src/mining_gpu.rs#L190-L196)。
- **[7–13 行](../mining.wgsl#L7-L13)**，`struct NameInfo { offset: u32, length: u32 }` 描述一项名字元数据；`names` 是绑定 1 的只读 storage 数组。上传模式中 `offset` 是拼接字节流的字节偏移，`length` 是 UTF-8 字节数，不是字符数；绑定 2 的 `name_bytes: array<u32>` 每个字打包四个字节。例如 `A中` 是 4 字节而非 2 字节。生成模式另有复用含义，见后文。
- **[15–24 行](../mining.wgsl#L15-L24)**，`struct Record` 定义一个返回槽：`index` 为本批候选下标，偏移 0；`props[8]` 是八个属性，偏移 4；`skill_ids[16]` 和 `skill_freqs[16]` 是前 16 个技能编号及对应频率，偏移分别 36、100；`shadow_numerator: i32` 是影子特征的带符号分子，偏移 164；`xp`、`xd` 为两个近似分数，偏移 168、172。44 个 32 位字共 176 字节；数组都是标量，避免 `vec3` 可能引入的对齐填充。
- **[25–32 行](../mining.wgsl#L25-L32)**，`struct Output` 的四个 `atomic<u32>` 是支持并发安全操作的原子计数器：`count` 记录回传槽数，`prop_pass` 记录进入技能阶段数，`scored` 记录完成评分数，`shadow_count` 记录计算影子的数量；运行时长度的 `records` 从偏移 16 开始。绑定 3 是可读写 storage。host 每批清空 16 字节头，保证至少每候选一个记录槽；见 [mining_gpu.rs:212–216](../src/mining_gpu.rs#L212-L216)、[363–393](../src/mining_gpu.rs#L363-L393)。
- **[34–45 行](../mining.wgsl#L34-L45)**，`struct Params` 和绑定 4 的 uniform 共 8 个 u32、32 字节：`count` 是本批候选数；`prop_threshold` 是属性阈值；`xp_threshold`、`xd_threshold` 是评分阈值；`capture_all` 非零时保留所有候选；`pad0`/`pad1` 在生成模式分别存起始 64 位 ID 的低/高 32 位；`pad2` 为保留填充。host 将属性阈值钳到 10000，避免后面乘 3 溢出，具体赋值见 [mining_gpu.rs:363–377](../src/mining_gpu.rs#L363-L377)。
- **[46–48 行](../mining.wgsl#L46-L48)**，`weights: array<vec2<f32>>` 是绑定 5 的只读模型权重。每项两分量分别对应 XP 与 XD（CPU 侧模型称 QD），`weights[0]` 是两模型常数，其后 44 个线性项和 990 个二次项，共 1035 对、8280 字节，数组步长 8。数据由 [core.rs:226–231](../src/core.rs#L226-L231) 生成。
- **[50–56 行](../mining.wgsl#L50-L56)**，`LANES=32` 为工作组大小，`WORDS=8192` 为共享 u32 数，`PACKED=false` 选宽字状态，`GENERATED=false` 选上传名字，`COMPACT_MODEL=false` 选稠密遍历；`var<workgroup> values` 分配组内共享状态。**这些是源码模板值，不等于实际默认运行值**：[mining_gpu.rs:21–60](../src/mining_gpu.rs#L21-L60) 替换常量；当前 host 默认 packed、32 lane、compact，允许 packed 32/64/128，wide 只允许 32。wide 每候选 256 个 u32 即 1024 字节；packed 每候选 64 个 u32 即 256 字节。布局按槽转置：同槽的各 lane 相邻，每 lane 独占自己的列。

## Sbox 读写与恢复

- **[58–63 行](../mining.wgsl#L58-L63)**，`fn read_sbox(lane, slot) -> u32` 读取某 lane 的一个 0..255 槽。`if (PACKED)` 时，字索引 `(slot / 4) * LANES + lane`；`(slot % 4) * 8` 是字节位移，右移后 `& 255` 截出一字节。否则 `values[slot * LANES + lane]` 直接读宽字。例如 slot=5、lane=2：读第 1 个打包字的第 2 字节，地址为 `LANES+2`、右移 8 位。
- **[65–74 行](../mining.wgsl#L65-L74)**，`write_sbox(lane, slot, value)` 写槽。packed 分支先计算 `word` 和 `shift`；`values[word] = (values[word] & ~(255u << shift)) | (value << shift)` 先清目标 8 位，再并入值，保留同字其他槽；调用者传入的值来自字节状态。非 packed 的 `else` 直接写 `values[slot * LANES + lane]`。同一个字里的两槽交换必须在第二次写时读第一次更新后的字；这里每次赋值都重新读取，且列不跨 lane 共享。
- **[76–87 行](../mining.wgsl#L76-L87)**，`restore_team(lane)` 恢复输入状态。packed 分支循环 `i=0..63`，取 `team_state[i]` 的 `x/y/z/w`，依次左移 0/8/16/24 位并按位或，写入该 lane 的第 i 个字。wide 分支循环 0..255，以 `i / 4` 选向量、`i % 4` 选分量，逐槽复制。此处只恢复 lane 自己的列。

## 名字字节与 KSA

- **[89–100 行](../mining.wgsl#L89-L100)**，`shadow_byte(index)` 用 `switch` 按索引返回后缀 `?shadow` 的 ASCII 字节：0→63 `?`，1→115 `s`，2→104 `h`，3→97 `a`，4→100 `d`，5→111 `o`；`default` 返回 119 `w`，合法调用的剩余索引是 6。后缀直接接到原始名字字节，不解析名字内的 `@`，也不修改原队伍。
- **[102–116 行](../mining.wgsl#L102-L116)**，`KEY_BEGIN`/`KEY_END` 是 host 替换片段的标记，不是 WGSL 控制语句。模板里的 `prepare_generated(index) {}` 为空函数；上传模式的 `key_byte(info, k)` 在 `k == 0` 返回前导零，否则 `byte_index = k - 1`。如果该索引小于名字长度，`address = info.offset + byte_index`，读取对应打包字、右移 `(address % 4) * 8` 并屏蔽 255；否则通过 `shadow_byte(byte_index - info.length)` 读取后缀。只有调用者选了影子长度才会走到后缀。例：名字 `A` 的普通键是字节 `[0, 65]`，影子键是 `[0, 65, 63, 115, 104, 97, 100, 111, 119]`。

  **生成模式不同：**[mining_gpu.rs:53–57](../src/mining_gpu.rs#L53-L57) 把两个标记之间整段替换为 [generated-key.wgsl](../generated-key.wgsl)，并设置 `GENERATED=true`。因此实际生成模式的 `prepare_generated` 并不为空，`key_byte` 也不是上面的上传实现。独立片段根据起始 ID 加本批下标生成 Base16384 或标准/URL Base64 字节，再根据模板映射取固定字节或生成字节。host 上传的 `names[0]` 为 `[prefix_steps, name_len]`，`names[1]` 为 `[prefix_j, encoding]`，绑定 2 是每 u32 一个 token 的模板映射，不是打包名字；见 [mining_gpu.rs:265–271](../src/mining_gpu.rs#L265-L271)、[gpu_input.rs:14–55](../src/gpu_input.rs#L14-L55)。例如模板 `A[]` 的固定 `A` 是 token 65，生成部分第 0 字节是 token `0x100`；数字 Number 编码不走该生成路径。此处只概述独立文件，不把它误当作本文件的逐行内容。
- **[118–135 行](../mining.wgsl#L118-L135)**，`name_ksa(lane, info, with_shadow)` 对已恢复的状态做两轮名字 KSA。`key_length = info.length + select(0u, 7u, with_shadow)` 是不含前导零的名字或名字加 7 字节后缀长度；host 限制原始名字最多 256 字节，加后缀内部可到 263。每轮 `round=0,1` 都把状态下标 `s`、键下标 `k` 和起始状态位置 `first` 置零，但第二轮继续使用第一轮修改后的 Sbox。
  - **125 行**：仅生成模式的第 0 轮设置 `first = info.offset; k = first; s = names[1u].offset`，从 host 已算好的公共前缀之后继续；第二轮从零开始。前缀步数包含前导零，快照保存的是已经经过那些步骤的状态和 `s`，见 [gpu_input.rs:39–48](../src/gpu_input.rs#L39-L48)。影子也有相同的原名字前缀，故可复用此快照。
  - **126–133 行**：`i` 从 `first` 到 255；读槽 i 为 `current`，`s = (s + key_byte(info, k) + current) & 255` 按 256 回绕；读槽 s 为 `other`，再互换槽 i 和 s。若 `k == key_length` 则回到 0，否则加 1，即循环使用前导零加名字（可含后缀）。当 i 和 s 相同，互换仍成立。每轮最多执行 256 步，并不意味着长度超过 255 的键的每个字节都会在该轮被读到。

## 提取属性

- **[137–148 行](../mining.wgsl#L137-L148)**，`collect_base(lane, base: ptr<function, array<u32, 128>>)` 将结果写到调用者的局部数组；`(*base)[count]` 是解引用后写数组。`count` 从 0 开始，遍历状态 256 槽，计算 `mapped = (S[i] * 181 + 160) & 255`。若 `89 <= mapped < 217`，存入 `mapped & 63` 然后计数加一。181 与 256 互质，所以对 0..255 的排列该仿射变换仍是排列；半开区间 [89,217) 恰有 128 个数，因而恰好填满数组。按原 Sbox 顺序筛选，不排序、不改变 Sbox。例如 89 入选，存 `89 & 63 = 25`；217 不入选。
- **[150–152 行](../mining.wgsl#L150-L152)**，`median3(x, y, z)` 返回 `max(min(x,y), min(max(x,y),z))`，即三数的中间值，例如 2、8、5 返回 5，相同值也适用。
- **[154–170 行](../mining.wgsl#L154-L170)**，`properties(base)` 返回 8 个属性。先将 `base[0..10]` 复制到局部 `hp`，保留原 base 给后面技能用。对 `i=1..9` 做插入排序：保存当前 `value`、令 `j=i`；`loop` 中如果 `j==0` 或前项 `<= value` 则 `break`，否则把前项右移、`j -= 1`；退出后把值放到 `hp[j]`。结果为升序。
- **[171–178 行](../mining.wgsl#L171-L178)**，`props[0] = 154 + hp[3] + hp[4] + hp[5] + hp[6]`，取排序后中间四项加固定 154 得 HP。对 `i=1..7`，`start = 10 + (i-1)*3`，在原 base 的三个数中取中位数加 36，存入其余七项；最后返回 `props`。例如 i=2 取 base[13..15]（下标 13、14、15）。

## 伪随机字节和数值检查

- **[180–189 行](../mining.wgsl#L180-L189)**，`prga_byte(lane, a, b)` 接收局部状态变量指针并返回一个字节：`*a = (*a+1)&255`；取 S[*a] 为 `current`；`*b = (*b+current)&255`，取 S[*b] 为 `other`；互换两槽；再读取 `S[(current+other)&255]` 返回。交换前后两数的和相同，故不必重新读取两槽计算下标。指针使 a/b 跨调用持续更新，Sbox 也会被原地修改。
- **[191–194 行](../mining.wgsl#L191-L194)**，`finite_score(value)` 用 `bitcast<u32>` 原样查看 f32 的位，而不是数值转换；`& 0x7f800000` 取 IEEE 754 指数字段，若不全为 1 就返回真，即有限值；全 1 包含正负无穷与 NaN，返回假。这样异常值不会因为 NaN 比较行为而悄悄丢失。

## 计算入口、初筛和技能

- **[196–202 行](../mining.wgsl#L196-L202)**，`@compute @workgroup_size(LANES)` 声明计算入口和组大小；`position: vec3<u32>` 是 GPU 提供的全局三维 invocation ID，`lane` 是组内线性编号。host 只沿 x 方向派发，`index = position.x` 是本批下标。向上取整派发可能多出尾部 invocation，`index >= params.count` 立即返回。例如 count=33、LANES=32 派发 2 组，共 64 次，后 31 次返回。上传模式用 `names[index]`，生成模式用共同的 `names[0]`；生成模式还调用替换后的 `prepare_generated(index)`。
- **[204–208 行](../mining.wgsl#L204-L208)**，先 `restore_team`，再对原名字调用 `name_ksa(..., false)`；创建局部 `base[128]` 并提取；`props = properties(&base)` 获得八项属性。
- **[210–218 行](../mining.wgsl#L210-L218)**，`prop_numerator` 从 HP 开始，加上其余七属性各自的 3 倍，即属性总值 `HP/3 + 其余七项之和` 的精确三倍，不先做整数除法。若非 `capture_all` 且 `prop_numerator <= 3 * prop_threshold` 就返回，故通过条件是严格大于；否则 `atomicAdd(&output.prop_pass, 1)` 安全累加。例：分子 301、阈值 100 时 301 > 300，通过；300 不通过。capture_all 下每个有效候选均计入。
- **[220–240 行](../mining.wgsl#L220-L240)**，局部 `skill_ids[40]` 初始化为 0..39，`skill_freqs[16]` 初始为零；`a`、`b` 是 PRGA 下标，`s` 是技能洗牌下标，三者初始为零，两个轮次连续保留。每轮遍历 i=0..39：读 `high` 和 `low` 两个 PRGA 字节，`rnd = ((high << 8) | low) % 40` 组成 16 位数取余；`s = ((s + rnd + skill_ids[i]) & 255) % 40` 先按 u8 回绕、再对 40 取余；将 `skill_ids[i]` 与 `skill_ids[s]` 交换。共 2×40×2=160 字节。先截低 8 位不能省略：例如总和 256 时先回绕为 0，再 %40 为 0；直接 256%40 为 16。
- **[242–255 行](../mining.wgsl#L242-L255)**，`last = -1` 表示尚未找到低频槽。遍历前 16 个技能槽，`start = 64+i*4`，`p` 是 base 连续四项的最小值；若 `p>10` 且 `skill_ids[i]<35`，频率为 `p-10`，且若 `p<35` 更新 `last=i`。否则频率明确为 0，35..39 是空技能。循环结束若 `last>=0` 则将最后一个满足条件的槽频率翻倍；没有则跳过。例如 p=20、合法技能，频率先为 10，若最后被选中则变 20；p=35 不成为 last。
- **[256–262 行](../mining.wgsl#L256-L262)**，对于槽 14：仅当它不是已翻倍的 last 且其频率非零，加上 `min(base[60], base[61], skill_freqs[14])`。槽 15 同理用 base[62]、base[63] 和自身频率。各槽判断独立；last 为 14 或 15 时只跳过该槽自己的尾槽加成，另一槽仍可加；频率为零时不加。

## 特征与影子

- **[264–272 行](../mining.wgsl#L264-L272)**，`features: array<f32, 44>` 的局部项初始全零；先把 8 个属性转为浮点放在下标 0..7；再遍历 16 个槽，只把非零频率放到 `features[skill_ids[i] + 8]`。由于非零频率的技能 ID 小于 35，技能特征占 8..42；位置由技能 ID 而不是技能槽号决定，ID 来自排列所以不会重复。下标 43 留给影子。
- **[274–289 行](../mining.wgsl#L274-L289)**，`shadow_numerator` 初始为有符号 0；`features[32] > 0` 等价于技能 ID 24 有非零频率，才进入影子分支并将 `shadow_count` 原子加一。重新恢复队伍/前缀状态，以 `with_shadow=true` 用原名字加 `?shadow` 做 KSA，复用 Sbox 和 base，重新提取 `shadow_props`；原 `props`、技能编号及频率仍保留。`raw_sum` 累加影子属性 1..7 各减 36 后的值；`shadow_numerator = shadow_props[0] + 3*raw_sum - 9*(shadow_props[7]-36) - 630`，结果用 i32 可为负；`features[43] = shadow_numerator / 3.0 * features[32] / 100.0`。例：分子 90、技能频率 20 时影子特征为 6。未触发时分子及影子特征都为零。
- **[290–292 行](../mining.wgsl#L290-L292)**，若 `features[42] > 0`，即 ID 34 的技能有非零频率，给该特征额外加 20；否则保持零。这发生在影子计算之后、评分之前。

## 两个评分模型

- **[294–308 行](../mining.wgsl#L294-L308)**，`score = weights[0]` 同时初始化 XP 与 XD 分量；线性项有 44 个，二次项按原始索引 `i=0..43, j=i..43` 排列，共 `44×45/2=990` 个，因此模型总计 1034 项加一个常数。`COMPACT_MODEL` 为真时，`active_indices[32]` 保存非零特征的原下标，`active_count` 初始 0；顺序扫描 44 个 `original`，把非零 `value` 的下标存入 active_indices，再把值紧凑写到 `features[active_count]` 并递增。写入位置不会超过当前 original，因而不覆盖未来尚未读取的项。最多 8 个属性、16 个非零技能、1 个影子，25 项装得进 32 槽。
- **[309–320 行](../mining.wgsl#L309-L320)**，compact 路径先对每个活跃特征 i 加 `weights[1 + active_indices[i]] * features[i]`。之后对活跃位置 `a`、`b>=a`，取原下标 `i`、`j`，`row = 45 + i*(89-i)/2` 是原始二次项第 i 行的首权重下标，`term = features[a]*features[b]`，加上 `weights[row+j-i]*term`；两个向量分量同时运算。例：i=0 行从权重 45 开始，(0,1) 用 46；i=1 行从 89 开始。该索引是包含常数项的权重下标。
- **[321–342 行](../mining.wgsl#L321-L342)**，`else` 是非 compact 路径：`weight_index=1`；依序扫描所有 44 个线性特征，非零才乘加，但每次都将权重索引加一。二次外层对每个 i：若 `features[i]==0`，则跳过整行 `44-i` 个权重并 `continue`；否则内层 j 从 i 到 43，j 非零时计算乘积并乘加，无论 j 是否为零都将索引加一。两分支都遵循原始非零项顺序与相应权重，不分配完整多项式数组；浮点结果仍受精度和平台实现影响。
- **[343 行](../mining.wgsl#L343)**，`atomicAdd(&output.scored, 1u)` 在完成任一路径的模型计算后累计一次。

## 粗筛与回传

- **[345–351 行](../mining.wgsl#L345-L351)**，`invalid_score` 在任一分数非有限时为真。`possible_hit` 要求 XP 和 XD 两条件都满足：各自阈值为 0，或该近似分数加 10 后不小于阈值。阈值 0 特判允许负分候选继续，因为 Rust 浮点转 u32 会把负值饱和为 0。仅在非 capture_all、分数有限且不可能命中时返回；非有限分数强制回传让 host 明确报错。10 分是粗筛余量，不是此文件内证明的全局误差上界。host 对 capture_all、要求精确或接近非零阈值 10 分内的回传记录用整数结果重建 f64；误差达到 10 会报错，再做最终筛选，见 [mining_gpu.rs:438–467](../src/mining_gpu.rs#L438-L467)。未回传候选无法通过这一步重算来检查误差。
- **[353–364 行](../mining.wgsl#L353-L364)**，`slot = atomicAdd(&output.count, 1u)` 返回递增前的旧值，为每个通过者分配唯一槽；每个有效 invocation 最多写一次，所以 `count <= params.count`，host 按候选数分配时不需额外限容。依次写入本批 `index`、原属性 `props`；循环复制前 16 个 `skill_ids` 和 `skill_freqs`；写 `shadow_numerator`、`score.x`（XP）和 `score.y`（XD）。并发分配的槽顺序不保证是输入顺序；host 回读后按 index 排序，见 [mining_gpu.rs:475–477](../src/mining_gpu.rs#L475-L477)。最后花括号结束入口函数。

## 核对与阅读提示

本说明覆盖模板文件从第 1 行到第 364 行的所有非空有效代码，含各分支、循环、字段和结束作用域；注释与空行不另算执行代码。对应 CPU 算法可对照 [core.rs:38–231](../src/core.rs#L38-L231)，实际资源设置、派发和回读可对照 [mining_gpu.rs](../src/mining_gpu.rs)。本说明是对当前源码的静态核对，不代表在所有 GPU、驱动和输入上进行了运行验证；源文件、host 替换逻辑、行号或模型版本变化时应重新核对。
