# 从零读懂 mining.wgsl：代码、状态与筛选

本文直接按 [mining.wgsl](../mining.wgsl) 的源码顺序贴出完整**上传名字模板**，只在原代码行之间增加中文注释；原有注释也照录。代码块按结构和计算步骤切分，跨块的函数仍属于同一个函数；顺序拼接才是完整模板，不是每一块都能单独编译。代码是阅读主体：从绑定数据开始，跟着一个候选走到回传。

## 阅读导航：先认识执行环境

CPU（host，宿主）准备缓冲区并派发 GPU 计算。一次调用是 invocation，一组调用称 workgroup，组内编号为 lane；这里每个有效 invocation 处理一个候选，各 lane 在共享数组中独占自己的列，没有跨 lane barrier，因此尾批和筛选失败可直接返回。Sbox 是 0..255 的排列；KSA 是密钥调度，PRGA 更新状态并产出字节。这是确定性计算，不是随机试验的保证。

常用记号：u32/i32/f32 分别是 32 位无符号整数、有符号整数和浮点数；u/i 后缀标识整数类型；数组下标从 0 开始，vec2/vec4 有 2/4 个分量。let 不可重新赋值，var 可修改，函数局部变量未显式初始化时按类型置零。表达式里的 & 可取地址，也可按位与；指针前的 * 表示解引用，数值之间的 * 表示乘法。| 是按位或、~ 是按位取反、<< 和 >> 是移位、% 是取余；x & 255 保留低 8 位。select(a, b, 条件) 在条件为假时选 a，为真时选 b。

数据流：恢复队伍或公共前缀状态 → 用名字做两轮 KSA → 提取属性并初筛 → PRGA 洗牌技能 → 按需计算影子 → XP/XD 两模型评分 → 粗筛和回传；宿主再做必要的 f64 精筛。

**模板和实际运行：**下面的常量是源码模板值，不代表生产默认值。[宿主配置](../src/mining_gpu.rs#L21-L60) 替换常量：默认 packed、32 lane、compact；packed 支持 32/64/128 lane，wide 只支持 32。生成模式设置 GENERATED，并用 [名字生成的内嵌代码](gpu-learning-03-other-wgsl.md) 替换 KEY_BEGIN 到 KEY_END 标记区间，不是让上传模板中的空函数直接完成生成。生成代码是独立片段，本文不混贴它，以便核对模板。

## 一、输入资源：队伍状态

```wgsl
// 一个 invocation（GPU 并行调用）处理一个候选，lane 是组内编号；S-box 是 256 字节的排列状态。
// 本文件不含跨 lane barrier，因此尾批与属性筛选均可直接 return。
// host 每批清零 Output 的 16B header，records 至少分配 count 个 176B 槽。
// team_state为raw队伍KSA状态；GENERATED模式则为名字公共前缀KSA快照（各字节占u32）。
// 绑定组 0、编号 0 的只读 uniform；64 个 vec4 共 256 个 u32，每个 u32 存一个状态字节。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>;

```

## 二、名字元数据和字节流

```wgsl
// 上传模式 offset 是拼接字节流的字节偏移，length 是 UTF-8 字节长度，不是字符数；例如 A中占 4 字节。
struct NameInfo { // 上传模式记录名字在字节流中的偏移和 UTF-8 字节长度；生成模式复用这些字段。
    // 从整个字节流开头数的字节偏移；生成模式复用为公共前缀已完成的步数。
    offset: u32,
    // 名字的字节数；影子后缀不包含在这个数内。
    length: u32,
}
// 绑定 1 是只读 storage；上传模式每个候选对应一项 NameInfo。
@group(0) @binding(1) var<storage, read> names: array<NameInfo>; // 绑定 1：只读候选元数据；由宿主按绑定号提供缓冲区。
// 上传模式为紧凑UTF8；GENERATED模式为逐u32模板映射，offset复用为prefix_steps。
// 上传模式每个 u32 打包 4 个 UTF-8 字节，低地址字节放在低位；生成模式另有含义。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>;

```

## 三、输出记录与并发计数

```wgsl
// 严格 44 个 32bit 字，align=4、size/stride=176；不增加 vec3 对齐填充。
// 一个记录由 44 个 32 位字组成，共 176 字节；标量数组避免 vec3 的额外对齐填充。
struct Record {
    index: u32,                       // 字节偏移 0；本批候选下标，宿主据此还原名字。
    props: array<u32, 8>,              // 字节偏移 4；八个属性值。
    skill_ids: array<u32, 16>,         // 字节偏移 36；前 16 个技能编号。
    skill_freqs: array<u32, 16>,       // 字节偏移 100；对应技能频率。
    shadow_numerator: i32,            // 字节偏移 164；影子特征的整数分子。
    xp: f32,                          // 字节偏移 168；GPU 计算的 XP 近似分数。
    xd: f32,                          // 字节偏移 172；GPU 计算的 XD 近似分数。
}
// 前 4 个原子计数器共 16 字节；随后是运行时长度数组，从字节偏移 16 开始。
struct Output { // 输出头的计数器供多个 invocation 并发更新，后面接变长记录数组。
    count: atomic<u32>, // 已领取的记录槽数。
    prop_pass: atomic<u32>, // 进入技能阶段的候选数。
    scored: atomic<u32>, // 完成评分的候选数。
    shadow_count: atomic<u32>, // 计算影子的候选数。
    records: array<Record>,           // 字节偏移 16
}
// 绑定 3 可读写；宿主每批清零头部，并至少给每个候选分配一个记录槽。
@group(0) @binding(3) var<storage, read_write> output: Output; // 绑定 3：可读写输出；每个命中候选占一个记录槽。

```

## 四、参数与模型权重

```wgsl
// 八个 u32，uniform 共 32B；host 将 prop_threshold 限制到 3*p 不溢出。
// 绑定 4 的 uniform 参数共 8 个 u32、32 字节；字段顺序须与宿主一致。
struct Params {
    count: u32, // 批内真实候选数，派发补齐的 invocation 不能越界。
    prop_threshold: u32, // 属性粗筛阈值。
    xp_threshold: u32, // XP 粗筛阈值。
    xd_threshold: u32, // XD 粗筛阈值。
    capture_all: u32, // 非零时保留所有候选，跳过两次淘汰。
    // 生成模式把起始 64 位 ID 拆成低、高两个 u32；pad2 不参与计算。
    pad0: u32, // GENERATED: 起始ID低32bit
    pad1: u32, // GENERATED: 起始ID高32bit
    pad2: u32, // 填充，保持宿主和着色器布局一致。
}
// 只读参数块；宿主把属性阈值限制到 10000，避免后面乘 3 时溢出。
@group(0) @binding(4) var<uniform> params: Params; // 绑定 4：宿主上传的小型只读参数块。
// weights[0] = vec2(BASE, BASE_QD)；weights[k+1] 对应第 k 项 XP/XD 系数。
// 一共 1035 个 vec2<f32>，storage stride=8，共 8280B。
// 绑定 5 的每个 vec2 同时保存 XP 和 XD（CPU 模型称 QD）权重：常数 1 对、线性 44 对、二次 990 对。
@group(0) @binding(5) var<storage, read> weights: array<vec2<f32>>;

```

## 五、模板常量和工作组内存

```wgsl
// 宿主替换编译期常量：LANES 为 32/64/128；packed 每候选 256B、wide 为 1024B，布局分支由常量决定。
// 下面各值可由宿主替换；wide 每候选用 256 个 u32（1024 字节），packed 用 64 个（256 字节）。
const LANES: u32 = 32u; // 每个工作组的 invocation 数；此处是模板值，宿主按配置替换。
const WORDS: u32 = 8192u; // 工作组共享数组长度，以 u32 为单位。
const PACKED: bool = false; // true 时每个 u32 打包四个 S-box 字节；false 时每槽一个 u32。
const GENERATED: bool = false; // true 时从 ID 生成名字；false 时读取上传的名字。
const COMPACT_MODEL: bool = false; // true 时仅遍历非零特征；false 时遍历全部特征。
// 布局按槽转置：同一槽的各 lane 相邻，不同 lane 不写同一列。
var<workgroup> values: array<u32, WORDS>; // 同组共享存储按槽转置，各 lane 独占一列，不依赖跨线程同步。

```

## 六、读取 Sbox 槽

```wgsl
// 给定 lane 和 0..255 的槽号，统一读取 packed 或 wide 状态。
fn read_sbox(lane: u32, slot: u32) -> u32 { // 读取当前 lane 的 S-box 槽；lane 是组内编号，slot 是 0 至 255 的状态下标。
    // packed：每 4 槽放进一个字；先选字，再选该字的字节。
    if (PACKED) {
        // 例如 slot=5、lane=2：字索引为 LANES+2，右移 8 位，再用 255 截出 1 字节。
        return (values[(slot / 4u) * LANES + lane] >> ((slot % 4u) * 8u)) & 255u; // 先选打包字，再右移取出槽内字节；& 255 保留低八位。
    }
    // wide：每槽独占 u32，槽号乘 LANES 后加 lane 得地址。
    return values[slot * LANES + lane]; // 宽字布局：相同槽的不同 lane 在内存中相邻。
}

```

## 七、写入 Sbox 槽

```wgsl
// 调用者传入的是状态字节；写 packed 时只替换目标 8 位。
fn write_sbox(lane: u32, slot: u32, value: u32) { // 写入一个状态字节，保留同一打包字中的其他槽。
    // packed 时多个槽共用一个字，必须保留同字中其他字节。
    if (PACKED) {
        // slot / 4 找到该 lane 的打包字，乘 LANES 跨过之前的槽组。
        let word = (slot / 4u) * LANES + lane;
        // slot % 4 得到字内位置，每字节占 8 位。
        let shift = (slot % 4u) * 8u;
        // 每 lane 独占一列。同一 word 的两个槽交换时，第二次必须读更新后的 word。
        // 先将 255 左移得到目标掩码，取反清空目标位，再把 value 移到目标位置并合并。
        values[word] = (values[word] & ~(255u << shift)) | (value << shift); // 先用掩码清空目标字节，再按位或写入新值。
    // wide 模式无需掩码：一个槽就是一个完整 u32。
    } else {
        // 仍按槽转置寻址，写入的只属于当前 lane。
        values[slot * LANES + lane] = value;
    }
}

```

## 八、恢复队伍或前缀快照

```wgsl
// 每次开始原名字或影子计算前，都恢复宿主提供的初始状态。
fn restore_team(lane: u32) { // 把宿主准备的队伍状态或前缀快照复制到本 lane。
    // packed 模式把 256 个状态字节装成 64 个 u32。
    if (PACKED) {
        // 256 字节打包成 64 个字；每次从一个 vec4 取四个状态字节。
        for (var i = 0u; i < 64u; i += 1u) {
            // v 的 x、y、z、w 分别对应连续四个槽。
            let v = team_state[i];
            // 依次放到位 0、8、16、24；每个目标字只写当前 lane 的列。
            values[i * LANES + lane] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u); // 四个 u32 里的状态字节按低字节在前打包。
        }
    } else {
        // wide 路径逐个复制 256 个槽，无需打包。
        for (var i = 0u; i < 256u; i += 1u) {
            // 整除 4 选 vec4，余数选它的分量。
            values[i * LANES + lane] = team_state[i / 4u][i % 4u];
        }
    }
}

```

## 九、影子后缀的字节

```wgsl
// 以原始名字字节拼接 ?shadow；绝不解析名字内的 @，也不改 raw team。
// 调用方只传合法后缀下标 0..6；值是 ASCII 字节而非 Unicode 字符数。
fn shadow_byte(index: u32) -> u32 {
    // switch 根据位置返回 ?、s、h、a、d、o、w 的字节值。
    switch index {
        case 0u: { return 63u; }   // ?
        case 1u: { return 115u; }  // s
        case 2u: { return 104u; }  // h
        case 3u: { return 97u; }   // a
        case 4u: { return 100u; }  // d
        case 5u: { return 111u; }  // o
        // default 对合法调用只对应第 6 项 w。
        default: { return 119u; } // w，唯一剩余的合法索引为 6
    }
}

```

## 十、名字密钥：上传模板与生成替换

生成模式中，宿主传入 names[0] = [prefix_steps, name_len]、names[1] = [prefix_j, encoding]，绑定 2 为每 u32 一个 token 的模板映射，而不是打包名字。固定字节用 0..255，生成字节位置用 0x100 加位置；例如模板 A[] 的 A 为 65，首个生成字节为 0x100。生成片段按起始 64 位 ID 加批内下标生成 Base16384 或标准/URL Base64；变长 Number 不走该生成路径：自动模式回退上传，强制生成模式会报错。公共前缀快照已处理前导零及首个占位符之前的字节，见 [输入准备](../src/gpu_input.rs#L14-L55) 和 [宿主绑定](../src/mining_gpu.rs#L265-L271)。

```wgsl
// KEY_BEGIN
// 上传模板不生成名字；生成模式由宿主替换整个标记区间，所以实际函数实现不同。
fn prepare_generated(index: u32) {} // 上传模式无需生成名字；生成模式下宿主替换整个标记区间。

// 例如名字 A 的普通键为 [0, 65]，影子键为 [0, 65, 63, 115, 104, 97, 100, 111, 119]。
// k=0 是编码前导零；k>0 对应 name 或 name+?shadow 的第 k-1 个字节。
fn key_byte(info: NameInfo, k: u32) -> u32 {
    // k=0 始终是前导零，不访问名字数据。
    if (k == 0u) { return 0u; }
    // 把包含前导零的键下标换算为不含前导零的名字字节下标。
    let byte_index = k - 1u;
    // 还在原名字范围内时，从上传的字节流读取。
    if (byte_index < info.length) {
        // 名字起点加相对下标，得到整个字节流中的绝对字节地址。
        let address = info.offset + byte_index; // 上传模式：偏移和下标都以字节为单位。
        // 地址除 4 选 u32，余数乘 8 选位移，最后屏蔽其他字节。
        return (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u; // 每个输入字含四字节，除以 4 选字、余数选字节。
    }
    // 超出原名字时只应是影子路径，减去名字长度得到后缀下标。
    return shadow_byte(byte_index - info.length);
}

// KEY_END

```

## 十一、名字的两轮 KSA

```wgsl
// KSA（密钥调度）依据名字反复交换 S-box；名字最长 256B，追加 shadow 后允许 263B。
// 在已恢复的状态上连续做两轮；第二轮沿用第一轮改过的 Sbox，但游标重新从零开始。
fn name_ksa(lane: u32, info: NameInfo, with_shadow: bool) {
    // select 的真假分支分别增加 0 或 7；key_length 不包括额外的前导零。
    let key_length = info.length + select(0u, 7u, with_shadow); // 名字字节长度；影子路径追加七字节 ?shadow。密钥还另有前导零。
    for (var round = 0u; round < 2u; round += 1u) { // 连续执行两轮 KSA，状态保留，游标每轮重置。
        // s 是交换位置，k 是密钥游标，first 是这一轮起始状态槽。
        var s = 0u;
        // k 指向下一次使用的密钥字节，默认从前导零开始。
        var k = 0u;
        // 默认从 Sbox 第 0 槽开始，生成模式可能从快照处续算。
        var first = 0u;
        // 生成模式第一轮接续公共前缀：offset 是已处理步数，names[1].offset 是快照中的 s；第二轮仍从零开始。
        if (GENERATED && round == 0u) { first = info.offset; k = first; s = names[1u].offset; } // 生成模式第一轮从宿主预计算的公共前缀快照继续；第二轮从零开始。
        // 每轮从 first 到 255；长键也只执行最多 256 步，不保证每个字节都在该轮被读取。
        for (var i = first; i < 256u; i += 1u) {
            // 先保存当前槽，之后要和槽 s 交换。
            let current = read_sbox(lane, i);
            // 当前 s、密钥字节和槽值求和后按 256 回绕。
            s = (s + key_byte(info, k) + current) & 255u; // 累加密钥和当前槽，& 255 等价于对 256 取余。
            // 读取目标槽的旧值；即使 i 与 s 相同，这个交换仍成立。
            let other = read_sbox(lane, s);
            // 先把目标值写到 i；packed 下每次写都会重新读取所属字。
            write_sbox(lane, i, other); // 交换 i 与 s 两槽；两次写入各自重新读取打包字。
            // 再把原来的 current 写到 s，完成交换。
            write_sbox(lane, s, current);
            // k 从 0 到 key_length 循环，因此包括前导零和名字的全部字节。
            if (k == key_length) { k = 0u; } else { k += 1u; } // k 在前导零和全部名字字节间循环。
        }
    }
}

```

## 十二、从排列提取 base

```wgsl
// Sbox 是完整排列，仿射映射亦为排列，[89,217) 恰好命中 128 项。
// 按原 Sbox 顺序稳定压缩，不排序，不修改 PRGA 使用的 Sbox。
// base 是指向调用者局部数组的指针，(*base)[count] 表示解引用后写入数组。
fn collect_base(lane: u32, base: ptr<function, array<u32, 128>>) {
    // count 指向下一个待写入位置，初始为零。
    var count = 0u;
    // 按原 Sbox 槽顺序扫描，既不排序也不修改状态。
    for (var i = 0u; i < 256u; i += 1u) {
        // 181 与 256 互质，因此模 256 的仿射映射仍把排列映射成排列。
        let mapped = (read_sbox(lane, i) * 181u + 160u) & 255u; // 对排列做模 256 仿射映射，供后续提取属性与技能数据。
        // 半开区间 [89,217) 恰有 128 个数，所以完整排列恰好命中 128 次。
        if (mapped >= 89u && mapped < 217u) {
            // 只保留低 6 位；例如 89 入选并写入 25，217 不入选。
            (*base)[count] = mapped & 63u; // 保留映射值低六位，按原槽顺序放入 base。
            // 每写入一项后前进，最终恰好填满 base[0..127]。
            count += 1u;
        }
    }
}

```

## 十三、三数中位数与属性

```wgsl
// 三个数的中位数不必排序全部数组；例如 2、8、5 的结果是 5。
fn median3(x: u32, y: u32, z: u32) -> u32 { // 三个值的中位数，不需要对整个数组排序。
    // 先比较 x、y，再用 z 截住范围，重复值也适用。
    return max(min(x, y), min(max(x, y), z));
}

// 从 128 项 base 计算八项属性；排序只影响局部 hp，原 base 还要用于技能。
fn properties(base: ptr<function, array<u32, 128>>) -> array<u32, 8> {
    // HP 排序只作用于局部副本，保留 skills 所用完整原 base。
    // 只复制前 10 项用于 HP。
    var hp: array<u32, 10>;
    // 将前十项逐一复制，避免修改调用方的 base。
    for (var i = 0u; i < 10u; i += 1u) {
        // (*base) 解引用指针，i 是复制的位置。
        hp[i] = (*base)[i];
    }
    // 插入排序：假设 i 左边已升序，把当前值插入正确位置。
    for (var i = 1u; i < 10u; i += 1u) { // 插入排序前十个值；只对 hp 副本操作。
        // 暂存待插入值，避免移动数组时丢失它。
        let value = hp[i];
        // j 是当前空位，从 i 向左移动。
        var j = i;
        // 一直左移直到遇到边界或不大于 value 的前项。
        loop {
            // 到达最左端就停止，防止 j-1 越界。
            if (j == 0u) { break; }
            // 前项已不大于 value，说明当前位置就是插入点。
            if (hp[j - 1u] <= value) { break; }
            // 前项太大时右移一格，给 value 腾位置。
            hp[j] = hp[j - 1u];
            // 空位随之前移一格。
            j -= 1u;
        }
        // 退出循环后把暂存值写回空位。
        hp[j] = value;
    }
    // 结果数组有八项；第 0 项为 HP，其余七项来自三个数的中位数。
    var props: array<u32, 8>;
    // 升序的第 3、4、5、6 项是中间四项，加固定基数 154。
    props[0] = 154u + hp[3] + hp[4] + hp[5] + hp[6]; // 取排序后中间四项之和，加固定基数得到 HP。
    // 其余属性依次处理 i=1..7。
    for (var i = 1u; i < 8u; i += 1u) {
        // 每项读取三个连续 base 值；例如 i=2 从下标 13 开始。
        let start = 10u + (i - 1u) * 3u; // 其余七项每项使用连续三个 base 值的中位数。
        // 三数取中位数后加 36，结果放进对应属性槽。
        props[i] = median3((*base)[start], (*base)[start + 1u], (*base)[start + 2u]) + 36u;
    }
    // 返回八项属性，供筛选、特征和回传使用。
    return props;
}

```

## 十四、PRGA 字节和有限值检查

```wgsl
// a、b 是调用者的局部游标指针；反复调用会延续游标和被原地修改的 Sbox。
fn prga_byte(lane: u32, a: ptr<function, u32>, b: ptr<function, u32>) -> u32 { // PRGA 每次更新同一 lane 的状态并返回一个伪随机字节；a、b 通过指针延续。
    // 先让 a 前进一步并在 255 后回到 0。
    *a = (*a + 1u) & 255u; // & 255 让状态游标在 0 至 255 间循环。
    // 保存当前 a 槽值，之后既用于更新 b，也用于交换。
    let current = read_sbox(lane, *a);
    // b 加上 current 后按 256 回绕。
    *b = (*b + current) & 255u;
    // 读出 b 槽旧值作为交换的另一半。
    let other = read_sbox(lane, *b);
    // 把 b 槽旧值写到 a 槽。
    write_sbox(lane, *a, other);
    // 把 a 槽旧值写到 b 槽，完成交换。
    write_sbox(lane, *b, current);
    // 交换前后两数之和不变，避免再次读取两个随机共享槽。
    // 两数的和交换前后不变；把和按 256 回绕，读出对应状态字节。
    return read_sbox(lane, (current + other) & 255u);
}

// 用位模式判断 f32 是否有限，避免 NaN 在普通比较中造成静默漏报。
fn finite_score(value: f32) -> bool {
    // 位级检测同时覆盖 NaN 和正负 Inf，避免比较 NaN 时静默丢弃记录。
    // bitcast 保留原始位而非数值转换；指数位全 1 表示无穷或 NaN，因此返回假。
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

```

## 十五、计算入口和候选定位

```wgsl
// GPU 入口使用 LANES 大小的工作组；宿主仅沿 x 方向向上取整派发。
@compute @workgroup_size(LANES) // 计算入口：每组 LANES 个 invocation，由宿主按候选数向上取整派发。
// GPU 自动提供三维全局坐标和组内线性编号，本程序使用 x 和 lane。
fn mine(@builtin(global_invocation_id) position: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) { // 全局 x 区分候选，lane 是当前工作组内的编号。
    // global_invocation_id.x 即当前批内候选下标。
    let index = position.x;
    // 例如 count=33、LANES=32 时派发 64 个，后 31 个在这里返回。
    if (index >= params.count) { return; } // 尾组中多派发的 invocation 直接退出。
    // select 在 GENERATED 为假时选 index，为真时选 0；生成模式共用模板元数据。
    let info = names[select(index, 0u, GENERATED)]; // 上传模式取当前候选；生成模式所有候选共用 names[0] 模板信息。
    // 生成模式的函数由独立片段替换，按当前下标准备名字；上传模式不执行。
    if (GENERATED) { prepare_generated(index); } // 仅生成模式按批内下标生成当前名字的编码缓存。

```

## 十六、恢复、调度和提取属性

```wgsl
    // 把初始队伍或公共前缀快照恢复到这个 lane。
    restore_team(lane);
    // 不追加影子后缀，针对原名字完成两轮 KSA。
    name_ksa(lane, info, false);
    // 局部 base 保存 128 项，供属性和技能使用。
    var base: array<u32, 128>;
    // 传入地址，由 collect_base 按排列顺序填充。
    collect_base(lane, &base);
    // 得到原名字的八项属性；后面影子计算不会覆盖这个结果。
    var props = properties(&base);

```

## 十七、整数属性粗筛

```wgsl
    // 这里保存属性总值 HP/3 + 其余七项之和的精确三倍，避免提前整数除法。
    var prop_numerator = props[0]; // 用整数分子表示 HP/3 加其余属性之和，避免提前做浮点除法。
    // 从属性 1 到 7 累加各自的三倍。
    for (var i = 1u; i < 8u; i += 1u) {
        // 加到分子中后，可直接与阈值的三倍比较。
        prop_numerator += 3u * props[i];
    }
    // 非 capture_all 时要求严格大于阈值；例如分子 301 对阈值 100 通过，300 不通过。
    if (params.capture_all == 0u && prop_numerator <= 3u * params.prop_threshold) { // 非 capture_all 时仅保留属性分子严格大于三倍阈值的候选。
        // 属性不达标时直接结束当前 invocation，不进入技能和评分。
        return;
    }
    // capture_all 模式统计所有进入技能阶段者，即本批实际候选数。
    // 通过初筛才计数；capture_all 时所有有效候选都计入。原子操作避免并发丢失更新。
    atomicAdd(&output.prop_pass, 1u); // 原子加法让并发候选安全统计，不保证执行顺序。

```

## 十八、洗牌技能编号

```wgsl
    // 先构造 0..39 的排列，频率只为洗牌后的前 16 个槽计算。
    var skill_ids: array<u32, 40>; // 初始化 40 个技能编号；洗牌后只计算前 16 个的频率。
    // 未显式初始化的局部数组按类型置零。
    var skill_freqs: array<u32, 16>;
    // 遍历全部 40 个技能编号。
    for (var i = 0u; i < 40u; i += 1u) {
        // 初始每个槽的值等于槽号，形成不重复的排列。
        skill_ids[i] = i;
    }
    // 两轮共同保留 a/b/s；每轮 40 项，每项消耗两个 PRGA 字节，共 160B。
    // a、b 是 PRGA 游标，s 是技能交换游标；两轮之间都保留。
    var a = 0u;
    // b 与 a 一起跨 PRGA 调用更新。
    var b = 0u;
    // s 独立于 PRGA 游标，用于选择技能交换槽。
    var s = 0u;
    // 两轮各 40 项，每项消耗 2 个字节，总共 160 字节。
    for (var round = 0u; round < 2u; round += 1u) {
        // 每次选择一个 i，并与由随机字节计算出的 s 槽交换。
        for (var i = 0u; i < 40u; i += 1u) {
            // 取第一个字节作为 16 位值的高字节。
            let high = prga_byte(lane, &a, &b);
            // 取第二个字节作为低字节；a/b 和 Sbox 已被上一次调用更新。
            let low = prga_byte(lane, &a, &b);
            // 高字节左移 8 位与低字节合并，再对 40 取余。
            let rnd = ((high << 8u) | low) % 40u; // 两个随机字节组成大端 16 位数，再映射到 40 个槽。
            // 保留 Rust 的 u8 截断再 %40，不可改成直接 %40。
            // 先模拟 u8 回绕，再对 40 取余；例如和为 256 时结果是 0，而直接 256%40 是 16。
            s = ((s + rnd + skill_ids[i]) & 255u) % 40u; // 先截断到 u8 的范围，再对 40 取余，与 CPU 算法一致。
            // 保存 i 槽旧编号，以便完成交换。
            let current = skill_ids[i];
            // 将 s 槽编号放到 i。
            skill_ids[i] = skill_ids[s];
            // 将 i 槽旧编号放到 s，排列依然不重复。
            skill_ids[s] = current;
        }
    }

```

## 十九、计算技能频率

```wgsl
    // -1 是哨兵，表示目前还没有满足翻倍条件的技能槽。
    var last = -1i; // last 追踪最后一个满足附加条件的有效技能位置。
    // 只为洗牌后的前 16 个槽计算频率。
    for (var i = 0u; i < 16u; i += 1u) {
        // 每个槽取 base 中 4 个连续值，从 64 开始。
        let start = 64u + i * 4u;
        // p 是四项最小值；base[64..127] 恰好容纳 16 组。
        let p = min(min(base[start], base[start + 1u]), min(base[start + 2u], base[start + 3u])); // 四个 base 值取最小值，决定该技能的基础频率。
        // p 必须大于 10，且技能 ID 0..34 才是有效技能；35..39 视为空技能。
        if (p > 10u && skill_ids[i] < 35u) { // p 须大于 10 且技能编号小于 35 才有效。
            // 有效时基础频率为 p-10；例如 p=20 得 10。
            skill_freqs[i] = p - 10u;
            // 仅 p<35 时刷新 last；后面出现的有效槽会覆盖之前的槽。
            if (p < 35u) { last = i32(i); } // 记录最后一个 p 小于 35 的有效位置，供后续翻倍。
        // 无效技能或 p 不够时进入此分支。
        } else {
            // 明确清零，后续不把它写入模型技能特征。
            skill_freqs[i] = 0u;
        }
    }
    // 若找到了符合条件的槽，只翻倍最后一个；没找到就跳过。
    if (last >= 0i) { // 没有符合条件的技能时不做翻倍。
        // 转换为无符号下标前已检查 last 非负。
        skill_freqs[u32(last)] *= 2u;
    }
    // last=14/15 时该槽已翻倍，不再叠加自身的尾槽加成；另一槽照常。
    // 槽 14 未翻倍且自身频率非零时才加尾槽奖励。
    if (last != 14i && skill_freqs[14] != 0u) {
        // 奖励取 base[60]、base[61] 和该频率三者最小值。
        skill_freqs[14] += min(min(base[60], base[61]), skill_freqs[14]);
    }
    // 槽 15 的判断独立：即使槽 14 已翻倍，也可对 15 加成。
    if (last != 15i && skill_freqs[15] != 0u) {
        // 槽 15 对应 base[62]、base[63]，同样不超过该槽当前频率。
        skill_freqs[15] += min(min(base[62], base[63]), skill_freqs[15]);
    }

```

## 二十、组装模型特征

```wgsl
    // 特征 0..7 为属性，8..42 为技能 ID 0..34，43 为影子；未写入的项为零。
    var features: array<f32, 44>; // 44 个特征：前 8 项属性、后续技能频率和影子特征；未赋值项为零。
    // 先把八个整数属性转换为浮点数。
    for (var i = 0u; i < 8u; i += 1u) {
        // 模型使用 f32，因此逐项转换后写入。
        features[i] = f32(props[i]);
    }
    // 遍历前 16 个已洗牌槽，只有非零技能频率进入特征。
    for (var i = 0u; i < 16u; i += 1u) {
        // 零频率包括无效技能；不需要占据模型特征。
        if (skill_freqs[i] != 0u) {
            // 例如 ID 24 映射到特征 32；ID 来自排列，不会互相覆盖。
            features[skill_ids[i] + 8u] = f32(skill_freqs[i]); // 技能编号决定特征下标，而非洗牌后的位置。
        }
    }

```

## 二十一、按需计算影子

```wgsl
    // 用有符号整数保存影子分子，未触发影子时保持 0。
    var shadow_numerator = 0i;
    // 特征 32 对应技能 ID 24；只有它的频率非零才需要影子计算。
    if (features[32] > 0.0) { // 特征 32 对应技能编号 24，只有它有效时才计算影子。
        // 统计实际进入影子分支的候选数。
        atomicAdd(&output.shadow_count, 1u);
        // 技能已结束，直接复用该 lane 的 Sbox/base，原 props/ids/freq 保留。
        // 技能洗牌已完成，复用当前 lane 的共享状态，但先恢复初始队伍或前缀快照。
        restore_team(lane);
        // true 让密钥追加 ?shadow；同一公共原名字前缀仍可使用快照。
        name_ksa(lane, info, true);
        // 重新提取影子的 base，覆盖局部 base；原 props、技能 ID 和频率不受影响。
        collect_base(lane, &base);
        // 从影子 base 计算一份独立的八项属性。
        var shadow_props = properties(&base);
        // 累加影子七项属性去除固定基数 36 后的值。
        var raw_sum = 0u;
        // 遍历属性下标 1..7。
        for (var i = 1u; i < 8u; i += 1u) {
            // properties 对这些项都加了 36，因此先减去该基数。
            raw_sum += shadow_props[i] - 36u;
        }
        // 整数分子允许负值：HP + 3×七项原始值之和，再减去下行的修正项。
        shadow_numerator = i32(shadow_props[0] + 3u * raw_sum) // 以整数分子保存影子属性组合，宿主可据此重建精确值。
            // 减 9 倍第 7 项原始值及常量 630；使用 i32 保留负号。
            - 9i * i32(shadow_props[7] - 36u) - 630i;
        // 分子除 3 再乘技能频率除 100；例如分子 90、频率 20 得 6。
        features[43] = f32(shadow_numerator) / 3.0 * features[32] / 100.0; // 除以 3 并按技能频率缩放，得到影子特征。
    }
    // 特征 42 对应技能 ID 34；只有它已存在时才增加 20。
    if (features[42] > 0.0) { // 技能编号 34 有效时另加 20。
        // 加成发生在影子计算之后、评分之前。
        features[42] += 20.0;
    }

```

## 二十二、紧凑与稠密模型评分

```wgsl
    // 模型有 44 个线性项和 44×45/2=990 个二次项，加常数共 1035 对权重。
    // 与 poly 完全相同的 44 个线性项及 i=0..43,j=i..43 二次项次序。
    // 不构造 1034 项多项式数组；零特征跳过乘加但仍推进权重索引。
    // vec2 的 x 和 y 同时累计 XP 与 XD；先加载各自常数。
    var score = weights[0]; // vec2 同时累加 XP 和 XD，从模型常数项开始。
    // 紧凑路径只计算非零特征，但必须保留它们的原始下标以找到权重。
    if (COMPACT_MODEL) {
        // 最多8属性+16非零技能+1影子=25项；保持原特征索引顺序，32槽留余量。
        // 最多 8 个属性、16 个技能、1 个影子共 25 项，32 个槽足够。
        var active_indices: array<u32, 32>;
        // active_count 同时表示已收集数量和下一个紧凑写入位置。
        var active_count = 0u;
        // 按原始下标从小到大扫描全部 44 项。
        for (var original = 0u; original < 44u; original += 1u) {
            // 先取出原始值，再决定是否保留。
            let value = features[original];
            // 零项对线性和二次乘积均无贡献，可以跳过。
            if (value != 0.0) {
                // 记录原下标，以便压缩后仍能正确定位原模型权重。
                active_indices[active_count] = original; // 保存原始下标，压缩后仍可索引原模型权重。
                // 原地压缩安全：active_count 不超过 original，不会覆盖尚未读取的未来项。
                features[active_count] = value; // 原地压缩非零值；写入位置不超过当前读取位置。
                // 收集完成一项后，活跃数量加 1。
                active_count += 1u;
            }
        }
        // 逐个累加非零线性项。
        for (var i = 0u; i < active_count; i += 1u) {
            // 权重 0 是常数，线性项从 1 开始；按原下标定位系数。
            score += weights[1u + active_indices[i]] * features[i]; // 按原特征下标读取线性项权重。
        }
        // 二次项只取上三角，包括平方项，避免重复计算 i,j 与 j,i。
        for (var a = 0u; a < active_count; a += 1u) {
            // i 是第 a 个活跃特征的原始下标。
            let i = active_indices[a];
            // 45 跳过常数和 44 个线性项；第 i 行起始为 45+i*(89-i)/2，例如第 0 行从 45、第 1 行从 89 开始。
            let row = 45u + i * (89u - i) / 2u; // 跳过常数和 44 个线性项，再定位二次项上三角矩阵第 i 行。
            // b 从 a 起，保证原始 j 不小于 i。
            for (var b = a; b < active_count; b += 1u) {
                // 取出第二个活跃特征的原始下标。
                let j = active_indices[b];
                // 计算特征乘积，可同时用于两个模型。
                let term = features[a] * features[b];
                // row+j-i 定位上三角行内权重；例如 (0,1) 使用权重 46。
                score += weights[row + j - i] * term; // 行内 j-i 偏移对应特征 i 与 j 的乘积系数。
            }
        }
    // 稠密路径扫描全部 44 项；跳过零乘法时仍要推进权重位置。
    } else { // 稠密路径保持原索引顺序，跳过零项乘法但不跳过权重位置。
    // 从权重 1 开始，因为 0 已作为常数加过。
    var weight_index = 1u;
    // 按原顺序处理 44 个线性特征。
    for (var i = 0u; i < 44u; i += 1u) {
        // 仅非零值需要乘加。
        if (features[i] != 0.0) {
            // 同一权重对的两个分量分别累加进 XP 和 XD。
            score += weights[weight_index] * features[i];
        }
        // 无论是否乘加，都前进一个权重。
        weight_index += 1u;
    }
    // 之后按原下标遍历上三角二次项的各行。
    for (var i = 0u; i < 44u; i += 1u) {
        // 若第 i 个特征为零，整行乘积都为零。
        if (features[i] == 0.0) {
            // 即使跳过计算，也要跨过这一行的 44-i 个权重。
            weight_index += 44u - i; // 第 i 行含 44-i 个二次项，整行权重仍须跳过。
            // 直接开始下一行。
            continue;
        }
        // 本行从 j=i 开始，包含平方项。
        for (var j = i; j < 44u; j += 1u) {
            // 仅第二个特征非零时需要乘加。
            if (features[j] != 0.0) {
                // 计算当前两个特征的乘积。
                let term = features[i] * features[j];
                // 按当前顺序取二次权重，两个分数同时累计。
                score += weights[weight_index] * term;
            }
            // 零和非零项都占一个模型权重位置。
            weight_index += 1u;
        }
    }
    }
    // 无论走哪条评分路径，完成后都原子增加 scored。
    atomicAdd(&output.scored, 1u);

```

## 二十三、分数粗筛

10 分余量是**经验性粗筛策略，并非已证明的全局误差上界**。宿主对 capture_all、要求精确或距非零阈值 10 分以内的回传记录，用整数结果重建 f64；误差达到 10 时报告错误，再作最终筛选。未回传的候选无法通过这一步检查误差，详见 [宿主回读](../src/mining_gpu.rs#L438-L467)。

```wgsl
    // 10分余量做候选粗筛；阈值附近由host基于整数记录重建f64精筛。
    // 非有限结果强制记录，交由 host validation 明确报告，不静默通过/丢弃。
    // 任一分数是 NaN 或无穷时，强制回传让宿主明确报告错误。
    let invalid_score = !finite_score(score.x) || !finite_score(score.y); // 只要任一分数非有限，就交给宿主校验。
    // Rust as u32 将负分饱和为 0：阈值为 0 时不能丢弃负分候选。
    // 两个模型都要可能命中；每一项允许阈值为零，或近似分数加 10 不小于阈值。
    let possible_hit = (params.xp_threshold == 0u || score.x + 10.0 >= f32(params.xp_threshold))
        // 阈值为零时即使近似分数为负也不能丢弃：Rust 浮点转 u32 对负值饱和到 0。
        && (params.xd_threshold == 0u || score.y + 10.0 >= f32(params.xd_threshold));
    // 只有非 capture_all、结果有限且判定不可能命中时才返回。
    if (params.capture_all == 0u && !invalid_score && !possible_hit) { return; } // 非 capture_all 且分数有限、按 10 分余量粗筛判为不可能命中时才淘汰。

```

## 二十四、原子分配记录并回传

```wgsl
    // 每 invocation 最多写一次，故 count <= params.count；无需限容竞争或回退。
    // atomicAdd 返回增加之前的旧值，作为当前候选唯一的写入槽。
    let slot = atomicAdd(&output.count, 1u); // 原子领取唯一槽位；写入顺序不等于候选顺序。
    // 回传原批内下标；槽位分配顺序不保证输入顺序，宿主之后按 index 排序。
    output.records[slot].index = index; // 记录原下标，方便宿主排序和还原候选。
    // 写入原名字的八项属性，不是影子属性。
    output.records[slot].props = props;
    // 逐项写入前 16 个技能槽及其频率。
    for (var i = 0u; i < 16u; i += 1u) {
        // 写洗牌后的技能编号。
        output.records[slot].skill_ids[i] = skill_ids[i];
        // 写同一槽对应的频率，宿主可据此重建特征。
        output.records[slot].skill_freqs[i] = skill_freqs[i];
    }
    // 回传有符号影子分子，未触发影子时为 0。
    output.records[slot].shadow_numerator = shadow_numerator;
    // vec2 的 x 是 XP 的 f32 近似分数。
    output.records[slot].xp = score.x;
    // vec2 的 y 是 XD 的 f32 近似分数；完成后本 invocation 结束。
    output.records[slot].xd = score.y;
}
```

## 接下来怎么核对

原始算法可对照 [core.rs](../src/core.rs#L38-L231)，实际资源设置、派发和回读可对照 [mining_gpu.rs](../src/mining_gpu.rs)。本文覆盖上传模板的全部源码；生产模式还会替换常量和生成片段。源码、宿主逻辑或模型变化后，应重新核对；静态讲解不等于已在所有 GPU、驱动和输入上验证。
