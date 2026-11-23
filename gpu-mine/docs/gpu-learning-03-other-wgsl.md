# GPU 入门 03：名字生成片段与两个 RC4 实验着色器

本文按原顺序直接展示三份源码，保留原注释，并在代码附近补充中文说明。`u32` 是 32 位无符号整数，`&`、`|` 是按位与、或，`>>`、`<<` 是移位，`& 255` 留下低 8 位；`select(a, b, 条件)` 为假选 a、为真选 b。UTF-8 字节数不一定等于字符数。

每份文件按源码顺序拆成几个代码块；同一文件的代码块顺序拼接后是完整源码，文档额外增加了注释。单个代码块不一定能独立编译。这里是当前源码快照，日后修改着色器时也需要同步文档。

一次 GPU compute 派发含多个工作组，每组有多个 invocation；一维派发的 `position.x` 是批内全局编号，`lane` 是组内编号。KSA 是 RC4 密钥调度：反复根据密钥和当前状态更新游标、交换 256 槽状态盒。本篇的两份实验只返回 KSA 状态，并不完成属性、技能或模型等完整测号。完整路径见 [正式内核的内嵌代码](gpu-learning-02-mining-wgsl.md)，实验选择和工作组大小见 [gpu.rs:12–24](../src/gpu.rs#L12-L24)。

## 一、generated-key.wgsl：生成名字的替换片段

[generated-key.wgsl](../generated-key.wgsl) 不是独立计算入口：生成模式下，Rust 把它插入 [mining.wgsl](../mining.wgsl#L102-L116) 的 `KEY_BEGIN` 与 `KEY_END` 标记区间，替换原有函数，并设置 `GENERATED`；它借用正式内核已有的类型、绑定和 `shadow_byte`，并非 WGSL 的 import（[mining_gpu.rs:37–59](../src/mining_gpu.rs#L37-L59)）。上传模式保留原区间。

宿主为模板每个最终名字字节上传一个 u32 token：固定 UTF-8 字节为 0～255，`0x100 | n` 指向候选编码的第 n 字节。多个 `[]` 引用同一编码；Base16384 为 18 字节，两种 Base64 为 12 字节，名字最多 256 字节（[gpu_input.rs:14–38](../src/gpu_input.rs#L14-L38)）。例如 Base64 模板 `A[]!` 的 token 是 65、256～267、33。生成模式中 `names[0]` 的 `offset` 是首轮公共前缀步数而非字节地址，`length` 是名字字节数；`names[1]` 的两个字段存前缀 j 和编码编号。CPU 预跑公共前缀，第一轮从快照续算，第二轮从零开始（[gpu_input.rs:39–55](../src/gpu_input.rs#L39-L55)、[mining.wgsl:118–135](../mining.wgsl#L118-L135)）。

批起点拆为高低两个 u32，低位加批内编号产生进位；输入位序按 64 位大端理解，而编码结果在每个 u32 **低字节先存**，两种顺序是不同概念。举例，字节 `11 22 33 44` 打包成数值 `0x44332211`，第 2 个字节（从 0 数）是 `(word >> 16) & 255 = 0x33`。宿主会检查 ID 范围溢出（[mining_gpu.rs:323–340](../src/mining_gpu.rs#L323-L340)）。

```wgsl
// 每个 invocation 独占缓存：Base16384为6个三字节UTF-8字，Base64为3个四字节字。
// 此片段替代原key_byte，不新增绑定；names[1].length存编码编号，params.pad0/1存起始ID。宿主会整体替换 KEY_BEGIN 与 KEY_END 间的函数定义；它不是独立计算入口。
// 六个槽属于当前 invocation；不同候选互不覆盖。每个槽是 32 位无符号整数。
var<private> generated_chunks: array<u32, 6>; // private 表示每个 invocation 拥有自己的缓存。

// 入口由正式内核调用，index 是本批中的候选编号，不是全局持久 ID。
fn prepare_generated(index: u32) { // 根据批次起始 ID 和当前候选编号生成固定宽度编码。
    // 把批起点的低 32 位加上批内编号；u32 加法按 2³² 回绕。
    let low = params.pad0 + index;
    // 若结果比原低位小，则低位溢出，select 返回 1 并向高位进位；否则返回 0。例：ffffffff + 1 得 0，high 增 1。
    let high = params.pad1 + select(0u, 1u, low < params.pad0); // 低 32 位回绕时向高 32 位进一，拼成 64 位 ID。
    // 宿主约定 0 为 Base16384、1 为标准 Base64、2 为 URL 安全 Base64。
    let encoding = names[1].length; // 0 为 Base16384，1 为标准 Base64，2 为 URL 安全 Base64。
    // 先处理固定宽 18 字节的 Base16384；其他受支持编码进入下方 Base64。
    if (encoding == 0u) { // Base16384 把高位在前的 ID 位段映射为六个 Unicode 码点。
        // 64位大端输入：前56bit拆四组14bit，最后8bit左移6，末尾是固定㴁。
        // 依次构造六个 Unicode 码点，每个码点最终占三个 UTF-8 字节。
        for (var group = 0u; group < 6u; group += 1u) {
            // point 暂存当前组的码点，随后由 switch 按组号赋值。
            var point = 0u;
            // 64 位 ID 按大端顺序理解：先处理 high 的高位，再处理 low。
            switch group {
                // 取 high 第 31～18 位的 14 位，映射到从 U+4E00 开始的范围。
                case 0u: { point = 0x4e00u + ((high >> 18u) & 0x3fffu); }
                // 取 high 第 17～4 位；0x3fff 保留低 14 位。
                case 1u: { point = 0x4e00u + ((high >> 4u) & 0x3fffu); }
                // 跨边界组：high 低 4 位接 low 高 10 位，再保留 14 位。
                case 2u: { point = 0x4e00u + (((high << 10u) | (low >> 22u)) & 0x3fffu); } // 跨越高低字边界：高字低 4 位与低字高 10 位组成一组。
                // 取 low 第 21～8 位的 14 位，仍加 U+4E00。
                case 3u: { point = 0x4e00u + ((low >> 8u) & 0x3fffu); }
                // low 最后 8 位左移 6 位补零，形成第五个 14 位组。
                case 4u: { point = 0x4e00u + ((low & 255u) << 6u); } // 最后 8 位左移 6 位补零，形成第 5 组。
                // 第六组不是 ID 位段，而是固定结尾 U+3D01，即“㴁”。
                default: { point = 0x3d01u; } // 第 6 个码点是固定结尾。
            }
            // U+4E00..U+8DFF和U+3D01均是三个字节的合法UTF-8，最高U+8DFF不排除。
            // UTF-8 首字节取码点高位并带前缀 1110；放入 u32 最低字节。
            generated_chunks[group] = (0xe0u | (point >> 12u)) // 按 UTF-8 三字节格式编码，每字节依次放入 u32 的低到高位。
                // 第二字节取中间 6 位并带前缀 10，左移 8 位放到第二字节。
                | ((0x80u | ((point >> 6u) & 63u)) << 8u)
                // 第三字节取最低 6 位，左移 16 位放到第三字节；最高字节未使用。
                | ((0x80u | (point & 63u)) << 16u);
        }
        // 六组已全部生成，直接结束函数，不再执行下方 Base64 分支。
        return;
    }
    // Base64的前十字符消费60bit，第11字符保存剩余4bit<<2，第12字符固定'='。
    // Base64 共 12 字符；外层三组，每组打包四个 ASCII 字节。
    for (var group = 0u; group < 3u; group += 1u) { // Base64 共 12 个字符，每个 u32 打包四个 ASCII 字节。
        // 清空当前组的打包字，避免混入上一组内容。
        var packed = 0u;
        // 组内 offset 从 0 到 3，表示当前字符放在哪个字节位置。
        for (var offset = 0u; offset < 4u; offset += 1u) {
            // 把组号与组内偏移合成 0～11 的完整字符序号。
            let character = group * 4u + offset;
            // ASCII 61 是 =；先设为填充符，最后一个字符会保留这个值。
            var ascii = 61u; // 默认填充字符 =；只有前 11 个字符需要编码。
            // 前 11 个字符才包含 ID 的编码位；第 12 个字符仍为 =。
            if (character < 11u) {
                // value 保存当前字符对应的六位整数，范围是 0～63。
                var value = 0u;
                // 前五个字符全部取自 high，依次从最高位开始。
                if (character < 5u) {
                    // 按字符位置右移，& 63 截取连续 6 位。
                    value = (high >> (26u - character * 6u)) & 63u;
                // 第六字符跨越 high 与 low 的边界。
                } else if (character == 5u) {
                    // 拼接 high 末 2 位和 low 前 4 位，& 63 去掉多余位。
                    value = ((high << 4u) | (low >> 28u)) & 63u; // 第 6 个字符跨越高低字边界，取高字末 2 位和低字前 4 位。
                // 第七至第十字符完全取自 low，继续依次截取 6 位。
                } else if (character < 10u) {
                    // 58 - character*6 计算此字符对应的右移位数。
                    value = (low >> (58u - character * 6u)) & 63u;
                // 第十一字符使用 low 剩余的最低 4 位。
                } else {
                    // 左移两位补零并保留低六位，以满足 Base64 的末组格式。
                    value = (low << 2u) & 63u; // 最后剩余 4 位左移 2 位补零。
                }
                // 查 Base64 字母表：先判断大写字母 A～Z。
                if (value < 26u) { // 六位值映射到 Base64 字母表：大写、小写、数字及两个符号。
                    // ASCII A 为 65，将 0～25 映射为 A～Z。
                    ascii = 65u + value;
                // 值 26～51 对应小写字母。
                } else if (value < 52u) {
                    // ASCII a 为 97，减去 26 把值换成从零开始的下标。
                    ascii = 97u + value - 26u;
                // 值 52～61 对应数字字符。
                } else if (value < 62u) {
                    // ASCII 0 为 48，减去 52 得到数字下标。
                    ascii = 48u + value - 52u;
                // 值 62 对应两种字母表中的第一个特殊字符。
                } else if (value == 62u) {
                    // 标准 Base64 用 +（43），URL 安全版本用 -（45）。
                    ascii = select(43u, 45u, encoding == 2u); // 标准表用 +，URL 安全表用 -。
                // 剩余的值 63 对应第二个特殊字符。
                } else {
                    // 标准 Base64 用 /（47），URL 安全版本用 _（95）。
                    ascii = select(47u, 95u, encoding == 2u); // 标准表用 /，URL 安全表用 _。
                }
            }
            // 字符放到 offset 对应的字节位置，按位或到打包字；字符顺序为低字节先。
            packed |= ascii << (offset * 8u); // 按字符在组内的位置放入对应字节，低字节先输出。
        }
        // 四个 ASCII 字节组成的 u32 保存到当前组缓存。
        generated_chunks[group] = packed;
    }
}

```

接下来是密钥字节读取函数：它把固定模板、生成字节和可选 shadow 后缀连接成 KSA 的逻辑密钥。

```wgsl
// info.offset是prefix_steps而非字节地址；template_map逐u32上传，不再是packed原名字。names[0].length 是名字字节长度；names[1] 保存前缀游标和编码编号。
// 返回 KSA 密钥在位置 k 的一个字节；info 和 shadow_byte 来自正式内核。
fn key_byte(info: NameInfo, k: u32) -> u32 {
    // 约定密钥以一个零字节开头，k 为零时无需读取名字。
    if (k == 0u) { return 0u; } // KSA 的密钥有一个隐式前导零。
    // 其余 k 从 1 开始，减一变成从零开始的名字字节下标。
    let byte_index = k - 1u;
    // 名字范围内查模板映射；范围之外交给 shadow 后缀。
    if (byte_index < info.length) {
        // 生成模式每个 token 单独占一个 u32，这里不是上传模式的四字节打包数组。
        let token = name_bytes[byte_index]; // 每个 token 占一个 u32：常量字节或生成字节的引用。
        // 0～255 表示模板中的固定 UTF-8 字节，原样返回。
        if (token < 256u) { return token; } // 小于 256 的 token 直接是固定字节。
        // 0x100 | n 表示取生成结果第 n 字节；低八位得到偏移 n。
        let offset = token & 255u; // 其他 token 的低八位是编码结果中的字节偏移。
        // 编码 0 的每个缓存槽存一个三字节 UTF-8 码点。
        if (names[1].length == 0u) {
            // 除以 3 选码点槽，余数选槽内字节；右移后 & 255 截出该字节。
            return (generated_chunks[offset / 3u] >> ((offset % 3u) * 8u)) & 255u; // Base16384 每槽三字节；除法选槽，余数选槽内字节。
        }
        // Base64 每槽四字节：除以 4 选槽，余数选对应 ASCII 字节。
        return (generated_chunks[offset / 4u] >> ((offset % 4u) * 8u)) & 255u; // Base64 每槽四字节，同样右移并掩码提取。
    }
    // 名字之后仅 shadow 路径会访问此处；依次取 ?shadow 的七个字节。
    return shadow_byte(byte_index - info.length); // 只有影子密钥越过名字长度后才读取 ?shadow。
}
```

## 二、main.wgsl：每个候选使用私有状态的 RC4 实验

[main.wgsl](../main.wgsl) 是独立实验 compute 内核，入口 `rc4_name`，每组 64 个 invocation。CPU 先算队伍状态；每个候选复制这份状态，再对 `[0] + UTF-8(name)` 做两轮 KSA，只输出完整状态盒，不进行完整测号。宿主拼接名字字节并记录偏移、长度，按四字节打包上传（[gpu.rs:211–256](../src/gpu.rs#L211-L256)）。例如名字 `A` 的密钥依次循环 0、65；空名字只有前导零。

```wgsl
// 每个候选在独立 invocation 内执行两轮名字 KSA；队伍 KSA 在 CPU 预计算一次。这是 RC4 对照实验，只回传状态，不计算完整属性、技能或模型。
// 绑定 0 是只读 uniform：64 个四分量向量，共 256 个预计算队伍状态值。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>; // 绑定 0：宿主预计算的 256 个队伍状态值，四个一组放入 uniform。
// 元数据 offset 是连续名字字节流中的起始偏移，length 是 UTF-8 字节数。
struct NameInfo { offset: u32, length: u32, } // offset 和 length 均以 UTF-8 字节为单位。
// 绑定 1 是只读 storage 数组，每个候选对应一条元数据。
@group(0) @binding(1) var<storage, read> names: array<NameInfo>; // 绑定 1：只读候选元数据数组。
// UTF-8 字节连续打包，每个 u32 的低字节在前，避免将每个输入字节扩张成 u32。
// 绑定 2 存连续名字字节；每 4 个字节按低字节在前装入一个 u32。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>; // 绑定 2：按低字节在前打包的只读名字字节。
// 绑定 3 保存输出，每个候选需要 64 个 u32，即完整 256 字节状态。
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>; // 绑定 3：可写结果，每个候选 64 个 u32。
// count 是本批真实候选数；三个填充字使参数块达到 16 字节。
struct Params { count: u32, padding0: u32, padding1: u32, padding2: u32, } // count 是真实候选数，其他字段用于填充到 16 字节。
// 绑定 4 是宿主上传的只读批次参数。
@group(0) @binding(4) var<uniform> params: Params; // 绑定 4：宿主上传的只读批次参数。

```

以上为绑定和数据结构；下面是完整计算入口，接续前面的声明。

```wgsl
// compute 派发以 64 个 invocation 为一组，一个 invocation 负责一个候选。
@compute @workgroup_size(64) // 每组 64 个 invocation；一个 invocation 处理一个候选。
// global_invocation_id 是三维全局编号，这里只使用 x 分量。
fn rc4_name(@builtin(global_invocation_id) position: vec3<u32>) {
    // index 同时用于查候选元数据和定位对应输出。
    let index = position.x; // 只使用三维全局编号的 x 分量作为候选下标。
    // 宿主按工作组大小向上取整；尾组补出来的编号必须在访问缓冲前退出。
    if (index >= params.count) { return; } // 派发按组数向上取整，超出真实数量的线程退出。
    // 读取当前候选的字节偏移与长度。
    let info = names[index];
    // 函数局部状态数组只属于当前 invocation，其他候选不会访问。
    var values: array<u32, 256>; // 函数内局部数组属于当前 invocation，其他候选不能访问。
    // 遍历状态盒的 256 个槽，将 CPU 预计算的队伍状态复制过来。
    for (var i = 0u; i < 256u; i += 1u) {
        // i / 4 选向量，i % 4 选分量；状态值保存在 values 的同一槽。
        values[i] = team_state[i / 4u][i % 4u]; // 从 64 个四分量向量中取出 256 项队伍状态。
    }
    // 密钥为 [0, UTF-8(name)...]；每一轮都重置局部 j 和密钥游标。
    // 名字 KSA 做两轮；第二轮沿用第一轮交换后的状态，不重新复制队伍状态。
    for (var round = 0u; round < 2u; round += 1u) { // 两轮保留 S-box 状态，但每轮重置 j 和密钥游标。
        // j 是交换目标游标；每轮从 0 开始。
        var j = 0u;
        // k 是密钥游标；密钥逻辑上是前导零加名字的全部 UTF-8 字节。
        var k = 0u;
        // 每轮处理状态盒的 256 个槽，i 从 0 到 255。
        for (var i = 0u; i < 256u; i += 1u) {
            // k=0 表示前导零，不需要加值；k>0 时读取名字的第 k-1 字节。
            if (k != 0u) { // k 为零表示隐式前导零，否则读取名字的第 k-1 个字节。
                // 把名字内部下标加上本名字在连续字节流中的偏移。
                let address = info.offset + k - 1u;
                // 除以 4 选 u32，余数乘 8 选低到高的字节，右移并 &255 后加入 j。
                j += (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u; // 除以 4 选打包字，余数乘 8 选字节，& 255 截取低八位。
            }
            // 加上当前状态值后取低 8 位，相当于模 256，得到 0～255 的目标槽。
            j = (j + values[i]) & 255u; // 累加当前状态并对 256 取余，得到交换目标 j。
            // 先保存 i 槽旧值，避免下面覆盖时丢失。
            let temporary = values[i]; // 保存旧值，交换当前槽和目标槽；相同槽也安全。
            // 将 j 槽值写入 i 槽。
            values[i] = values[j];
            // 将保存的旧 i 值写入 j 槽，完成交换；i=j 也安全。
            values[j] = temporary;
            // k 在 0～length 间循环；空名字时一直为 0，不会出现零长度取模。
            if (k == info.length) { k = 0u; } else { k += 1u; } // 循环使用前导零和全部名字字节，空名字也能循环。
        }
    }
    // 保留完整置换但只回读 256 字节/名字，而不是原型的 1024 字节。
    // 每次输出四个槽，因此 256 个槽需要循环 64 次。
    for (var i = 0u; i < 64u; i += 1u) {
        // base 是这一组四个状态字节的起始槽下标。
        let base = i * 4u;
        // 按候选和组号定位输出；前两个状态值放在结果字最低和次低字节。
        output_states[index * 64u + i] = values[base] | (values[base + 1u] << 8u) // 四个状态字节依次放入结果字的低至高位，结果按候选连续排列。
            // 后两个值分别放在第 3、4 字节；或运算合并四个互不重叠的字节。
            | (values[base + 2u] << 16u) | (values[base + 3u] << 24u);
    }
}
```

## 三、rc4-shared.wgsl：按 lane 转置的工作组状态实验

[rc4-shared.wgsl](../rc4-shared.wgsl) 与上一节是互斥选择的实验内核，同样只返回两轮 KSA 状态，并非完整测号；每组 32 个 invocation。工作组数组布局是 `values[槽号 * 32 + lane]`，同槽的不同 lane 相邻。每个 invocation 只初始化、交换、读取自己的列，因此不依赖其他 lane 的数据，不需要跨 lane 的 barrier；尾组的无效 invocation 可以提前退出。若将算法改为跨列通信，就必须重新设计同步及一致控制流，不能直接沿用此提前退出方式。转置旨在减少 bank 冲突，实际效果取决于硬件；该实现要求至少 32 KiB 工作组存储（[gpu.rs:76–83](../src/gpu.rs#L76-L83)）。

```wgsl
// 32 个候选共享一块 32KiB workgroup 状态，每列属于一个 invocation。
// 转置布局让相同槽的 lane 相邻，意在减少共享内存 bank 冲突；实际效果取决于硬件。
// 绑定 0 是只读 uniform，64 个四分量向量合计 256 个队伍状态值。
@group(0) @binding(0) var<uniform> team_state: array<vec4<u32>, 64>; // 绑定 0：宿主预计算的队伍状态，64 个四分量共 256 项。
// offset 和 length 描述候选在连续 UTF-8 字节流中的位置与字节数。
struct NameInfo { offset: u32, length: u32, } // offset 和 length 是名字在连续 UTF-8 字节流中的偏移与长度。
// 绑定 1 提供每个候选的只读元数据。
@group(0) @binding(1) var<storage, read> names: array<NameInfo>; // 绑定 1：只读候选元数据。
// 绑定 2 将连续名字字节按低字节在前，每四字节打包成一个 u32。
@group(0) @binding(2) var<storage, read> name_bytes: array<u32>; // 绑定 2：每个 u32 低字节在前的名字数据。
// 绑定 3 是可写输出，每个候选保留 64 个 u32 状态。
@group(0) @binding(3) var<storage, read_write> output_states: array<u32>; // 绑定 3：可写状态结果，每个候选 64 个 u32。
// count 是真实候选数，三个填充字段使参数块为 16 字节。
struct Params { count: u32, padding0: u32, padding1: u32, padding2: u32, } // count 是真实候选数，三个字段填充参数块到 16 字节。
// 绑定 4 提供当前批次参数。
@group(0) @binding(4) var<uniform> params: Params; // 绑定 4：宿主上传的只读批次参数。
// 工作组共享数组有 256×32 个 u32，占 32768 字节；每个 lane 独占一列。
var<workgroup> values: array<u32, 8192>; // 256 个槽乘 32 列，每列独占一个候选，共 32 KiB。

```

以上为绑定与共享数组；下面是计算入口，接续前面的声明。

```wgsl
// 每组固定 32 个 invocation；只回传两轮名字 KSA 结果。
@compute @workgroup_size(32) // 每组 32 个 invocation；这里只回传两轮 KSA 状态。
// position.x 是批内全局候选编号，lane 是组内 0～31 的编号。
fn rc4_name(@builtin(global_invocation_id) position: vec3<u32>, @builtin(local_invocation_index) lane: u32) { // position.x 是批内全局候选编号，lane 是组内编号。
    // 用全局 x 编号选择候选和它的输出区域。
    let index = position.x;
    // 尾组向上补齐时，多余 invocation 在访问名字或状态前退出。
    if (index >= params.count) { return; } // 尾组补齐的 invocation 不处理候选，也不访问数组。
    // 读取当前候选的字节偏移和长度。
    let info = names[index];
    // 逐槽复制 CPU 预计算的队伍状态，准备当前候选的状态盒。
    for (var i = 0u; i < 256u; i += 1u) {
        // 转置地址为槽号×32+lane；例如槽 5、lane 2 位于 162，其他 lane 使用不同地址。
        values[i * 32u + lane] = team_state[i / 4u][i % 4u]; // 将队伍状态复制到本 lane 的转置列：槽 i 的地址为 i*32+lane。
    }
    // 每列由同一个 invocation 初始化并读写，不跨列交换，无须轮内 barrier。
    // 执行两轮名字 KSA；状态跨轮保留，j 与 k 每轮重新从零开始。
    for (var round = 0u; round < 2u; round += 1u) { // 两轮连续更新状态；每轮重新从零设置 j 和密钥游标 k。
        // j 跟踪当前轮的交换目标槽。
        var j = 0u;
        // k 跟踪密钥位置：0 表示前导零，其余位置对应名字字节。
        var k = 0u;
        // 每轮从 i=0 处理到 i=255，依次更新每个槽。
        for (var i = 0u; i < 256u; i += 1u) {
            // 前导零不改变 j，只有 k 非零时才读取名字字节。
            if (k != 0u) { // k=0 对应隐式前导零，其余位置读取名字字节。
                // 将名字内部位置 k-1 转成连续输入流中的绝对字节地址。
                let address = info.offset + k - 1u;
                // 除以 4 选打包字，余数乘 8 选字节，右移并掩码后累加到 j。
                j += (name_bytes[address / 4u] >> ((address % 4u) * 8u)) & 255u; // 除以 4 选输入字，余数选字节，右移并掩码提取。
            }
            // 把当前槽 i 转成共享数组中的地址，并固定使用本 invocation 的 lane。
            let address_i = i * 32u + lane;
            // 保存当前槽旧值，既用于计算 j，也用于后面的交换。
            let current = values[address_i];
            // 加当前值后取低 8 位，相当于对 256 取余。
            j = (j + current) & 255u; // 加当前槽值后对 256 取余，得到交换目标 j。
            // 把目标槽 j 映射到同一 lane 的列，绝不读写其他候选的列。
            let address_j = j * 32u + lane; // 目标地址仍在同一 lane 的列中。
            // 先把目标槽的值写到当前槽；当前旧值已经保存在 current。
            values[address_i] = values[address_j]; // 交换两个槽；即使 i 与 j 相同也安全。
            // 再把旧当前值写到目标槽；两个槽相同也不会破坏数据。
            values[address_j] = current;
            // 密钥游标在 0～length 间循环，包含前导零；空名字也有效。
            if (k == info.length) { k = 0u; } else { k += 1u; } // 密钥按前导零和全部名字字节循环。
        }
    }
    // 256 个状态槽四个一组，循环 64 次打包输出。
    for (var i = 0u; i < 64u; i += 1u) { // 每四个状态槽打包成一个 u32，每候选输出 64 个字。
        // base 是这一组的首个状态槽编号。
        let base = i * 4u;
        // 当前候选的输出区从 index×64 开始；第一个状态值占最低字节。
        output_states[index * 64u + i] = values[base * 32u + lane] // 依次将四个槽放到结果字的低至高字节，按候选连续排列。
            // 读取同一 lane 的下一槽，左移 8 位放入第二字节。
            | (values[(base + 1u) * 32u + lane] << 8u)
            // 第三槽左移 16 位，放入第三字节。
            | (values[(base + 2u) * 32u + lane] << 16u)
            // 第四槽左移 24 位，四个字节按位或为一个输出 u32。
            | (values[(base + 3u) * 32u + lane] << 24u);
    }
}
```

可继续参考 [固定编码与进位测试](../src/gpu_input.rs#L154-L210)、[实验内核对照测试](../src/gpu.rs#L320-L340) 和 [RC4 实验记录](rc4-experiment.md)。测试源码存在不代表本次文档修改实际运行过 GPU 测试。
