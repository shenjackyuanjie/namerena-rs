# target-cpu=native 单变量对照（2026-10-30）

此前所有报告使用默认Release，没有启用native。本轮仅改变Rust的-C target-cpu=native，保留相同源码、依赖、默认simd特性、优化等级和codegen/LTO设置。仓库已有的profile.miner没有混入本次对照。

## 构建与特性

本机5800X由rustc识别为znver3。默认print cfg仅列SSE/SSE2/SSE3等基础特性；native增加AVX2、FMA、BMI1/2、POPCNT、AES等。不包含AVX512。RUSTFLAGS和CARGO_ENCODED_RUSTFLAGS原本为空，Cargo配置也未发现隐含native。

默认二进制保留在target/release，native独立放在target/native/release。没有改全局环境、生产源码、权重或默认构建配置。构建命令（临时环境）：

```powershell
$oldFlags = $env:RUSTFLAGS
$oldWrapper = $env:RUSTC_WRAPPER
try {
    $env:RUSTC_WRAPPER = ""
    $env:RUSTFLAGS = "-C target-cpu=native"
    cargo build --workspace --release --locked --target-dir target/native -j 2
} finally {
    $env:RUSTFLAGS = $oldFlags
    $env:RUSTC_WRAPPER = $oldWrapper
}
```

Rust参数是-C target-cpu=native，不是--target-arch=native。这里影响CPU机器代码；WGSL仍由适配器驱动针对当前显卡编译。native二进制只适用于本机或支持所用全部指令集的CPU，分发给其它机器请保留默认构建。

## 串行交错A/B

每种构建先预热，每项三次正式样本，顺序default/native → native/default → default/native。CPU每样本500万、固定-b10000；GPU每样本十亿、原模板/队伍/阈值、packed32、批524287、GPU直接生成、压缩模型、开启verify。没有并行构建/性能实验或终止用户任务。

| 模式 | 默认Release中位数 | native中位数 | 同轮差值 |
| --- | ---: | ---: | ---: |
| CPU 2线程 | 639736.99名/s | 656496.06名/s | **+2.62%** |
| CPU 6线程 | 1683384.19名/s | 1769269.62名/s | **+5.10%** |
| GPU E2E | 37442882名/s | 37753446名/s | **+0.83%，未确认稳定收益** |

CPU两组native的样本区间均高于对应默认样本区间，说明本轮有小幅实际收益；不能把可用AVX2/FMA等同于随机查表和分支循环自动大幅向量化。

GPU正式三轮（万名/s）：默认3744.29、3710.56、3772.92；native3775.34、3714.71、4162.14。两组区间重叠，且native末轮出现约12%的组内漂移，明显大于0.83%的中位差。因此不把最后一个快样本或中位差当作native给GPU带来确定提升的证据。

本轮多数GPU样本低于上一轮约4246万的环境结果，结论采用同轮交错A/B，不跨轮直接相减。本轮没有连续记录GPU时钟、温度和其它桌面负载，不能确定漂移原因；若要确认小于1%的收益，应增加受控重复并同步记录这些指标。

全部正式GPU样本均33个命中，默认/native所有结果CSV SHA256一致。指标同时保留进程启动/退出墙钟，原始数据：[native-comparison.csv](../benchmarks/native-comparison.csv)。

## 复现

```powershell
& ./wgpu-mine/benchmarks/compare-native.ps1 -Output wgpu-mine/benchmarks/native-local-new.csv -LogDir target/native-local-new
```

脚本：[compare-native.ps1](../benchmarks/compare-native.ps1)。round=0是预热，不计入中位数；每条样本即时落盘，旧路径存在时拒绝覆盖。

## 正确性

- native CPU独立金标准测试28通过，仍为原先3个历史模型golden失败，实际数值与默认版本相同，没有新增失败。
- native CPU 24组黑盒：4种编码、1/2/6线程、固定/动态调度、u64上界和CSV特殊字符通过。与默认构建1000个全命中候选的55列（含完整f64评分）逐字段完全一致。
- native GPU包43项默认测试及5项真实硬件测试全部通过。20万样本的整数结果与CPU逐项一致，XP最大误差0.00526540、XD最大误差0.05541708；见[完整验证数据](../benchmarks/native-validation.csv)。
- GPU正例校验与所有构建的结果哈希检查均计入上面持续运行；推理误差预算仍为10，权重没有变化。

本机构建可直接使用target/native/release下的程序；默认二进制继续保留。CPU有小幅收益，GPU暂未证实有稳定增幅。

上阶段GPU极限算法优化及完整计时口径见[GPU极限报告](gpu-extreme.md)。
