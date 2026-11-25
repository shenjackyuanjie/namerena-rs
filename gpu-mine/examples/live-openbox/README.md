# PBB 扫描 + 实时开箱

一个进程管完整链路：**pbb-cuda 在 GPU 上扫编号 → 候选一出现就送 tswn-cli 跑满 100% 实战 → 需要的再送 openbox-cli 跑单人组 cqp**。适合「边扫边出分」，不用等扫描跑完再批量开箱。

## 前置

| 组件 | 说明 |
| --- | --- |
| `pbb-cuda.exe` | 含五字填充与 5–50 字节前核的构建，例如 `target/pbb-cuda-multi-final/pbb-cuda.exe` |
| `tswn-cli.exe` | tswn-core release 构建，提供 `namer-pf` |
| `openbox-cli.exe` | tswn-core release 构建，提供 `cqd`（= cqp）；工作目录必须含 `setting/settings.toml` |

## 用法

~~~powershell
python gpu-mine/examples/live-openbox/live_openbox.py `
  --miner  target/pbb-cuda-multi-final/pbb-cuda.exe `
  --tswn-cli  ../tswn-core/target/release/tswn-cli.exe `
  --openbox-cli ../tswn-core/target/release/openbox-cli.exe `
  --cqp-workdir ../tswn-core `
  --plan gpu-mine/examples/live-openbox/plan-qp5200.kv `
  --out target/live-openbox-1t `
  --start 0 --count 1000000000000 `
  --team qyb --name-template "xpxd工作流测试1 []" `
  --pf-batch 32 --pf-thread 10 `
  --cqp-mode qualified --cqp-min-qp 6000
~~~

跑满 1 0000 0000 0000 个编号约 22 分钟；`--seconds 3600` 可改成按时间停。

## 参数

| 参数 | 说明 |
| --- | --- |
| `--plan` | PBB 评分计划；本目录的 `plan-qp5200.kv` 只算 QP 一族 |
| `--start` / `--count` / `--seconds` | 编号区间或运行时长；`--count` 与 `--seconds` 互斥 |
| `--team` / `--name-template` | 模板含一个 `[]`，最终名字是 `模板前缀 + 五字填充 + @队伍` |
| `--pf-n` | pf 场次，10000 = 满 100% |
| `--pf-batch` / `--pf-thread` | 每批送几个候选进 tswn-cli、几线程 |
| `--cqp-mode` | `off` 不跑；`all` 全跑；`qualified` 只跑 pf QP 达标的（默认） |
| `--cqp-min-qp` | `qualified` 模式的 pf QP 下限 |
| `--cqp-preset` | 靶池预设 id，`1` 是「单人组」 |
| `--cqp-n` | 每靶场次，10000 = 满 100% |

## 产物

| 文件 | 内容 |
| --- | --- |
| `scan/results.csv` | pbb-cuda 原始输出（含 `keep` 列） |
| `candidates.jsonl` | 每个候选的 ID、完整名字、模型分 |
| `scored.jsonl` | pf 满 100% 的 qp/pp/pd/qd/sum |
| `cqp.jsonl` | 单人组 cqp 百分比 |
| `status.json` | 进度、积压、`next_id`，每 `--report-interval` 秒刷新 |
| `miner.log` | 矿工原始日志 |

中断后可以用 `status.json` 的 `next_id` 作 `--start` 续跑，输出到新目录。

## 实测产能（RTX 5060 Ti + Ryzen 7 5800X）

模板 `xpxd工作流测试1 []` + 队伍 `qyb`（21 字节前缀）：

| 项 | 实测 |
| --- | ---: |
| 扫描 | **66.72 T/天**（7.72 亿候选/秒） |
| 扫描 + 10 线程 pf 并发 | 66.65 T/天（-0.1%） |
| pf 满 100%，6 线程 | 11.6 个/秒 |
| pf 满 100%，8 线程 | 14.2 个/秒 |
| pf 满 100%，10 线程 | 16.0 个/秒 |
| cqp 单人组满 100%（50 靶 × 1 0000） | 0.43–0.45 秒/候选 |
| cqp 全量批跑 1 2470 个候选 | 95 分钟（0.457–0.513 秒/候选） |
| cqp 只跑 QP 前 10%（1 247 个） | **9 分 43 秒**（0.468 秒/候选） |
| cqp 全量但只跑 10% 场（`-n 1000`） | 0.060 秒/候选，1 2470 个约 12 分钟 |
| pf 持续满载时的扫描 | 60.1 T/天（-10%，pf 让出 2 核可回到 66.7） |

## 门槛与候选数

单族计划实测通过率，用来反推「1 T 出多少候选」：

| 计划 | 通过率 | 每秒候选 | 1 T 候选 | 满速 6 小时 |
| --- | ---: | ---: | ---: | ---: |
| `qp ≥ 4900` | 1.774e-7 | 137 | 17.7 万 | 296 万 |
| `qp ≥ 5200` | 1.269e-8 | 9.8 | **1.27 万** | 21.2 万 |
| `qp ≥ 5400` | 2.06e-9 | 1.6 | 2060 | 3.4 万 |
| `qd ≥ 5800` | 3.42e-10 | 0.27 | 342 | 5470 |
| `qd ≥ 6200` | 4.74e-11 | 0.03 | 47 | 790 |

**注意 PBB 的门控是「任一达标」，不是 AND。** 同时写 `min_qp` 和 `min_qd` 时，效果是「qp 过或 qd 过」，实际候选数接近两条里更宽的那条。`score_mode=all` 只控制是否把族全部算完，不影响 `keep` 判定。

## 积压判断

pf 能力是每秒 16 个（10 线程），所以扫描产出超过这个数就会开始积压：

- `qp ≥ 5200`（9.8 个/秒）→ 能跟上；
- `qp ≥ 4900`（137 个/秒）→ 差 8.6 倍，必须降扫描或加门槛。

cqp 只对 `--cqp-min-qp` 达标的号跑时，成本可以忽略；若 `--cqp-mode all`，单候选 0.45 秒，产出必须压到每秒 2 个以内才不积压。
