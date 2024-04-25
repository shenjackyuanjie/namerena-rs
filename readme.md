# 名字竞技场 但是 rust

[https://deepmess.com/namerena/](https://deepmess.com/namerena/)

## 用法

```text
Usage: tswn.exe [OPTIONS] --team <TEAM>

Options:
      --start <START>                      开始的 id [default: 0]
      --end <END>                          结束的 id [default: 18446744073709551615]
  -t, --thread-count <THREAD_COUNT>        线程数 [default: 10]
  -p, --prop-expected <PROP_EXPECT>        八围预期值 [default: 640]
  -q, --qp-expected <QP_EXPECT>            qp 预期值 [default: 0]
      --team <TEAM>                        队伍名称
  -r, --report-interval <REPORT_INTERVAL>  预期状态输出时间间隔 (秒) [default: 10]
  -h, --help                               Print help
```

```text
2024-04-25T14:38:37.928073Z  INFO tswn: 输出文件: "./namerena/namerena-shenjacks-2024-04-25_22-38-37.txt"
2024-04-25T14:38:37.928448Z  INFO tswn: 开始: 0 结尾: 100000000
2024-04-25T14:38:37.928527Z  INFO tswn: 线程数: 1
2024-04-25T14:38:37.928582Z  INFO tswn: 八围预期: 640
2024-04-25T14:38:37.928633Z  INFO tswn: 队伍名: shenjacks
2024-04-25T14:38:37.928689Z  INFO tswn: 输出文件名: "./namerena/namerena-shenjacks-2024-04-25_22-38-37.txt"
2024-04-25T14:38:37.928773Z  INFO tswn: 开始计算
2024-04-25T14:38:37.928857Z  INFO tswn: 线程 thread_0 开始计算
2024-04-25T14:38:37.947371Z  INFO tswn::cacluate: | 1|Id:          10000|542408.18/s 468.641E/d  0.02⬆️|0  |预计:0:3:4|
2024-04-25T14:38:48.111355Z  INFO tswn::cacluate: | 1|Id:        5434080|533663.84/s 461.086E/d 10.16⬇️|0  |预计:0:2:57|
2024-04-25T14:38:58.566294Z  INFO tswn::cacluate: | 1|Id:       10770710|510448.33/s 441.027E/d 10.45⬇️|0  |预计:0:2:54|
2024-04-25T14:39:08.419676Z  INFO tswn::cacluate: | 1|Id:       15875190|518062.38/s 447.606E/d  9.85⬆️|0  |预计:0:2:42|
2024-04-25T14:39:18.371574Z  INFO tswn::cacluate: | 1|Id:       21055810|520575.43/s 449.777E/d  9.95⬆️|0  |预计:0:2:31|
2024-04-25T14:39:28.277127Z  INFO tswn::cacluate: | 1|Id:       26261560|525550.00/s 454.075E/d  9.91⬆️|0  |预计:0:2:20|
2024-04-25T14:39:38.073338Z  INFO tswn::cacluate: | 1|Id:       31517060|536495.12/s 463.532E/d  9.80⬆️|0  |预计:0:2:7|
2024-04-25T14:39:48.224642Z  INFO tswn::cacluate: | 1|Id:       36882010|528507.78/s 456.631E/d 10.15⬇️|0  |预计:0:1:59|
```
