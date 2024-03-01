# 名字竞技场 但是 rust

[https://deepmess.com/namerena/](https://deepmess.com/namerena/)

## 用法

```text
Usage: namerena-rs.exe [OPTIONS] --team <TEAM>

Options:
      --start <START>                      开始的 id [default: 0]
      --end <END>                          结束的 id [default: 18446744073709551615]
  -t, --thread-count <THREAD_COUNT>        线程数 [default: 10]
  -p, --prop-expected <PROP_EXPECT>        八围预期值 [default: 740]
      --team <TEAM>                        队伍名称
      --report-interval <REPORT_INTERVAL>  预期状态输出时间间隔 (秒) [default: 10]
  -h, --help                               Print help
```

```text
2024-03-01T18:05:23.839032Z  INFO namerena_rs: | 1|Id:      753431199|595224.82/s 514.274E/d 10.00 ⬆️ 预计:4304342210:30:47|
2024-03-01T18:05:23.840289Z  INFO namerena_rs: | 2|Id:      753355740|595089.89/s 514.158E/d 10.00 ⬆️ 预计:4305318679:54:45|
2024-03-01T18:05:33.662249Z  INFO namerena_rs: | 2|Id:      765257520|605880.29/s 523.481E/d  9.82 ⬆️ 预计:4228638984:28:1|
2024-03-01T18:05:33.663742Z  INFO namerena_rs: | 1|Id:      765335679|605851.08/s 523.455E/d  9.82 ⬆️ 预计:4228841394:50:46|
2024-03-01T18:05:43.562524Z  INFO namerena_rs: | 1|Id:      777452699|612050.45/s 528.812E/d  9.90 ⬆️ 预计:4186010600:17:22|
2024-03-01T18:05:43.570757Z  INFO namerena_rs: | 2|Id:      777375120|611481.35/s 528.320E/d  9.91 ⬆️ 预计:4189905799:2:50|
```
