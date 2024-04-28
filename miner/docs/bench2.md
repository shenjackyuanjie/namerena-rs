# 还是benchmark

> --team shenjacka -q 5500 --end 1000000000 --bench
>
> 5600u 单核

## `RUSTFLAGS="-C target-cpu=native"` 0.1.10

```text
❯ .\runs\tswn-0110-native.exe --team shenjacka -q 5500 --end 1000000000 --bench
2024-04-28T13:20:10.027586Z  INFO tswn: 输出文件: "./namerena/namerena-shenjacka-2024-04-28_21-20-10.csv"
2024-04-28T13:20:10.027835Z  INFO tswn: 开始: 0 结尾: 1000000000
2024-04-28T13:20:10.027905Z  INFO tswn: 线程数: 10
2024-04-28T13:20:10.027973Z  INFO tswn: 八围预期: 640
2024-04-28T13:20:10.028035Z  INFO tswn: 队伍名: shenjacka
2024-04-28T13:20:10.028094Z  INFO tswn: 输出文件名: "./namerena/namerena-shenjacka-2024-04-28_21-20-10.csv"
2024-04-28T13:20:10.028153Z  INFO tswn: 预期状态输出时间间隔: 10 秒
2024-04-28T13:20:10.028215Z  INFO tswn: 是否启动 benchmark 模式: true
2024-04-28T13:20:10.028273Z  INFO tswn: 开始 benchmark
2024-04-28T13:20:10.028348Z  INFO tswn: 设置进程亲和性成功 1
2024-04-28T13:20:10.028444Z  INFO tswn: 设置线程亲和性成功 1
2024-04-28T13:20:10.194177Z  INFO tswn::cacluate: | 1|Id:         999991|603371.40/s 521.313E/d  0.17⬆️|0  |预计:0:2:45|
2024-04-28T13:20:20.343255Z  INFO tswn::cacluate: | 1|Id:       61337091|594521.21/s 513.666E/d 10.15⬇️|0  |预计:0:2:37|
2024-04-28T13:20:30.919826Z  INFO tswn::cacluate: | 1|Id:      120789191|562124.60/s 485.676E/d 10.58⬇️|0  |预计:0:2:36|
2024-04-28T13:20:40.691081Z  INFO tswn::cacluate: | 1|Id:      177001591|575293.72/s 497.054E/d  9.77⬆️|0  |预计:0:2:23|
2024-04-28T13:20:50.443663Z  INFO tswn::cacluate: | 1|Id:      234530891|589903.04/s 509.676E/d  9.75⬆️|0  |预计:0:2:9|
2024-04-28T13:21:00.479388Z  INFO tswn::cacluate: | 1|Id:      293521191|587813.53/s 507.871E/d 10.04⬇️|0  |预计:0:2:0|
2024-04-28T13:21:10.520597Z  INFO tswn::cacluate: | 1|Id:      352302491|585411.71/s 505.796E/d 10.04⬇️|0  |预计:0:1:50|
2024-04-28T13:21:20.511369Z  INFO tswn::cacluate: | 1|Id:      410843591|585959.76/s 506.269E/d  9.99⬆️|0  |预计:0:1:40|
2024-04-28T13:21:30.775552Z  INFO tswn::cacluate: | 1|Id:      469439491|570886.38/s 493.246E/d 10.26⬇️|0  |预计:0:1:32|
2024-04-28T13:21:40.943627Z  INFO tswn::cacluate: | 1|Id:      526528091|561459.41/s 485.101E/d 10.17⬇️|0  |预计:0:1:24|
2024-04-28T13:21:50.682140Z  INFO tswn::cacluate: | 1|Id:      582673991|576545.70/s 498.135E/d  9.74⬆️|0  |预计:0:1:12|
2024-04-28T13:22:00.546955Z  INFO tswn::cacluate: | 1|Id:      640328491|584459.75/s 504.973E/d  9.86⬆️|0  |预计:0:1:1|
2024-04-28T13:22:10.665629Z  INFO tswn::cacluate: | 1|Id:      698774391|577612.11/s 499.057E/d 10.12⬇️|0  |预计:0:0:52|
2024-04-28T13:22:20.758077Z  INFO tswn::cacluate: | 1|Id:      756535591|572334.07/s 494.497E/d 10.09⬇️|0  |预计:0:0:42|
2024-04-28T13:22:31.143197Z  INFO tswn::cacluate: | 1|Id:      813768991|551118.64/s 476.167E/d 10.38⬇️|0  |预计:0:0:33|
2024-04-28T13:22:40.422587Z  INFO tswn::cacluate: | 1|Id:      868880791|593931.90/s 513.157E/d  9.28⬆️|0  |预计:0:0:22|
2024-04-28T13:22:50.487884Z  INFO tswn::cacluate: | 1|Id:      928273891|590089.41/s 509.837E/d 10.07⬇️|0  |预计:0:0:12|
2024-04-28T13:23:00.533788Z  INFO tswn::cacluate: | 1|Id:      987282791|587401.82/s 507.515E/d 10.05⬇️|0  |预计:0:0:2|
```

## 正常 0.1.10

```text
❯ .\runs\tswn-0110.exe --team shenjacka -q 5500 --end 1000000000 --bench
2024-04-28T12:57:44.582513Z  INFO tswn: 输出文件: "./namerena/namerena-shenjacka-2024-04-28_20-57-44.csv"
2024-04-28T12:57:44.582769Z  INFO tswn: 开始: 0 结尾: 1000000000
2024-04-28T12:57:44.582874Z  INFO tswn: 线程数: 10
2024-04-28T12:57:44.582964Z  INFO tswn: 八围预期: 640
2024-04-28T12:57:44.583057Z  INFO tswn: 队伍名: shenjacka
2024-04-28T12:57:44.583143Z  INFO tswn: 输出文件名: "./namerena/namerena-shenjacka-2024-04-28_20-57-44.csv"
2024-04-28T12:57:44.583237Z  INFO tswn: 预期状态输出时间间隔: 10 秒
2024-04-28T12:57:44.583324Z  INFO tswn: 是否启动 benchmark 模式: true
2024-04-28T12:57:44.583409Z  INFO tswn: 开始 benchmark
2024-04-28T12:57:44.583531Z  INFO tswn: 设置进程亲和性成功 1
2024-04-28T12:57:44.583705Z  INFO tswn: 设置线程亲和性成功 1
2024-04-28T12:57:44.770450Z  INFO tswn::cacluate: | 1|Id:         999991|535481.54/s 462.656E/d  0.19⬆️|0  |预计:0:3:6|
2024-04-28T12:57:55.025801Z  INFO tswn::cacluate: | 1|Id:       54548091|522159.58/s 451.146E/d 10.26⬇️|0  |预计:0:3:1|
2024-04-28T12:58:05.311264Z  INFO tswn::cacluate: | 1|Id:      106763991|507680.19/s 438.636E/d 10.29⬇️|0  |预计:0:2:55|
2024-04-28T12:58:15.145313Z  INFO tswn::cacluate: | 1|Id:      157531991|516262.88/s 446.051E/d  9.83⬆️|0  |预计:0:2:43|
2024-04-28T12:58:25.054223Z  INFO tswn::cacluate: | 1|Id:      209158191|521019.78/s 450.161E/d  9.91⬆️|0  |预计:0:2:31|
2024-04-28T12:58:35.103209Z  INFO tswn::cacluate: | 1|Id:      261260091|518491.51/s 447.977E/d 10.05⬇️|0  |预计:0:2:22|
2024-04-28T12:58:44.987401Z  INFO tswn::cacluate: | 1|Id:      313109191|524580.20/s 453.237E/d  9.88⬆️|0  |预计:0:2:10|
2024-04-28T12:58:55.053296Z  INFO tswn::cacluate: | 1|Id:      365567191|521155.18/s 450.278E/d 10.07⬇️|0  |预计:0:2:1|
2024-04-28T12:59:05.073971Z  INFO tswn::cacluate: | 1|Id:      417682691|520090.85/s 449.358E/d 10.02⬇️|0  |预计:0:1:51|
2024-04-28T12:59:15.031897Z  INFO tswn::cacluate: | 1|Id:      469691691|522300.46/s 451.268E/d  9.96⬆️|0  |预计:0:1:41|
2024-04-28T12:59:25.049878Z  INFO tswn::cacluate: | 1|Id:      521921691|521375.29/s 450.468E/d 10.02⬇️|0  |预计:0:1:31|
2024-04-28T12:59:35.034084Z  INFO tswn::cacluate: | 1|Id:      574059191|522212.29/s 451.191E/d  9.98⬆️|0  |预计:0:1:21|
2024-04-28T12:59:45.023445Z  INFO tswn::cacluate: | 1|Id:      626280391|522782.69/s 451.684E/d  9.99⬆️|0  |预计:0:1:11|
2024-04-28T12:59:55.137604Z  INFO tswn::cacluate: | 1|Id:      678558591|516893.69/s 446.596E/d 10.11⬇️|0  |预计:0:1:2|
2024-04-28T13:00:05.051609Z  INFO tswn::cacluate: | 1|Id:      730247891|521388.91/s 450.480E/d  9.91⬆️|0  |预计:0:0:51|
2024-04-28T13:00:15.045721Z  INFO tswn::cacluate: | 1|Id:      782386691|521707.33/s 450.755E/d  9.99⬆️|0  |预计:0:0:41|
2024-04-28T13:00:25.023265Z  INFO tswn::cacluate: | 1|Id:      834557391|522893.05/s 451.780E/d  9.98⬆️|0  |预计:0:0:31|
2024-04-28T13:00:35.036379Z  INFO tswn::cacluate: | 1|Id:      886846691|522217.54/s 451.196E/d 10.01⬇️|0  |预计:0:0:21|
2024-04-28T13:00:45.065655Z  INFO tswn::cacluate: | 1|Id:      939068391|520704.57/s 449.889E/d 10.03⬇️|0  |预计:0:0:11|
2024-04-28T13:00:55.104846Z  INFO tswn::cacluate: | 1|Id:      991138791|518683.00/s 448.142E/d 10.04⬇️|0  |预计:0:0:1|
```