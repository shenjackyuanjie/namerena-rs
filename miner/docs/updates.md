# tswn 的 更新

## 0.3.x

### 0.3.1

- 去掉了 `--bench` 选项
- 现在可以直接使用 `--core-pick` 选项来选择核心了
- 并且单线程模式下效果和 benchmark 模式下一样了

> 然后我准备改成内置多进程模式

### 0.3.0

把 `--bench-core` 改成了 `--core-pick`
同时也支持在单线程运行的时候跑在指定的核心上了

然后加入了强制的过滤机制

> 就是卡常
反正现在在我的 5800x 上能跑到 700E/d (单线程) 了
> 我预计在后面加一下多线程运行的时候加个总计效率

## 0.2.15~16

一些提前过滤的东西

## 0.1.x

### 0.1.6~9

更新了一大堆核心亲和性相关的东西
