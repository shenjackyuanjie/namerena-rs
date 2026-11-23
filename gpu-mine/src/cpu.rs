//! 仅执行 RC4 KSA 的安全、多线程 CPU 对照基线，不包含属性或评分逻辑。
//!
//! 公平基准应将构造、队伍状态计算和线程启动排除在计时外。每次 run 的计时
//! 包含任务派发、结果返回、返回 Vec 的分配以及按输入顺序拼接完整输出。
//! worker 的结果容量会跨 run 复用；首次运行或扩大批次可能增长容量，基准应披露
//! 是否预热。即使只使用一个线程，也通过相同的持久 worker 和通道执行。

use std::ops::Range;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

pub const STATE_SIZE: usize = 256;

/// 从恒等 Sbox 开始，执行一轮密钥为 [0] + team UTF-8 字节的 KSA。
/// 输入按字节处理，保留内嵌 NUL；不施加名字或队伍长度限制。
pub fn team_state(team: &str) -> [u8; STATE_SIZE] {
    let mut state = std::array::from_fn(|index| index as u8);
    ksa_round(&mut state, team.as_bytes());
    state
}

/// 独立朴素金标准：队伍一轮、名字两轮，每轮重新令 j = 0。
/// 刻意使用显式密钥分配和取模，不调用优化路径或 team_state。
pub fn reference(team: &str, name: &str) -> [u8; STATE_SIZE] {
    let mut state = [0_u8; STATE_SIZE];
    for (index, value) in state.iter_mut().enumerate() {
        *value = index as u8;
    }
    for text in [team, name, name] {
        let key: Vec<u8> = std::iter::once(0).chain(text.bytes()).collect();
        let mut j = 0_usize;
        for i in 0..STATE_SIZE {
            j = (j + usize::from(state[i]) + usize::from(key[i % key.len()])) % STATE_SIZE;
            state.swap(i, j);
        }
    }
    state
}

/// 隐式前导零密钥：循环键下标，热路径不分配密钥 Vec，也不做取模。
#[inline]
fn ksa_round(state: &mut [u8; STATE_SIZE], bytes: &[u8]) {
    let mut j = 0_u8;
    let mut key_index = 0;
    for i in 0..STATE_SIZE {
        let key_byte = if key_index == 0 { 0 } else { bytes[key_index - 1] };
        j = j.wrapping_add(state[i]).wrapping_add(key_byte);
        state.swap(i, usize::from(j));
        key_index = if key_index == bytes.len() { 0 } else { key_index + 1 };
    }
}

struct Task {
    names: Arc<[String]>,
    range: Range<usize>,
    output: Vec<u8>,
}

struct Worker {
    tasks: Option<mpsc::Sender<Task>>,
    results: mpsc::Receiver<Vec<u8>>,
    handle: Option<JoinHandle<()>>,
    // 结果消费后留在主线程；下一轮随任务送回 worker，复用其容量。
    output: Vec<u8>,
}

/// 持久工作线程池。每个 worker 独占状态数组和结果缓冲，无 unsafe 共享写入。
pub struct CpuRc4 {
    workers: Vec<Worker>,
    failed: bool,
}

impl CpuRc4 {
    /// 缓存一次队伍状态，创建指定数量的持久线程。threads 必须大于零。
    /// 部分线程创建失败时，已创建的线程同样会关闭通道并 join。
    pub fn new(team: &str, threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("CPU RC4 线程数必须大于零".to_owned());
        }
        let cached_team = team_state(team);
        let mut pool = Self {
            workers: Vec::new(),
            failed: false,
        };
        pool.workers
            .try_reserve_exact(threads)
            .map_err(|error| format!("分配 CPU RC4 worker 列表失败：{error}"))?;
        for index in 0..threads {
            let (task_tx, task_rx) = mpsc::channel::<Task>();
            let (result_tx, result_rx) = mpsc::channel::<Vec<u8>>();
            let handle = thread::Builder::new()
                .name(format!("rc4-cpu-{index}"))
                .spawn(move || worker_loop(cached_team, task_rx, result_tx))
                .map_err(|error| format!("创建 CPU RC4 worker {index} 失败：{error}"))?;
            pool.workers.push(Worker {
                tasks: Some(task_tx),
                results: result_rx,
                handle: Some(handle),
                output: Vec::new(),
            });
        }
        Ok(pool)
    }

    /// 连续范围均分任务，按名字原始顺序返回每人完整的 256 字节状态。
    /// 空输入直接返回空 Vec；非空输入始终通过持久 worker 执行。
    pub fn run(&mut self, names: Arc<[String]>) -> Result<Vec<u8>, String> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        if self.failed {
            return Err("CPU RC4 工作线程已失效，请重新创建 CpuRc4".to_owned());
        }
        let output_len = names.len().checked_mul(STATE_SIZE).ok_or_else(|| "CPU RC4 输出长度溢出".to_owned())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|error| format!("分配 CPU RC4 返回缓冲失败：{error}"))?;

        let active = self.workers.len().min(names.len());
        let base = names.len() / active;
        let extra = names.len() % active;
        // 在派发前完成可报告失败的容量增长，避免部分派发后因分配失败留下旧结果。
        for (index, worker) in self.workers.iter_mut().take(active).enumerate() {
            let count = base + usize::from(index < extra);
            worker.output.clear();
            worker
                .output
                .try_reserve_exact(count * STATE_SIZE)
                .map_err(|error| format!("分配 CPU RC4 worker {index} 结果缓冲失败：{error}"))?;
        }

        let mut start = 0;
        for (index, worker) in self.workers.iter_mut().take(active).enumerate() {
            let end = start + base + usize::from(index < extra);
            let task = Task {
                names: Arc::clone(&names),
                range: start..end,
                output: std::mem::take(&mut worker.output),
            };
            let Some(sender) = worker.tasks.as_ref() else {
                self.failed = true;
                return Err(format!("CPU RC4 worker {index} 任务通道已关闭"));
            };
            if sender.send(task).is_err() {
                self.failed = true;
                return Err(format!("CPU RC4 worker {index} 已停止，任务派发失败"));
            }
            start = end;
        }

        // worker 各自处理连续范围；按相同顺序消费，不受实际完成顺序影响。
        for (index, worker) in self.workers.iter_mut().take(active).enumerate() {
            let buffer = match worker.results.recv() {
                Ok(buffer) => buffer,
                Err(error) => {
                    // 后续拒绝新任务，防止异常后把未消费的旧结果当成下一轮结果。
                    self.failed = true;
                    return Err(format!("接收 CPU RC4 worker {index} 结果失败：{error}"));
                }
            };
            output.extend_from_slice(&buffer);
            worker.output = buffer;
        }
        Ok(output)
    }
}

fn worker_loop(cached_team: [u8; STATE_SIZE], tasks: mpsc::Receiver<Task>, results: mpsc::Sender<Vec<u8>>) {
    let mut state = [0_u8; STATE_SIZE];
    while let Ok(Task {
        names,
        range,
        mut output,
    }) = tasks.recv()
    {
        output.clear();
        for name in &names[range] {
            state.copy_from_slice(&cached_team);
            let bytes = name.as_bytes();
            ksa_round(&mut state, bytes);
            ksa_round(&mut state, bytes);
            output.extend_from_slice(&state);
        }
        if results.send(output).is_err() {
            break;
        }
    }
}

impl Drop for CpuRc4 {
    fn drop(&mut self) {
        // 先关闭全部任务通道，再等待线程；无界结果通道不会阻塞 worker 退出。
        for worker in &mut self.workers {
            drop(worker.tasks.take());
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_names(count: usize) -> Arc<[String]> {
        (0..count)
            .map(|index| match index % 10 {
                0 => String::new(),
                1 => format!("名字{index}中文"),
                2 => format!("name-{index}"),
                3 => format!("含\0字节{index}"),
                4 => "x".repeat(256),
                5 => format!("{}x", "中".repeat(85)), // 恰好 256 个 UTF-8 字节。
                6 => "长名字".repeat(400),
                7 => "\0".to_owned(),
                8 => "a".repeat(255),
                _ => format!("{}-{index}", "z".repeat(257)),
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn assert_matches_reference(team: &str, names: &Arc<[String]>, actual: &[u8]) {
        assert_eq!(actual.len(), names.len() * STATE_SIZE);
        for (index, (name, state)) in names.iter().zip(actual.as_chunks::<STATE_SIZE>().0).enumerate() {
            assert_eq!(*state, reference(team, name), "队伍 {team:?}，名字索引 {index}：{name:?}");
            let mut sorted = state.to_vec();
            sorted.sort_unstable();
            assert!(sorted.iter().enumerate().all(|(i, &value)| value == i as u8));
        }
    }

    #[test]
    fn team_state_matches_independent_single_round() {
        for team in [
            String::new(),
            "中文队伍".to_owned(),
            "含\0队伍".to_owned(),
            "x".repeat(256),
            "队".repeat(400),
        ] {
            let key: Vec<u8> = std::iter::once(0).chain(team.bytes()).collect();
            let mut expected = std::array::from_fn::<_, STATE_SIZE, _>(|i| i as u8);
            let mut j = 0_usize;
            for i in 0..STATE_SIZE {
                j = (j + usize::from(expected[i]) + usize::from(key[i % key.len()])) % STATE_SIZE;
                expected.swap(i, j);
            }
            assert_eq!(team_state(&team), expected, "队伍 {team:?}");
        }
    }

    #[test]
    fn edge_case_names_match_reference_for_various_teams() {
        let names = sample_names(20);
        for team in [
            String::new(),
            "中文队伍".to_owned(),
            "含\0队伍".to_owned(),
            "x".repeat(256),
            "长队伍".repeat(400),
        ] {
            let mut cpu = CpuRc4::new(&team, 3).unwrap();
            let actual = cpu.run(Arc::clone(&names)).unwrap();
            assert_matches_reference(&team, &names, &actual);
        }
    }

    #[test]
    fn batch_tails_and_thread_counts_preserve_order_and_results() {
        let team = "并行对拍队伍";
        let mut pools: Vec<_> = [1, 2, 7].into_iter().map(|threads| CpuRc4::new(team, threads).unwrap()).collect();
        for count in [0, 1, 2, 6, 7, 8, 255, 256, 257, 511, 512, 513] {
            let names = sample_names(count);
            let expected = pools[0].run(Arc::clone(&names)).unwrap();
            assert_matches_reference(team, &names, &expected);
            for pool in &mut pools[1..] {
                assert_eq!(pool.run(Arc::clone(&names)).unwrap(), expected, "名字数量 {count}");
            }
        }
    }

    #[test]
    fn repeated_calls_reuse_threads_and_worker_capacity() {
        let team = "持久线程队伍";
        let mut cpu = CpuRc4::new(team, 4).unwrap();
        let thread_ids: Vec<_> = cpu.workers.iter().map(|worker| worker.handle.as_ref().unwrap().thread().id()).collect();
        let large = sample_names(513);
        let first = cpu.run(Arc::clone(&large)).unwrap();
        assert_matches_reference(team, &large, &first);
        let buffers: Vec<_> = cpu
            .workers
            .iter()
            .map(|worker| (worker.output.as_ptr(), worker.output.capacity()))
            .collect();
        for count in [0, 1, 3, 257, 2, 513] {
            let names = sample_names(count);
            let actual = cpu.run(Arc::clone(&names)).unwrap();
            assert_matches_reference(team, &names, &actual);
            if count == 513 {
                assert_eq!(actual, first);
            }
            for (index, worker) in cpu.workers.iter().enumerate() {
                assert_eq!(worker.handle.as_ref().unwrap().thread().id(), thread_ids[index]);
                assert_eq!((worker.output.as_ptr(), worker.output.capacity()), buffers[index]);
            }
        }
    }

    #[test]
    fn zero_threads_and_impossible_thread_capacity_are_rejected() {
        assert!(CpuRc4::new("队伍", 0).is_err());
        assert!(CpuRc4::new("队伍", usize::MAX).is_err());
    }
}
