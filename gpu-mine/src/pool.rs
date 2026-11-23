use crate::{
    core::{self, Evaluation, Thresholds},
    input::Generator,
};
use std::{
    ops::Range,
    sync::{mpsc, Arc},
    thread::{self, JoinHandle},
};

pub struct CpuRecord {
    pub index: usize,
    pub value: Evaluation,
}
pub struct CpuBatch {
    pub records: Vec<CpuRecord>,
    pub prop_pass: u64,
    pub shadows: u64,
}

enum Task {
    Generate {
        start: u64,
        count: usize,
    },
    Evaluate {
        names: Arc<[String]>,
        range: Range<usize>,
        thresholds: Thresholds,
        capture_all: bool,
    },
}
enum Reply {
    Names(Vec<String>),
    Evaluated(CpuBatch),
}
struct Worker {
    send: Option<mpsc::Sender<Task>>,
    recv: mpsc::Receiver<Result<Reply, String>>,
    handle: Option<JoinHandle<()>>,
}

/// 生成和完整 CPU 对照共用的持久 worker；不使用旧矿工的可变别名或任务归属广播。
pub struct CpuPool {
    workers: Vec<Worker>,
    failed: bool,
}

impl CpuPool {
    pub fn new(generator: Generator, team: &str, threads: usize) -> Result<Self, String> {
        if threads == 0 || threads > 256 || team.len() > 256 {
            return Err("线程数须为 1..=256，队伍名最多 256 个 UTF-8 字节".to_owned());
        }
        let team_state = crate::cpu::team_state(team);
        let mut result = Self {
            workers: Vec::with_capacity(threads),
            failed: false,
        };
        for index in 0..threads {
            let (send, tasks) = mpsc::channel();
            let (results, recv) = mpsc::channel();
            let generator = generator.clone();
            let handle = thread::Builder::new()
                .name(format!("mine-worker-{index}"))
                .spawn(move || {
                    while let Ok(task) = tasks.recv() {
                        let result = execute(task, &generator, &team_state);
                        if results.send(result).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| format!("创建 worker 失败：{e}"))?;
            result.workers.push(Worker {
                send: Some(send),
                recv,
                handle: Some(handle),
            });
        }
        Ok(result)
    }

    fn dispatch(&mut self, tasks: Vec<Task>) -> Result<Vec<Reply>, String> {
        if self.failed {
            return Err("工作池已失效".to_owned());
        }
        let count = tasks.len();
        for (worker, task) in self.workers.iter().zip(tasks) {
            if worker.send.as_ref().ok_or("任务通道已关闭")?.send(task).is_err() {
                self.failed = true;
                return Err("任务派发失败".to_owned());
            }
        }
        let mut replies = Vec::with_capacity(count);
        let mut error = None;
        // 全部收完，错误也不会残留旧批次的回复。
        for worker in &self.workers[..count] {
            match worker.recv.recv() {
                Ok(Ok(reply)) => replies.push(reply),
                Ok(Err(e)) => {
                    error.get_or_insert(e);
                }
                Err(e) => {
                    self.failed = true;
                    error.get_or_insert_with(|| e.to_string());
                }
            }
        }
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(replies)
        }
    }

    pub fn generate(&mut self, range: Range<u64>) -> Result<Arc<[String]>, String> {
        if range.start > range.end {
            return Err("起始 ID 大于结束 ID".to_owned());
        }
        let count = usize::try_from(range.end - range.start).map_err(|_| "批次太大")?;
        if count == 0 {
            return Ok(Vec::<String>::new().into());
        }
        let tasks = partitions(count, self.workers.len())
            .into_iter()
            .map(|part| Task::Generate {
                start: range.start + part.start as u64,
                count: part.len(),
            })
            .collect();
        let mut names = Vec::with_capacity(count);
        for reply in self.dispatch(tasks)? {
            if let Reply::Names(part) = reply {
                names.extend(part);
            } else {
                return Err("工作池回复类型不匹配".to_owned());
            }
        }
        Ok(names.into())
    }

    pub fn evaluate(&mut self, names: Arc<[String]>, thresholds: Thresholds, capture_all: bool) -> Result<CpuBatch, String> {
        if names.is_empty() {
            return Ok(CpuBatch {
                records: vec![],
                prop_pass: 0,
                shadows: 0,
            });
        }
        let tasks = partitions(names.len(), self.workers.len())
            .into_iter()
            .map(|range| Task::Evaluate {
                names: Arc::clone(&names),
                range,
                thresholds,
                capture_all,
            })
            .collect();
        let mut result = CpuBatch {
            records: vec![],
            prop_pass: 0,
            shadows: 0,
        };
        for reply in self.dispatch(tasks)? {
            if let Reply::Evaluated(part) = reply {
                result.records.extend(part.records);
                result.prop_pass += part.prop_pass;
                result.shadows += part.shadows;
            } else {
                return Err("工作池回复类型不匹配".to_owned());
            }
        }
        Ok(result)
    }
}

fn partitions(count: usize, workers: usize) -> Vec<Range<usize>> {
    let active = count.min(workers);
    if active == 0 {
        return vec![];
    }
    let mut cursor = 0;
    (0..active)
        .map(|index| {
            let start = cursor;
            cursor += count / active + usize::from(index < count % active);
            start..cursor
        })
        .collect()
}

fn execute(task: Task, generator: &Generator, team_state: &[u8; 256]) -> Result<Reply, String> {
    match task {
        Task::Generate { start, count } => {
            let mut names = Vec::with_capacity(count);
            for offset in 0..count {
                let id = start.checked_add(offset as u64).ok_or("候选编号溢出")?;
                names.push(generator.generate(id)?);
            }
            Ok(Reply::Names(names))
        }
        Task::Evaluate {
            names,
            range,
            thresholds,
            capture_all,
        } => {
            let mut result = CpuBatch {
                records: vec![],
                prop_pass: 0,
                shadows: 0,
            };
            for index in range {
                let name = &names[index];
                let state = core::rc4_name(team_state, name);
                if !capture_all
                    && u64::from(core::property_numerator(&core::properties(&state))) <= u64::from(thresholds.prop) * 3
                {
                    continue;
                }
                result.prop_pass += 1;
                let value = core::evaluate_state(team_state, state, name);
                if value.skill_ids.iter().zip(&value.skill_freqs).any(|(&id, &freq)| id == 24 && freq != 0) {
                    result.shadows += 1;
                }
                if capture_all || value.matches(thresholds) {
                    result.records.push(CpuRecord { index, value });
                }
            }
            Ok(Reply::Evaluated(result))
        }
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            drop(worker.send.take());
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
    use crate::input::Encoding;

    #[test]
    fn generation_and_evaluation_cover_ranges_across_workers() {
        let generator = Generator::new("中文 [] []".to_owned(), Encoding::Number).unwrap();
        for threads in [1, 2, 6] {
            let mut pool = CpuPool::new(generator.clone(), "team", threads).unwrap();
            for count in [0, 1, 31, 32, 33, 257, 513] {
                let names = pool.generate(7..7 + count).unwrap();
                assert_eq!(names.len(), count as usize);
                for (i, name) in names.iter().enumerate() {
                    assert_eq!(name, &generator.generate(7 + i as u64).unwrap());
                }
                let batch = pool.evaluate(Arc::clone(&names), Thresholds::default(), false).unwrap();
                assert_eq!(batch.records.len(), count as usize);
                for (index, record) in batch.records.iter().enumerate() {
                    assert_eq!(record.index, index);
                    assert_eq!(record.value, core::evaluate("team", &names[index]));
                }
            }
        }
    }

    #[test]
    fn generation_errors_are_drained_and_pool_recovers() {
        let generator = Generator::new(format!("{}[]", "x".repeat(255)), Encoding::Number).unwrap();
        let mut pool = CpuPool::new(generator, "team", 2).unwrap();
        assert!(pool.generate(98..102).is_err());
        assert_eq!(pool.generate(0..2).unwrap().len(), 2);
        assert!(pool.generate(Range { start: 10, end: 9 }).is_err());
    }
}
