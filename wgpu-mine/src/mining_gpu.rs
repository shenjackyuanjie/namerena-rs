use crate::core::{self, Evaluation, Thresholds};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const RECORD_WORDS: usize = 44;
const RECORD_BYTES: usize = RECORD_WORDS * 4;

pub struct GpuRecord {
    pub index: usize,
    pub value: Evaluation,
    pub approximate: [f32; 2],
}

pub struct GpuBatch {
    pub records: Vec<GpuRecord>,
    pub prop_pass: u32,
    pub scored: u32,
    pub shadows: u32,
    pub kernel_time: Option<Duration>,
    pub wall_time: Duration,
    pub max_error: [f64; 2],
    pub refined: usize,
}

/// GPU 上执行 RC4、属性、技能、shadow 和 f32 模型，只回传可能命中的紧凑记录。
pub struct GpuMiner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    metadata_buffer: wgpu::Buffer,
    bytes_buffer: wgpu::Buffer,
    params: wgpu::Buffer,
    output: wgpu::Buffer,
    header_staging: wgpu::Buffer,
    record_staging: wgpu::Buffer,
    query: Option<wgpu::QuerySet>,
    resolve: Option<wgpu::Buffer>,
    capacity: usize,
    metadata: Vec<[u32; 2]>,
    bytes: Vec<u8>,
    pub adapter_description: String,
    /// 全部候选使用 f64 重算；默认仅对距非零阈值 10 分内的候选重算。
    pub exact_scores: bool,
}

impl GpuMiner {
    pub async fn new(team: &str, capacity: usize) -> Result<Self, String> {
        if team.len() > 256 || capacity == 0 {
            return Err("队伍最多 256 个 UTF-8 字节，批大小必须大于零".to_owned());
        }
        let input_bytes = capacity.checked_mul(256).ok_or("批大小溢出")? as u64;
        let output_bytes = capacity
            .checked_mul(RECORD_BYTES)
            .and_then(|n| n.checked_add(16))
            .ok_or("输出缓冲长度溢出")? as u64;
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("获取 GPU 失败：{e}"))?;
        let info = adapter.get_info();
        let adapter_description = format!("{} / {:?} / {:?}", info.name, info.backend, info.device_type);
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_compute_workgroup_storage_size = 32768;
        if !limits.check_limits(&adapter.limits())
            || input_bytes > limits.max_storage_buffer_binding_size
            || output_bytes > limits.max_storage_buffer_binding_size
            || (capacity as u64).div_ceil(32) > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err("设备需支持 32KiB 工作组共享内存，且批大小不能超过缓冲/工作组限制".to_owned());
        }
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("E2E 测号设备"),
                required_features: if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("创建 GPU 设备失败：{e}"))?;
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("完整测号内核"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../mining.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RC4 属性 技能 shadow XP/XD"),
            layout: None,
            module: &module,
            entry_point: Some("mine"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建测号管线失败：{error}"));
        }
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let create = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let state = crate::cpu::team_state(team).map(u32::from);
        let team_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("缓存队伍状态"),
            contents: bytemuck::cast_slice(&state),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let metadata_buffer = create(
            "候选偏移与长度",
            (capacity * 8) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let bytes_buffer = create(
            "候选名字字节",
            input_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let output = create(
            "候选与统计",
            output_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let params = create("筛选参数", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let weights = core::model_weights();
        let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("同源 XP XD 模型"),
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffers = [&team_buffer, &metadata_buffer, &bytes_buffer, &output, &params, &weight_buffer];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("完整测号资源"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let header_staging = create(
            "计数与时间戳回读",
            32,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let record_staging = create(
            "仅命中候选回读",
            output_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let query = timestamps.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("完整内核计时"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        let resolve = timestamps.then(|| {
            create(
                "时间戳解析",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            )
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建测号缓冲失败：{error}"));
        }
        Ok(Self {
            device,
            queue,
            pipeline,
            group,
            metadata_buffer,
            bytes_buffer,
            params,
            output,
            header_staging,
            record_staging,
            query,
            resolve,
            capacity,
            metadata: Vec::with_capacity(capacity),
            bytes: Vec::new(),
            adapter_description,
            exact_scores: false,
        })
    }

    pub fn run(&mut self, names: &[String], thresholds: Thresholds, capture_all: bool) -> Result<GpuBatch, String> {
        let start = Instant::now();
        if names.len() > self.capacity {
            return Err("候选数超过 GPU 批容量".to_owned());
        }
        self.metadata.clear();
        self.bytes.clear();
        for (i, name) in names.iter().enumerate() {
            if name.len() > 256 {
                return Err(format!("名字 {i} 超过 256 个 UTF-8 字节"));
            }
            self.metadata.push([self.bytes.len() as u32, name.len() as u32]);
            self.bytes.extend_from_slice(name.as_bytes());
        }
        if names.is_empty() {
            return Ok(GpuBatch {
                records: vec![],
                prop_pass: 0,
                scored: 0,
                shadows: 0,
                kernel_time: None,
                wall_time: start.elapsed(),
                max_error: [0.0; 2],
                refined: 0,
            });
        }
        self.bytes.resize(self.bytes.len().max(1).div_ceil(4) * 4, 0);
        self.queue.write_buffer(&self.metadata_buffer, 0, bytemuck::cast_slice(&self.metadata));
        self.queue.write_buffer(&self.bytes_buffer, 0, &self.bytes);
        self.queue.write_buffer(&self.output, 0, bytemuck::cast_slice(&[0_u32; 4]));
        // 八围物理上界远低于 10000，钳制仅避免 shader 中 3*p 的 u32 溢出，不改变判定。
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[
                names.len() as u32,
                thresholds.prop.min(10000),
                thresholds.xp,
                thresholds.xd,
                u32::from(capture_all),
                0,
                0,
                0,
            ]),
        );
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("完整批次测号"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU 完整计算与压缩"),
                timestamp_writes: self.query.as_ref().map(|q| wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups((names.len() as u32).div_ceil(32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.header_staging, 0, 16);
        if let (Some(query), Some(resolve)) = (&self.query, &self.resolve) {
            encoder.resolve_query_set(query, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, &self.header_staging, 16, 16);
        }
        self.queue.submit(Some(encoder.finish()));
        let header = map_read(&self.device, &self.header_staging, if self.query.is_some() { 32 } else { 16 })?;
        let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        let count = word(0) as usize;
        if count > names.len() || word(1) as usize > names.len() || word(2) as usize > names.len() {
            return Err("GPU 候选计数超出输入范围".to_owned());
        }
        let kernel_time = self.query.as_ref().map(|_| {
            let begin = u64::from_le_bytes(header[16..24].try_into().unwrap());
            let end = u64::from_le_bytes(header[24..32].try_into().unwrap());
            Duration::from_secs_f64(end.wrapping_sub(begin) as f64 * f64::from(self.queue.get_timestamp_period()) / 1e9)
        });
        let mut records = Vec::with_capacity(count);
        let mut max_error = [0.0_f64; 2];
        let mut refined = 0;
        if count != 0 {
            let bytes = (count * RECORD_BYTES) as u64;
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("紧凑候选回读"),
            });
            encoder.copy_buffer_to_buffer(&self.output, 16, &self.record_staging, 0, bytes);
            self.queue.submit(Some(encoder.finish()));
            let data = map_read(&self.device, &self.record_staging, bytes)?;
            for record in data.as_chunks::<RECORD_BYTES>().0 {
                let mut words = [0_u32; RECORD_WORDS];
                for (word, bytes) in words.iter_mut().zip(record.as_chunks::<4>().0) {
                    *word = u32::from_le_bytes(*bytes);
                }
                let index = words[0] as usize;
                if index >= names.len() {
                    return Err("GPU 返回了无效候选下标".to_owned());
                }
                let props: [u32; 8] = words[1..9].try_into().unwrap();
                let skill_ids: [u32; 16] = words[9..25].try_into().unwrap();
                let skill_freqs: [u32; 16] = words[25..41].try_into().unwrap();
                if skill_ids.iter().any(|&id| id >= 40) {
                    return Err("GPU 返回无效技能编号".to_owned());
                }
                let shadow_numerator = words[41] as i32;
                let approximate = [f32::from_bits(words[42]), f32::from_bits(words[43])];
                if !approximate.iter().all(|x| x.is_finite()) {
                    return Err("GPU 返回非有限评分".to_owned());
                }
                let near_boundary = (thresholds.xp != 0 && (f64::from(approximate[0]) - f64::from(thresholds.xp)).abs() <= 10.0)
                    || (thresholds.xd != 0 && (f64::from(approximate[1]) - f64::from(thresholds.xd)).abs() <= 10.0);
                let exact = if capture_all || self.exact_scores || near_boundary {
                    let exact = core::scores(&props, &skill_ids, &skill_freqs, shadow_numerator);
                    refined += 1;
                    for i in 0..2 {
                        let error = (f64::from(approximate[i]) - exact[i]).abs();
                        max_error[i] = max_error[i].max(error);
                        if error >= 10.0 {
                            return Err(format!("GPU 评分误差超过约定：候选 {index}，模型 {i}，绝对误差 {error}"));
                        }
                    }
                    exact
                } else {
                    approximate.map(f64::from)
                };
                let value = Evaluation {
                    props,
                    skill_ids,
                    skill_freqs,
                    shadow_numerator,
                    xp: exact[0],
                    xd: exact[1],
                };
                if capture_all || value.matches(thresholds) {
                    records.push(GpuRecord {
                        index,
                        value,
                        approximate,
                    });
                }
            }
            records.sort_unstable_by_key(|record| record.index);
            if records.windows(2).any(|pair| pair[0].index == pair[1].index) {
                return Err("GPU 返回重复候选".to_owned());
            }
        }
        Ok(GpuBatch {
            records,
            prop_pass: word(1),
            scored: word(2),
            shadows: word(3),
            kernel_time,
            wall_time: start.elapsed(),
            max_error,
            refined,
        })
    }
}

fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>, String> {
    let slice = buffer.slice(..bytes);
    let (sender, receiver) = flume::bounded(1);
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(120)),
        })
        .map_err(|e| format!("等待 GPU 测号失败：{e}"))?;
    receiver
        .recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("测号回读映射失败：{e}"))?;
    let view = slice.get_mapped_range().map_err(|e| e.to_string())?;
    let result = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| match i % 8 {
                0 => String::new(),
                1 => format!("中文😀名字{i}"),
                2 => format!("name-{i}"),
                3 => "a".repeat(256),
                4 => format!("{}a", "中".repeat(85)),
                5 => format!("raw@name-{i}"),
                6 => format!("一一七啺埀㴁-{i}"),
                _ => format!("完整金标准-{i}"),
            })
            .collect()
    }

    fn equal(actual: &Evaluation, expected: &Evaluation) {
        assert_eq!(actual.props, expected.props);
        assert_eq!(actual.skill_ids, expected.skill_ids);
        assert_eq!(actual.skill_freqs, expected.skill_freqs);
        assert_eq!(actual.shadow_numerator, expected.shadow_numerator);
        assert!((actual.xp - expected.xp).abs() < 1e-8);
        assert!((actual.xd - expected.xd).abs() < 1e-8);
    }

    #[test]
    #[ignore = "需要可用 GPU：完整属性、技能、shadow、评分和筛选对拍"]
    fn gpu_full_pipeline_matches_reference() {
        pollster::block_on(async {
            for team in ["", "shenjack", "中文队伍"] {
                let mut gpu = GpuMiner::new(team, 513).await.unwrap();
                for count in [0, 1, 31, 32, 33, 255, 256, 257, 513, 1] {
                    let names = samples(count);
                    let actual = gpu.run(&names, Thresholds::default(), true).unwrap();
                    assert_eq!(actual.records.len(), count);
                    assert_eq!(actual.scored as usize, count);
                    for record in actual.records {
                        equal(&record.value, &core::evaluate(team, &names[record.index]));
                    }
                }
                let names = samples(257);
                let expected: Vec<_> = names.iter().map(|name| core::evaluate(team, name)).collect();
                let pivot = &expected[2];
                for thresholds in [
                    Thresholds::default(),
                    Thresholds {
                        prop: 550,
                        xp: 4000,
                        xd: 5000,
                    },
                    Thresholds { prop: 577, xp: 0, xd: 0 },
                    Thresholds {
                        prop: u32::MAX,
                        xp: u32::MAX,
                        xd: u32::MAX,
                    },
                    Thresholds {
                        prop: 0,
                        xp: pivot.xp as u32,
                        xd: pivot.xd as u32,
                    },
                    Thresholds {
                        prop: 0,
                        xp: (pivot.xp as u32).saturating_add(1),
                        xd: 0,
                    },
                ] {
                    let actual = gpu.run(&names, thresholds, false).unwrap();
                    let wanted: Vec<_> = expected
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| value.matches(thresholds))
                        .map(|(i, _)| i)
                        .collect();
                    assert_eq!(
                        actual.records.iter().map(|record| record.index).collect::<Vec<_>>(),
                        wanted,
                        "team={team:?}, thresholds={thresholds:?}"
                    );
                }
                assert!(gpu.run(&["中".repeat(86)], Thresholds::default(), true).is_err());
                assert!(gpu.run(&vec![String::new(); 514], Thresholds::default(), true).is_err());
                assert_eq!(
                    gpu.run(&["x".to_owned()], Thresholds::default(), true).unwrap().records.len(),
                    1
                );
            }
        });
    }
}
