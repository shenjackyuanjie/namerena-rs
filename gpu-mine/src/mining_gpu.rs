use crate::core::{self, Evaluation, Thresholds};
use crate::gpu_input::GpuInput;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const RECORD_WORDS: usize = 44;
const RECORD_BYTES: usize = RECORD_WORDS * 4;
const HEADER_BYTES: u64 = 16;
const INLINE_RECORDS: usize = 4;
const MAX_DISPATCH_X: u32 = 65_535;
/// 显式大批次最多两行，限制分配量。
pub const MAX_GENERATED_BATCH: usize = MAX_DISPATCH_X as usize * 32 * 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SboxLayout {
    Wide,
    Packed,
}

#[derive(Clone, Copy, Debug)]
pub struct GpuOptions {
    pub sbox: SboxLayout,
    pub workgroup_size: u32,
    pub compact_model: bool,
}
impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            sbox: SboxLayout::Packed,
            workgroup_size: 32,
            compact_model: true,
        }
    }
}
impl GpuOptions {
    pub fn shared_bytes(self) -> Result<u32, String> {
        if ![32, 64, 128].contains(&self.workgroup_size) || (self.sbox == SboxLayout::Wide && self.workgroup_size != 32) {
            return Err("工作组须为32/64/128；wide S-box仅支持32，packed支持全部三种".to_owned());
        }
        Ok(self.workgroup_size * if self.sbox == SboxLayout::Packed { 256 } else { 1024 })
    }
    fn shader_source(self, generated: Option<&GpuInput>, tiled: bool) -> Result<String, String> {
        let bytes = self.shared_bytes()?;
        let mut source = include_str!("../mining.wgsl")
            .replace(
                "const LANES: u32 = 32u;",
                &format!("const LANES: u32 = {}u;", self.workgroup_size),
            )
            .replace("const WORDS: u32 = 8192u;", &format!("const WORDS: u32 = {}u;", bytes / 4))
            .replace(
                "const PACKED: bool = false;",
                &format!("const PACKED: bool = {};", self.sbox == SboxLayout::Packed),
            );
        source = source.replace(
            "const COMPACT_MODEL: bool = false;",
            &format!("const COMPACT_MODEL: bool = {};", self.compact_model),
        );
        if tiled { source = source.replace("const TILED: bool = false;", "const TILED: bool = true;"); }
        if let Some(input) = generated {
            source = source.replace("const GENERATED: bool = false;", "const GENERATED: bool = true;");
            let (before, rest) = source.split_once("// KEY_BEGIN").ok_or("着色器缺少KEY_BEGIN")?;
            let (_, after) = rest.split_once("// KEY_END").ok_or("着色器缺少KEY_END")?;
            source = format!("{before}{}\n{after}", include_str!("../generated-key.wgsl"));
            if let Some(cycle) = (self.sbox == SboxLayout::Packed).then(|| cycle_shader(input)).flatten() {
                let (left, tail) = source.split_once("// GENERATED_CYCLE_BEGIN").ok_or("着色器缺少键环起始标记")?;
                let (_, right) = tail.split_once("// GENERATED_CYCLE_END").ok_or("着色器缺少键环结束标记")?;
                source = format!("{left}{cycle}\n{right}");
            }
        }
        Ok(source)
    }
}

// 仅编译短单槽 Base16384 的字面字节；避免为长模板生成过大的 GPU 程序。
// 不内置任何模板，也不对实际 ID 的高位或取值范围作假定。
fn cycle_shader(input: &GpuInput) -> Option<String> {
    if input.encoding != 0 || input.name_len > 64 || input.template_map.len() != input.name_len as usize {
        return None;
    }
    let first = input.prefix_steps.checked_sub(1)? as usize;
    let end = first.checked_add(18)?;
    let map = &input.template_map;
    if !map.get(first..end)?.iter().copied().eq(0x100..0x112)
        || map[..first].iter().chain(&map[end..]).any(|&byte| byte >= 256)
    {
        return None;
    }
    let literal = |(k, byte)| format!("            cursor = segment_step(lane, i + {k}u, cursor, {byte}u);\n");
    let prefix = std::iter::once(0).chain(map[..first].iter().copied()).enumerate()
        .map(literal).collect::<String>();
    let suffix = map[end..].iter().copied().enumerate()
        .map(|(k, byte)| literal((end + 1 + k, byte))).collect::<String>();
    Some(include_str!("../generated-cycle.wgsl")
        .replace("/* KSA_FIRST */", &format!("{}u", first + 1))
        .replace("/* KSA_LEN */", &format!("{}u", input.name_len + 1))
        .replace("/* KSA_PREFIX */", &prefix).replace("/* KSA_SUFFIX */", &suffix))
}

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
    workgroup_size: u32,
    generated: bool,
    metadata: Vec<[u32; 2]>,
    bytes: Vec<u8>,
    pub adapter_description: String,
    /// 全部候选使用 f64 重算；默认仅对距非零阈值 10 分内的候选重算。
    pub exact_scores: bool,
}

impl GpuMiner {
    pub async fn new(team: &str, capacity: usize) -> Result<Self, String> {
        Self::with_options(team, capacity, GpuOptions::default()).await
    }

    pub async fn with_options(team: &str, capacity: usize, options: GpuOptions) -> Result<Self, String> {
        Self::create(team, capacity, options, None).await
    }

    pub async fn with_generated(team: &str, capacity: usize, options: GpuOptions, input: GpuInput) -> Result<Self, String> {
        Self::create(team, capacity, options, Some(input)).await
    }

    async fn create(team: &str, capacity: usize, options: GpuOptions, generated: Option<GpuInput>) -> Result<Self, String> {
        let shared_bytes = options.shared_bytes()?;
        if team.len() > 256 || capacity == 0 || capacity > MAX_GENERATED_BATCH {
            return Err(format!("队伍最多 256 个 UTF-8 字节，批大小须为 1..={MAX_GENERATED_BATCH}"));
        }
        let input_bytes = if let Some(input) = &generated {
            (input.template_map.len() * 4) as u64
        } else {
            capacity.checked_mul(256).ok_or("批大小溢出")? as u64
        };
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
        limits.max_compute_workgroup_storage_size = shared_bytes;
        // 对显式的大批次只申请真正需要的绑定大小，不抬高默认批次的要求。
        let binding_bytes = input_bytes.max(output_bytes).max((capacity * 8) as u64);
        limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(binding_bytes);
        limits.max_buffer_size = limits.max_buffer_size.max(binding_bytes);
        let tiled = (capacity as u32).div_ceil(options.workgroup_size) > MAX_DISPATCH_X;
        if !limits.check_limits(&adapter.limits()) {
            return Err("设备共享内存或批大小超过缓冲/工作组限制".to_owned());
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
            source: wgpu::ShaderSource::Wgsl(options.shader_source(generated.as_ref(), tiled)?.into()),
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
        let state = generated
            .as_ref()
            .map_or_else(|| crate::cpu::team_state(team).map(u32::from), |input| input.prefix_state);
        // 保持 uniform 的固定大小；packed 模式只读取开头 64 个打包字。
        let state = if options.sbox == SboxLayout::Packed {
            let mut compressed = [0; 256];
            for (out, bytes) in compressed.iter_mut().zip(state.as_chunks::<4>().0) {
                *out = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
            }
            compressed
        } else {
            state
        };
        let team_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("缓存队伍状态"),
            contents: bytemuck::cast_slice(&state),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let metadata_buffer = create(
            "候选偏移与长度",
            if generated.is_some() {
                16
            } else {
                (capacity * 8).max(16) as u64
            },
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
            "计数、少量命中与时间戳回读",
            HEADER_BYTES + (capacity.min(INLINE_RECORDS) * RECORD_BYTES) as u64 + 16,
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
        if let Some(input) = &generated {
            queue.write_buffer(
                &metadata_buffer,
                0,
                bytemuck::cast_slice(&[[input.prefix_steps, input.name_len], [input.prefix_j, input.encoding]]),
            );
            queue.write_buffer(&bytes_buffer, 0, bytemuck::cast_slice(&input.template_map));
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
            workgroup_size: options.workgroup_size,
            generated: generated.is_some(),
            metadata: Vec::with_capacity(if generated.is_some() { 0 } else { capacity }),
            bytes: Vec::new(),
            adapter_description,
            exact_scores: false,
        })
    }

    pub fn run(&mut self, names: &[String], thresholds: Thresholds, capture_all: bool) -> Result<GpuBatch, String> {
        let start = Instant::now();
        if self.generated {
            return Err("GPU直接生成管线须调用run_range".to_owned());
        }
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
        if !names.is_empty() {
            self.bytes.resize(self.bytes.len().max(1).div_ceil(4) * 4, 0);
            self.queue.write_buffer(&self.metadata_buffer, 0, bytemuck::cast_slice(&self.metadata));
            self.queue.write_buffer(&self.bytes_buffer, 0, &self.bytes);
        }
        self.execute(names.len(), thresholds, capture_all, 0, start)
    }

    pub fn uses_generated_names(&self) -> bool { self.generated }

    pub fn run_range(
        &mut self,
        first_id: u64,
        count: usize,
        thresholds: Thresholds,
        capture_all: bool,
    ) -> Result<GpuBatch, String> {
        let start = Instant::now();
        if !self.generated {
            return Err("上传管线须调用run".to_owned());
        }
        if count > self.capacity {
            return Err("候选数超过GPU批容量".to_owned());
        }
        if count != 0 {
            first_id.checked_add(count as u64 - 1).ok_or("GPU生成ID溢出")?;
        }
        self.execute(count, thresholds, capture_all, first_id, start)
    }

    fn execute(
        &mut self,
        input_count: usize,
        thresholds: Thresholds,
        capture_all: bool,
        first_id: u64,
        start: Instant,
    ) -> Result<GpuBatch, String> {
        if input_count == 0 {
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
        // 八围物理上界远低于 10000，钳制仅避免 shader 中 3*p 的 u32 溢出，不改变判定。
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[
                input_count as u32,
                thresholds.prop.min(10000),
                thresholds.xp,
                thresholds.xd,
                u32::from(capture_all),
                first_id as u32,
                (first_id >> 32) as u32,
                0,
            ]),
        );
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("完整批次测号"),
        });
        encoder.clear_buffer(&self.output, 0, Some(HEADER_BYTES));
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
            let groups = (input_count as u32).div_ceil(self.workgroup_size);
            let rows = groups.div_ceil(MAX_DISPATCH_X);
            pass.dispatch_workgroups(groups.div_ceil(rows), rows, 1);
        }
        // 低命中批把头与前几条记录一次回读；更多记录完整回退。
        let inline_capacity = self.capacity.min(INLINE_RECORDS);
        let inline_end = HEADER_BYTES + (inline_capacity * RECORD_BYTES) as u64;
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.header_staging, 0, inline_end);
        if let (Some(query), Some(resolve)) = (&self.query, &self.resolve) {
            encoder.resolve_query_set(query, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, &self.header_staging, inline_end, 16);
        }
        self.queue.submit(Some(encoder.finish()));
        let header_len = inline_end + if self.query.is_some() { 16 } else { 0 };
        let header = map_read(&self.device, &self.header_staging, header_len)?;
        let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        let count = word(0) as usize;
        if count > input_count || word(1) as usize > input_count || word(3) > word(1) {
            return Err("GPU 候选计数超出输入范围".to_owned());
        }
        let kernel_time = self.query.as_ref().map(|_| {
            let offset = inline_end as usize;
            let begin = u64::from_le_bytes(header[offset..offset + 8].try_into().unwrap());
            let end = u64::from_le_bytes(header[offset + 8..offset + 16].try_into().unwrap());
            Duration::from_secs_f64(end.wrapping_sub(begin) as f64 * f64::from(self.queue.get_timestamp_period()) / 1e9)
        });
        let mut records = Vec::with_capacity(count);
        let mut max_error = [0.0_f64; 2];
        let mut refined = 0;
        if count != 0 {
            let bytes = count * RECORD_BYTES;
            let overflow;
            let data = if count <= inline_capacity {
                &header[HEADER_BYTES as usize..HEADER_BYTES as usize + bytes]
            } else {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("紧凑候选回读"),
                });
                encoder.copy_buffer_to_buffer(&self.output, HEADER_BYTES, &self.record_staging, 0, bytes as u64);
                self.queue.submit(Some(encoder.finish()));
                overflow = map_read(&self.device, &self.record_staging, bytes as u64)?;
                &overflow
            };
            for record in data.as_chunks::<RECORD_BYTES>().0 {
                let mut words = [0_u32; RECORD_WORDS];
                for (word, bytes) in words.iter_mut().zip(record.as_chunks::<4>().0) {
                    *word = u32::from_le_bytes(*bytes);
                }
                let index = words[0] as usize;
                if index >= input_count {
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
            // 属性通过后每个候选必算评分，筛选在其后；不再做同一个计数的第二次原子操作。
            scored: word(1),
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

    #[test]
    fn cycle_generation_checks_encoding_layout_and_exact_token_span() {
        use crate::input::Encoding;

        let input = GpuInput::new("队@伍", "前[]后", Encoding::Base16384).unwrap().unwrap();
        let src = GpuOptions::default().shader_source(Some(&input), false).unwrap();
        assert!(src.contains("const DIRECT_CYCLE: bool = true;"));
        assert!(src.contains("const CYCLE_FIRST: u32 = 4u;"));
        assert!(src.contains("const CYCLE_LEN: u32 = 25u;"));
        assert!(src.contains("i + 22u, cursor, 229u")); // 后缀的第一个 UTF-8 字节。

        let wide = GpuOptions { sbox: SboxLayout::Wide, ..GpuOptions::default() };
        assert!(wide.shader_source(Some(&input), false).unwrap().contains("const DIRECT_CYCLE: bool = false;"));
        assert!(GpuOptions::default().shader_source(None, false).unwrap().contains("const DIRECT_CYCLE: bool = false;"));
        for (template, encoding) in [
            ("前[]后[]".to_owned(), Encoding::Base16384),
            ("前[]后".to_owned(), Encoding::Base64Url),
            (format!("{}[]", "x".repeat(47)), Encoding::Base16384),
        ] {
            let input = GpuInput::new("", &template, encoding).unwrap().unwrap();
            assert!(cycle_shader(&input).is_none(), "{template:?} {encoding:?}");
        }
    }

    fn samples(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| match i % 8 {
                0 => String::new(),
                1 => format!("中文😀名字{i}"),
                2 => format!("name-{i}"),
                3 => "a".repeat(256),
                4 => format!("{}x", "中".repeat(85)),
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
            for team in ["", "shenjack", "中文队伍", "边界队伍4"] {
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

    #[test]
    #[ignore = "需要GPU：验证不同共享状态布局和工作组"]
    fn gpu_packed_layouts_match_complete_reference() {
        pollster::block_on(async {
            let names = samples(257);
            let expected: Vec<_> = names.iter().map(|name| core::evaluate("边界队伍4", name)).collect();
            for options in [
                GpuOptions::default(),
                GpuOptions {
                    sbox: SboxLayout::Packed,
                    workgroup_size: 32,
                    compact_model: true,
                },
                GpuOptions {
                    sbox: SboxLayout::Packed,
                    workgroup_size: 64,
                    compact_model: true,
                },
                GpuOptions {
                    sbox: SboxLayout::Packed,
                    workgroup_size: 128,
                    compact_model: true,
                },
            ] {
                let mut gpu = GpuMiner::with_options("边界队伍4", 257, options).await.unwrap();
                let actual = gpu.run(&names, Thresholds::default(), true).unwrap();
                assert_eq!(actual.records.len(), names.len());
                for record in actual.records {
                    equal(&record.value, &expected[record.index]);
                }
                let thresholds = Thresholds {
                    prop: 550,
                    xp: 1500,
                    xd: 3000,
                };
                let actual = gpu.run(&names, thresholds, false).unwrap();
                let wanted: Vec<_> = expected
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| value.matches(thresholds))
                    .map(|(index, _)| index)
                    .collect();
                assert_eq!(
                    actual.records.iter().map(|record| record.index).collect::<Vec<_>>(),
                    wanted,
                    "{options:?}"
                );
            }
        });
    }

    #[test]
    #[ignore = "需要GPU：nonce生成、进位、编码和公共前缀完整对拍"]
    fn gpu_generated_names_match_reference_at_u64_boundaries() {
        use crate::input::{Encoding, Generator};
        pollster::block_on(async {
            let team = "生成@队伍";
            let mut cases = Vec::new();
            for options in [
                GpuOptions {
                    sbox: SboxLayout::Wide,
                    workgroup_size: 32,
                    compact_model: true,
                },
                GpuOptions::default(),
                GpuOptions {
                    sbox: SboxLayout::Packed,
                    workgroup_size: 64,
                    compact_model: true,
                },
                GpuOptions {
                    sbox: SboxLayout::Packed,
                    workgroup_size: 128,
                    compact_model: true,
                },
            ] {
                cases.push((Encoding::Base16384, "中文😀[]/[]tail".to_owned(), options));
            }
            for encoding in [Encoding::Base16384, Encoding::Base64Std, Encoding::Base64Url] {
                cases.push((encoding, "中文😀[]/[]tail".to_owned(), GpuOptions::default()));
                let prefix = match encoding {
                    Encoding::Base16384 => format!("{}a", "中".repeat(79)),
                    _ => format!("{}a", "中".repeat(81)),
                };
                cases.push((encoding, format!("{prefix}[]"), GpuOptions::default()));
            }
            cases.push((
                Encoding::Base16384,
                format!("原始@{}[]/[]😀", char::from(0)),
                GpuOptions::default(),
            ));
            for (encoding, template, options) in cases {
                let input = GpuInput::new(team, &template, encoding).unwrap().unwrap();
                let generator = Generator::new(template, encoding).unwrap();
                let mut gpu = GpuMiner::with_generated(team, 65, options, input).await.unwrap();
                for first in [
                    0,
                    1,
                    u64::from(u32::MAX) - 31,
                    1_u64 << 32,
                    0x0123_4567_89ab_cdef,
                    0x8000_0000_0000_0000,
                    u64::MAX - 64,
                ] {
                    let names: Vec<_> = (0..65).map(|offset| generator.generate(first + offset).unwrap()).collect();
                    let expected: Vec<_> = names.iter().map(|name| core::evaluate(team, name)).collect();
                    let actual = gpu.run_range(first, 65, Thresholds::default(), true).unwrap();
                    assert_eq!(actual.records.len(), 65, "{encoding:?} {first}");
                    for record in actual.records {
                        equal(&record.value, &expected[record.index]);
                    }
                    let thresholds = Thresholds {
                        prop: 550,
                        xp: 1500,
                        xd: 3000,
                    };
                    let actual = gpu.run_range(first, 65, thresholds, false).unwrap();
                    let wanted: Vec<_> = expected
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| value.matches(thresholds))
                        .map(|(i, _)| i)
                        .collect();
                    assert_eq!(
                        actual.records.iter().map(|record| record.index).collect::<Vec<_>>(),
                        wanted,
                        "{encoding:?} {first}"
                    );
                }
                assert!(gpu.run_range(u64::MAX, 2, Thresholds::default(), true).is_err());
                assert!(gpu.run_range(0, 66, Thresholds::default(), true).is_err());
                assert!(gpu.run(&[], Thresholds::default(), true).is_err());
                assert!(gpu.run_range(u64::MAX, 0, Thresholds::default(), true).unwrap().records.is_empty());
                let actual = gpu.run_range(u64::MAX, 1, Thresholds::default(), true).unwrap();
                equal(
                    &actual.records[0].value,
                    &core::evaluate(team, &generator.generate(u64::MAX).unwrap()),
                );
            }
        });
    }
}
