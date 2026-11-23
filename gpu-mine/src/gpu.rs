use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

pub const STATE_SIZE: usize = 256;
pub const MAX_NAME_BYTES: usize = 256;
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Kernel {
    Private,
    Shared,
}

impl Kernel {
    fn workgroup_size(self) -> u32 {
        match self {
            Self::Private => 64,
            Self::Shared => 32,
        }
    }
    fn source(self) -> &'static str {
        match self {
            Self::Private => include_str!("../main.wgsl"),
            Self::Shared => include_str!("../rc4-shared.wgsl"),
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Private => "private",
            Self::Shared => "shared",
        }
    }
}

pub struct GpuResult {
    pub states: Vec<u8>,
    pub kernel_time: Option<Duration>,
    pub preparation_time: Duration,
    pub wall_time: Duration,
}

/// 一个实例复用设备、管线、队伍状态及所有 GPU 缓冲；一次调用处理一个批次。
pub struct GpuRc4 {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    metadata_buffer: wgpu::Buffer,
    bytes_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    queries: Option<wgpu::QuerySet>,
    query_resolve: Option<wgpu::Buffer>,
    capacity: usize,
    workgroup_size: u32,
    metadata: Vec<[u32; 2]>,
    bytes: Vec<u8>,
    pub adapter_description: String,
}

impl GpuRc4 {
    pub async fn new(team: &str, capacity: usize, kernel: Kernel) -> Result<Self, String> {
        if capacity == 0 || team.len() > MAX_NAME_BYTES {
            return Err("批容量必须大于零，队伍名最多 256 个 UTF-8 字节".to_owned());
        }
        let buffer_bytes = capacity.checked_mul(STATE_SIZE).ok_or("批容量溢出")?;
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("获取 GPU 失败：{error}"))?;
        let info = adapter.get_info();
        let adapter_description = format!("{} / {:?} / {:?}", info.name, info.backend, info.device_type);
        let mut limits = wgpu::Limits::downlevel_defaults();
        let workgroup_size = kernel.workgroup_size();
        if matches!(kernel, Kernel::Shared) {
            if adapter.limits().max_compute_workgroup_storage_size < 32768 {
                return Err("shared 内核需要至少 32KiB 工作组共享内存；请改用 --kernel private".to_owned());
            }
            limits.max_compute_workgroup_storage_size = 32768;
        }
        if !limits.check_limits(&adapter.limits())
            || buffer_bytes as u64 > limits.max_storage_buffer_binding_size
            || (capacity as u64).div_ceil(u64::from(workgroup_size)) > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err("批容量超过 GPU 缓冲区或工作组限制".to_owned());
        }
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        tracing::debug!(capacity, timestamps, gpu = %adapter_description, "初始化可复用 RC4 引擎");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("批量 RC4 实验设备"),
                required_features: if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("创建 GPU 设备失败：{error}"))?;
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("批量 RC4 KSA"),
            source: wgpu::ShaderSource::Wgsl(kernel.source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("两轮名字 KSA，一候选一 invocation"),
            layout: None,
            module: &module,
            entry_point: Some("rc4_name"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建 RC4 管线失败：{error}"));
        }
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let team_values = crate::cpu::team_state(team).map(u32::from);
        let team_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("固定队伍 KSA 状态"),
            contents: bytemuck::cast_slice(&team_values),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let make_buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let metadata_buffer = make_buffer(
            "名字偏移与长度",
            (capacity * 8) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let bytes_buffer = make_buffer(
            "紧凑 UTF-8 名字字节",
            (buffer_bytes as u64).max(4),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params_buffer = make_buffer("真实候选数", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let output_buffer = make_buffer(
            "打包后的 256 字节 RC4 状态",
            buffer_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging_buffer = make_buffer(
            "RC4 状态与时间戳回读",
            buffer_bytes as u64 + 16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let buffers = [&team_buffer, &metadata_buffer, &bytes_buffer, &output_buffer, &params_buffer];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("批量 RC4 输入输出"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let queries = timestamps.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("RC4 pass 时间戳"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        let query_resolve = timestamps.then(|| {
            make_buffer(
                "时间戳解析",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            )
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建 RC4 缓冲区失败：{error}"));
        }
        Ok(Self {
            device,
            queue,
            pipeline,
            bindings,
            metadata_buffer,
            bytes_buffer,
            params_buffer,
            output_buffer,
            staging_buffer,
            queries,
            query_resolve,
            capacity,
            workgroup_size,
            metadata: Vec::with_capacity(capacity),
            bytes: Vec::new(),
            adapter_description,
        })
    }

    pub fn has_timestamps(&self) -> bool { self.queries.is_some() }

    /// 计时包含 UTF-8 打包、上传、提交、完整状态回读与结果复制；不包含设备/管线初始化和名字生成。
    pub fn run(&mut self, names: &[String]) -> Result<GpuResult, String> {
        let wall_start = Instant::now();
        if names.len() > self.capacity {
            return Err(format!("候选数 {} 超过批容量 {}", names.len(), self.capacity));
        }
        self.metadata.clear();
        self.bytes.clear();
        for (index, name) in names.iter().enumerate() {
            if name.len() > MAX_NAME_BYTES {
                return Err(format!("名字 {index} 超过 256 个 UTF-8 字节"));
            }
            self.metadata.push([self.bytes.len() as u32, name.len() as u32]);
            self.bytes.extend_from_slice(name.as_bytes());
        }
        self.bytes.resize(self.bytes.len().max(1).div_ceil(4) * 4, 0);
        let preparation_time = wall_start.elapsed();
        if names.is_empty() {
            return Ok(GpuResult {
                states: vec![],
                kernel_time: None,
                preparation_time,
                wall_time: wall_start.elapsed(),
            });
        }
        self.queue.write_buffer(&self.metadata_buffer, 0, bytemuck::cast_slice(&self.metadata));
        self.queue.write_buffer(&self.bytes_buffer, 0, &self.bytes);
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[names.len() as u32, 0, 0, 0]));
        let output_bytes = (names.len() * STATE_SIZE) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("批量 RC4")
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RC4 两轮 KSA"),
                timestamp_writes: self.queries.as_ref().map(|queries| wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups((names.len() as u32).div_ceil(self.workgroup_size), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, output_bytes);
        if let (Some(queries), Some(resolve)) = (&self.queries, &self.query_resolve) {
            encoder.resolve_query_set(queries, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, &self.staging_buffer, output_bytes, 16);
        }
        self.queue.submit(Some(encoder.finish()));
        let read_bytes = output_bytes + if self.queries.is_some() { 16 } else { 0 };
        let slice = self.staging_buffer.slice(..read_bytes);
        let (sender, receiver) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|error| format!("等待 GPU 失败：{error}"))?;
        receiver
            .recv()
            .map_err(|error| format!("回读通知失败：{error}"))?
            .map_err(|error| format!("缓冲映射失败：{error}"))?;
        let data = slice.get_mapped_range().map_err(|error| format!("读取映射范围失败：{error}"))?;
        let states = data[..output_bytes as usize].to_vec();
        let kernel_time = if self.queries.is_some() {
            let offset = output_bytes as usize;
            let start = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            let end = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
            let ticks = end.wrapping_sub(start);
            Some(Duration::from_secs_f64(
                ticks as f64 * f64::from(self.queue.get_timestamp_period()) / 1e9,
            ))
        } else {
            None
        };
        drop(data);
        self.staging_buffer.unmap();
        Ok(GpuResult {
            states,
            kernel_time,
            preparation_time,
            wall_time: wall_start.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| match i % 7 {
                0 => String::new(),
                1 => format!("中文名字{i}"),
                2 => format!("name-{i}"),
                3 => "x".repeat(256),
                4 => format!("{}a", "中".repeat(85)),
                5 => format!("NUL\0{i}"),
                _ => format!("😀尾{i}"),
            })
            .collect()
    }

    #[test]
    #[ignore = "需要可用 GPU，验证多工作组、尾部边界和持久缓冲重用"]
    fn gpu_matches_independent_cpu_rc4() {
        pollster::block_on(async {
            for (kernel, team) in [Kernel::Private, Kernel::Shared]
                .into_iter()
                .flat_map(|kernel| ["", "中文队伍", "shenjack"].map(|team| (kernel, team)))
            {
                let mut gpu = GpuRc4::new(team, 513, kernel).await.unwrap();
                for count in [0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 513, 1, 65] {
                    let names = names(count);
                    let actual = gpu.run(&names).unwrap();
                    let expected: Vec<u8> = names.iter().flat_map(|name| crate::cpu::reference(team, name)).collect();
                    assert_eq!(actual.states, expected, "team={team:?}, count={count}");
                }
                assert!(gpu.run(&vec![String::new(); 514]).is_err());
                assert!(gpu.run(&["中".repeat(86)]).is_err());
                assert_eq!(gpu.run(&["x".to_owned()]).unwrap().states, crate::cpu::reference(team, "x"));
            }
        });
    }
}
