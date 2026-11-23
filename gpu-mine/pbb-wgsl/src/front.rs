use std::{
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
    sync::mpsc,
    time::Duration,
};
use wgpu::util::DeviceExt;

const STATE_WORDS: usize = 64;
const KEY_WORDS: usize = 253;
const INLINE_ENTRIES: u32 = 4096;

struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}

struct Front {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    binding: wgpu::BindGroup,
    params: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    timer: Option<Timer>,
    block: u32,
    capacity: u32,
}

fn shader(gate: &[i32; 45], block: u32) -> String {
    let mut source = format!("const LANES: u32 = {block}u;\nconst MIN_P7: i32 = {};\n", gate[44]);
    let planes = ["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "TH", "REM2", "REM3"];
    for (name, plane) in planes.iter().zip(gate.as_chunks::<4>().0) {
        use std::fmt::Write;
        writeln!(
            source,
            "const {name} = vec4<i32>({}, {}, {}, {});",
            plane[0], plane[1], plane[2], plane[3]
        )
        .unwrap();
    }
    source.push_str(include_str!("prefilter.wgsl"));
    source
}

impl Front {
    async fn new(config: &[u32], gate: &[i32; 45], capacity: u32, block: u32) -> Result<Self, String> {
        if ![32, 64, 128].contains(&block) || !(1..=1_048_576).contains(&capacity) {
            return Err("WGSL 批容量或工作组大小无效".to_owned());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .map_err(|err| format!("找不到 Vulkan 适配器：{err}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("识别到 Vulkan 软件设备 {}，请选择实体 GPU", info.name));
        }
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: block * 256,
            ..wgpu::Limits::downlevel_defaults()
        };
        if !limits.check_limits(&adapter.limits()) || capacity.div_ceil(block) > limits.max_compute_workgroups_per_dimension {
            return Err("批大小或工作组共享内存超出设备限制，请减小 --batch 或 --block".to_owned());
        }
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("PBB WGSL 整数前筛"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|err| format!("创建 Vulkan 计算设备失败：{err}"))?;
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("原版 PBB 一级 Q 门"),
            source: wgpu::ShaderSource::Wgsl(shader(gate, block).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PBB WGSL 前筛"),
            layout: None,
            module: &module,
            entry_point: Some("prefilter"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建 WGSL 管线失败：{error}"));
        }
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBB 队伍快照、名字模板和 KSA 游标"),
            contents: bytemuck::cast_slice(config),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = buffer("PBB 批次参数", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let out_bytes = (u64::from(capacity) + 1) * 4;
        let output = buffer(
            "PBB Q 门计数和候选",
            out_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let staging = buffer(
            "PBB 前筛回读",
            out_bytes + 16,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBB WGSL 前筛资源"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let timer = features.contains(wgpu::Features::TIMESTAMP_QUERY).then(|| Timer {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("PBB WGSL 前筛时间戳"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: buffer(
                "PBB 时间戳解析",
                16,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建 WGSL 缓冲区失败：{error}"));
        }
        eprintln!("GPU {} / Vulkan，工作组 {block}，完整评分由 PBB 原版 CPU 执行", info.name);
        Ok(Self {
            device,
            queue,
            pipeline,
            binding,
            params,
            output,
            staging,
            timer,
            block,
            capacity,
        })
    }

    fn map<T>(&self, bytes: u64, read: impl FnOnce(&[u8]) -> T) -> Result<T, String> {
        let range = self.staging.slice(..bytes);
        let (tx, rx) = mpsc::sync_channel(1);
        range.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(120)),
            })
            .map_err(|err| format!("等待 WGSL 前筛失败：{err}"))?;
        rx.recv().map_err(|err| err.to_string())?.map_err(|err| err.to_string())?;
        let data = range.get_mapped_range().map_err(|err| err.to_string())?;
        let result = read(&data);
        drop(data);
        self.staging.unmap();
        Ok(result)
    }

    fn run(&self, first: u64, capture: bool, dst: &mut [u32]) -> Result<(u32, f64), String> {
        let n = u32::try_from(dst.len()).map_err(|_| "WGSL 批容量溢出")?;
        if n == 0 || n > self.capacity || first.checked_add(u64::from(n - 1)).is_none() {
            return Err("WGSL 批次范围无效".to_owned());
        }
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[first as u32, (first >> 32) as u32, n, u32::from(capture)]),
        );
        let mut commands = self.device.create_command_encoder(&Default::default());
        commands.clear_buffer(&self.output, 0, Some(4));
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PBB 一级门控"),
                timestamp_writes: self.timer.as_ref().map(|timer| wgpu::ComputePassTimestampWrites {
                    query_set: &timer.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.binding, &[]);
            pass.dispatch_workgroups(n.div_ceil(self.block), 1, 1);
        }
        let inline_count = if capture { n } else { n.min(INLINE_ENTRIES) };
        let inline_bytes = (u64::from(inline_count) + 1) * 4;
        commands.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, inline_bytes);
        if let Some(timer) = &self.timer {
            commands.resolve_query_set(&timer.queries, 0..2, &timer.resolve, 0);
            commands.copy_buffer_to_buffer(&timer.resolve, 0, &self.staging, inline_bytes, 16);
        }
        self.queue.submit(Some(commands.finish()));
        let (retained, seconds) = self.map(inline_bytes + if self.timer.is_some() { 16 } else { 0 }, |data| {
            let retained = u32::from_le_bytes(data[..4].try_into().unwrap());
            let seconds = if self.timer.is_some() {
                let i = inline_bytes as usize;
                let begin = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
                let end = u64::from_le_bytes(data[i + 8..i + 16].try_into().unwrap());
                end.wrapping_sub(begin) as f64 * f64::from(self.queue.get_timestamp_period()) / 1e9
            } else {
                0.0
            };
            let written = if capture { n } else { retained };
            if written <= inline_count {
                decode(&data[4..4 + written as usize * 4], dst);
            }
            (retained, seconds)
        })?;
        if retained > n {
            return Err("WGSL 候选计数超过批容量".to_owned());
        }
        if !capture && retained > inline_count {
            let bytes = u64::from(retained) * 4;
            let mut commands = self.device.create_command_encoder(&Default::default());
            commands.copy_buffer_to_buffer(&self.output, 4, &self.staging, 0, bytes);
            self.queue.submit(Some(commands.finish()));
            self.map(bytes, |data| decode(data, dst))?;
        }
        Ok((retained, seconds))
    }
}

fn decode(data: &[u8], dst: &mut [u32]) {
    for (slot, bytes) in dst.iter_mut().zip(data.as_chunks::<4>().0) {
        *slot = u32::from_le_bytes(*bytes);
    }
}

fn ffi<T>(call: impl FnOnce() -> Result<T, String>) -> Option<T> {
    // 异常时 C++ 停止本轮并释放设备，后续不会重用该前筛实例。
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(error)) => {
            eprintln!("PBB WGSL：{error}");
            None
        }
        Err(_) => {
            eprintln!("PBB WGSL 前筛异常终止");
            None
        }
    }
}

#[no_mangle]
unsafe extern "C" fn pbb_front_open(
    state: *const u32,
    key: *const u32,
    setup: *const u32,
    gate: *const [i32; 45],
    capacity: u32,
    block: u32,
) -> *mut c_void {
    ffi(|| {
        // C++ 桥接层在本次调用中保持固定长度数组存活，创建完成后 GPU 资源自持。
        let mut config = Vec::with_capacity(STATE_WORDS + KEY_WORDS + 4);
        config.extend_from_slice(unsafe { slice::from_raw_parts(state, STATE_WORDS) });
        config.extend_from_slice(unsafe { slice::from_raw_parts(key, KEY_WORDS) });
        config.extend_from_slice(unsafe { slice::from_raw_parts(setup, 4) });
        let front = pollster::block_on(Front::new(&config, unsafe { &*gate }, capacity, block))?;
        Ok(Box::into_raw(Box::new(front)).cast())
    })
    .unwrap_or(ptr::null_mut())
}

#[no_mangle]
unsafe extern "C" fn pbb_front_run(
    front: *mut Front,
    first: u64,
    n: u32,
    capture: u32,
    result: *mut u32,
    count: *mut u32,
    seconds: *mut f64,
) -> i32 {
    if ffi(|| {
        // 宿主串行调用，同一批次的输出数组与两个标量互不重叠。
        let dst = unsafe { slice::from_raw_parts_mut(result, n as usize) };
        let (written, elapsed) = unsafe { &*front }.run(first, capture != 0, dst)?;
        unsafe {
            *count = written;
            *seconds = elapsed;
        }
        Ok(())
    })
    .is_some()
    {
        0
    } else {
        1
    }
}

#[no_mangle]
unsafe extern "C" fn pbb_front_close(front: *mut Front) {
    // 对应 open 唯一返回的所有权句柄，C++ 析构时释放一次。
    drop(unsafe { Box::from_raw(front) });
}
