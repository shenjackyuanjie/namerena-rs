use std::{
    cell::Cell,
    ffi::c_void,
    fmt::Write,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
    sync::mpsc,
    time::Duration,
};
use wgpu::util::DeviceExt;

const HEADER: u64 = 8;
const INLINE_ENTRIES: u32 = 4096;
const CAPACITY_LIMIT: u32 = 1 << 24;
const FREE: u32 = 1 << 31;

struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}

struct Step {
    pipeline: wgpu::ComputePipeline,
    binding: wgpu::BindGroup,
}

struct Front {
    device: wgpu::Device,
    queue: wgpu::Queue,
    q: Step,
    g3: Step,
    params: wgpu::Buffer,
    qualified: wgpu::Buffer,
    selected: wgpu::Buffer,
    staging: wgpu::Buffer,
    timer: Option<Timer>,
    block: u32,
    g3_block: u32,
    capacity: u32,
    max_dispatch: u32,
    profile: Option<Cell<[f64; 3]>>,
}

fn shader(config: &[u32], gate: &[i32; 45], block: u32, fast: bool) -> String {
    let mut source = format!(
        "const LANES: u32 = {block}u;\nconst MIN_P7: i32 = {};\nconst FAST_KEY: bool = {fast};\n",
        gate[44]
    );
    let planes = ["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "TH", "REM2", "REM3"];
    for (name, w) in planes.iter().zip(gate.as_chunks::<4>().0) {
        writeln!(source, "const {name} = vec4<i32>({}, {}, {}, {});", w[0], w[1], w[2], w[3]).unwrap();
    }
    source.push_str(include_str!("prefilter.wgsl"));
    let mut fast_code = include_str!("ksa-fast.wgsl").to_owned();
    if fast {
        let pairs = &config[config.len() - 16..];
        let mut prefix = String::new();
        for (index, pair) in pairs.iter().enumerate().skip(1) {
            let k = index * 2;
            writeln!(prefix, "    if end > {k}u {{ j = ksa_fast_pair(lane, i + {k}u, j, {pair}u); }}").unwrap();
        }
        let mut crop = String::new();
        for k in (4..32).step_by(4) {
            let (a, b) = (pairs[k / 2], pairs[k / 2 + 1]);
            writeln!(crop, "    j = ksa_fast_crop_word(lane, i + {k}u, j, {a}u, {b}u);").unwrap();
        }
        fast_code = fast_code
            .replace("// FULL_LITERAL_PAIRS", &prefix)
            .replace("// CROPPED_LITERAL_PAIRS", &crop);
    }
    source.push_str(&fast_code);
    source.push_str(include_str!("g3.wgsl"));
    source
}

impl Front {
    async fn new(config: &[u32], gate: &[i32; 45], capacity: u32, block: u32, fast: bool) -> Result<Self, String> {
        if ![32, 64, 128].contains(&block) || !(1..=CAPACITY_LIMIT).contains(&capacity) {
            return Err("WGSL 批容量或工作组大小无效".to_owned());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .map_err(|e| format!("找不到 Vulkan 适配器：{e}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("识别到 Vulkan 软件设备 {}，请选择实体 GPU", info.name));
        }
        let out_bytes = (u64::from(capacity) + 1) * 4;
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: block * 256,
            ..wgpu::Limits::downlevel_defaults()
        };
        if !limits.check_limits(&adapter.limits()) || out_bytes > limits.max_storage_buffer_binding_size {
            return Err("批大小或工作组共享内存超出设备限制，请减小 --batch 或 --block".to_owned());
        }
        let max_dispatch = limits.max_compute_workgroups_per_dimension;
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("PBB WGSL Q/G3 门控"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("创建 Vulkan 计算设备失败：{e}"))?;
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let init = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBB 模板、前缀快照与 G3 权重"),
            contents: bytemuck::cast_slice(config),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = buffer("PBB 批次参数", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let output_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let qualified = buffer("Q 门压缩队列", out_bytes, output_usage);
        let selected = buffer("G3 保守过门队列", out_bytes, output_usage);
        let staging = buffer(
            "过门候选回读",
            out_bytes + HEADER + 32,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let g3_block = block.min(64);
        let make = |entry, lanes, bufs: &[&wgpu::Buffer]| {
            let descriptor = wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(shader(config, gate, lanes, fast).into()),
            };
            let module = if fast && entry == "prefilter" {
                // 只有固定 50 字节键环的 Q 核去掉运行时边界检查：fast 守卫了键环形状，
                // 顺序槽 0..255、随机槽按 255 掩码，每个 lane 只访问自己的 64 个字。
                // 属性扫描的 34 槽内只消费前 31 项，排序 10 项；早退守住 i<n，每个 i 最多入队一次。
                // 通用模板和读压缩队列的 G3 核保持完整运行时检查。
                unsafe { device.create_shader_module_trusted(descriptor, wgpu::ShaderRuntimeChecks::unchecked()) }
            } else {
                device.create_shader_module(descriptor)
            };
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });
            let entries: Vec<_> = bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            Step { pipeline, binding }
        };
        let q = make("prefilter", block, &[&init, &params, &qualified]);
        let g3 = make("secondary", g3_block, &[&init, &params, &qualified, &selected]);
        let timer = features.contains(wgpu::Features::TIMESTAMP_QUERY).then(|| Timer {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("PBB WGSL Q/G3 时间戳"),
                ty: wgpu::QueryType::Timestamp,
                count: 4,
            }),
            resolve: buffer(
                "PBB 时间戳解析",
                32,
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
        });
        if let Some(error) = errors.pop().await {
            return Err(format!("创建 WGSL 管线或缓冲区失败：{error}"));
        }
        eprintln!(
            "GPU {} / Vulkan，Q 工作组 {block}，G3 工作组 {g3_block}，尾部使用原版 CPU 精算",
            info.name
        );
        Ok(Self {
            device,
            queue,
            q,
            g3,
            params,
            qualified,
            selected,
            staging,
            timer,
            block,
            g3_block,
            capacity,
            max_dispatch,
            profile: std::env::var_os("PBB_WGSL_PROFILE").map(|_| Cell::new([0.0; 3])),
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
            .map_err(|e| format!("等待 WGSL 前筛失败：{e}"))?;
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let data = range.get_mapped_range().map_err(|e| e.to_string())?;
        let result = read(&data);
        drop(data);
        self.staging.unmap();
        Ok(result)
    }

    fn pass(&self, commands: &mut wgpu::CommandEncoder, step: &Step, x: u32, y: u32, i: u32) {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: self.timer.as_ref().map(|timer| wgpu::ComputePassTimestampWrites {
                query_set: &timer.queries,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            }),
        });
        pass.set_pipeline(&step.pipeline);
        pass.set_bind_group(0, &step.binding, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    fn read_queue(&self, queue: &wgpu::Buffer, count: u32, dst: &mut [u32]) -> Result<(), String> {
        if count != 0 {
            let bytes = u64::from(count) * 4;
            let mut commands = self.device.create_command_encoder(&Default::default());
            commands.copy_buffer_to_buffer(queue, 4, &self.staging, 0, bytes);
            self.queue.submit(Some(commands.finish()));
            self.map(bytes, |data| decode(data, dst, !0))?;
        }
        Ok(())
    }

    fn run(&self, first: u64, capture: bool, dst: &mut [u32], q_ids: Option<&mut [u32]>) -> Result<([u32; 2], f64), String> {
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
        commands.clear_buffer(&self.qualified, 0, Some(4));
        let groups = n.div_ceil(self.block);
        let rows = groups.div_ceil(self.max_dispatch);
        self.pass(&mut commands, &self.q, groups.div_ceil(rows), rows, 0);
        let (out, query_count) = if capture {
            (&self.qualified, 2)
        } else {
            commands.clear_buffer(&self.selected, 0, Some(4));
            // 固定网格跨步取 Q 队列，不等主机回读计数。少量批次不派发无用的大网格。
            let groups = n.div_ceil(64 * self.g3_block).clamp(1, 1024);
            self.pass(&mut commands, &self.g3, groups, 1, 2);
            (&self.selected, 4)
        };
        let inline_count = if capture { n } else { n.min(INLINE_ENTRIES) };
        let inline_bytes = HEADER + u64::from(inline_count) * 4;
        commands.copy_buffer_to_buffer(&self.qualified, 0, &self.staging, 0, 4);
        commands.copy_buffer_to_buffer(out, 0, &self.staging, 4, inline_bytes - 4);
        let timestamp_bytes = self.timer.as_ref().map_or(0, |timer| {
            commands.resolve_query_set(&timer.queries, 0..query_count, &timer.resolve, 0);
            let len = u64::from(query_count) * 8;
            commands.copy_buffer_to_buffer(&timer.resolve, 0, &self.staging, inline_bytes, len);
            len
        });
        self.queue.submit(Some(commands.finish()));
        let (count, q_time, g3_time) = self.map(inline_bytes + timestamp_bytes, |data| {
            let counts = [
                u32::from_le_bytes(data[..4].try_into().unwrap()),
                u32::from_le_bytes(data[4..8].try_into().unwrap()),
            ];
            let phase = |at| {
                let i = inline_bytes as usize + at;
                let begin = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
                let end = u64::from_le_bytes(data[i + 8..i + 16].try_into().unwrap());
                end.wrapping_sub(begin) as f64 * f64::from(self.queue.get_timestamp_period()) / 1e9
            };
            let q_time = if timestamp_bytes != 0 { phase(0) } else { 0.0 };
            let g3_time = if timestamp_bytes == 32 { phase(16) } else { 0.0 };
            let written = if capture { n } else { counts[1] };
            if written <= inline_count {
                decode(&data[8..8 + written as usize * 4], dst, !0);
            }
            (counts, q_time, g3_time)
        })?;
        if count[0] > n || count[1] > count[0] {
            return Err("WGSL 门控计数超过批容量或 Q 门计数".to_owned());
        }
        if !capture && count[1] > inline_count {
            self.read_queue(out, count[1], dst)?;
        }
        if let Some(q_ids) = q_ids {
            self.read_queue(&self.qualified, count[0], q_ids)?;
            for id in &mut q_ids[..count[0] as usize] {
                *id &= !FREE;
            }
        }
        if let Some(p) = &self.profile {
            let [q, g3, sent] = p.get();
            p.set([q + q_time, g3 + g3_time, sent + f64::from(count[1])]);
        }
        Ok((count, q_time + g3_time))
    }
}

impl Drop for Front {
    fn drop(&mut self) {
        if let Some(p) = &self.profile {
            let [q, g3, request] = p.get();
            eprintln!("WGSL 分段（含预热）：Q={q:.6}s G3={g3:.6}s CPU队列={request:.0}");
        }
    }
}

fn decode(data: &[u8], dst: &mut [u32], mask: u32) {
    for (slot, bytes) in dst.iter_mut().zip(data.as_chunks::<4>().0) {
        *slot = u32::from_le_bytes(*bytes) & mask;
    }
}

fn ffi<T>(call: impl FnOnce() -> Result<T, String>) -> Option<T> {
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
    state: *const [u32; 64],
    key: *const [u32; 253],
    setup: *const [u32; 4],
    gate: *const [i32; 45],
    g3: *const [u32; 49],
    capacity: u32,
    block: u32,
) -> *mut c_void {
    ffi(|| {
        // 调用期间数组有效；生成完的 GPU 资源不保留 C++ 指针。
        let (key, setup) = unsafe { (&*key, &*setup) };
        let fast = setup[0] == 32 && setup[1] == 31 && setup[3] == 49 && key[31..49].iter().copied().eq(256..274);
        let mut pairs = [0; 16];
        if fast {
            pairs[0] = key[0] << 8;
            for (j, p) in pairs.iter_mut().enumerate().skip(1) {
                *p = key[2 * j - 1] | (key[2 * j] << 8);
            }
        }
        let mut config = Vec::with_capacity(64 + 253 + 4 + 49 + pairs.len());
        config.extend_from_slice(unsafe { &*state });
        config.extend_from_slice(key);
        config.extend_from_slice(setup);
        config.extend_from_slice(unsafe { &*g3 });
        config.extend(pairs);
        let front = pollster::block_on(Front::new(&config, unsafe { &*gate }, capacity, block, fast))?;
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
    counts: *mut [u32; 2],
    gate_indices: *mut u32,
    seconds: *mut f64,
) -> i32 {
    if ffi(|| {
        // 宿主保证单线程串行调用，两个回读数组与计数和时间缓冲互不重叠。
        let dst = unsafe { slice::from_raw_parts_mut(result, n as usize) };
        let q_ids = (!gate_indices.is_null()).then(|| unsafe { slice::from_raw_parts_mut(gate_indices, n as usize) });
        let (written, elapsed) = unsafe { &*front }.run(first, capture != 0, dst, q_ids)?;
        unsafe {
            *counts = written;
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
    // 对应 open 返回的唯一所有权句柄。
    drop(unsafe { Box::from_raw(front) });
}
