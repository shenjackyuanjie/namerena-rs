use std::{borrow::Cow, io::Read};
use wgpu::util::DeviceExt;

/// 工作信息
pub struct Works {
    /// 队伍名(统一)
    pub team: String,
    /// 队伍名(每个任务)
    pub names: Vec<String>,
}

#[cfg_attr(test, allow(dead_code))]
async fn run() {
    let works = Works {
        team: "team".to_string(),
        names: vec!["name1".to_string(), "name2".to_string()],
    };

    let steps = execute_gpu(works).await.unwrap();

    let disp_steps: Vec<String> = steps.iter().map(|&n| n.to_string()).collect();

    println!("Steps: [{}]", disp_steps.join(", "));
}

#[cfg_attr(test, allow(dead_code))]
async fn execute_gpu(works: Works) -> Option<Vec<u32>> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, works).await
}

async fn execute_gpu_inner(device: &wgpu::Device, queue: &wgpu::Queue, works: Works) -> Option<Vec<u32>> {
    // Loads the shader from WGSL
    let source = {
        let mut file = std::fs::File::open("main.wgsl").unwrap();
        let mut source = String::new();
        file.read_to_string(&mut source).unwrap();
        source
    };

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&source)),
    });

    // 初始化一下 team bytes 计算出的 val
    let mut val: Vec<u8> = (0..=255).collect();
    let team_bytes = works.team.as_bytes();
    let t_len = works.team.len() + 1;
    let mut s = 0_u8;
    for i in 0..256 {
        if (i % t_len) != 0 {
            s = s.wrapping_add(team_bytes[(i % t_len) - 1]);
        }
        s = s.wrapping_add(val[i]);
        val.swap(i, s as usize);
    }

    // 满足 uniform 的对其要求, 把 vec<u8; 256> 转成 vec<[u8; 4]; 64>
    let uniform_val = val.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<[u8; 4]>>();

    // Gets the size in bytes of the buffer.
    let size = std::mem::size_of_val(&uniform_val) as wgpu::BufferAddress;

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("team bytes input buffer"),
        contents: bytemuck::cast_slice(&uniform_val),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "team_bytes",
        // constants: &Default::default(),
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    // encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub fn main() {
    tracing_subscriber::fmt::init();
    // env_logger::init();
    pollster::block_on(run());
}
