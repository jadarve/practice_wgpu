use std::{fs::File, vec};

use anyhow::{anyhow, Result};
use clap::Parser;

const SHADER_CODE: &str = r#"
#version 450

#ifndef ASSIGN_COMP_
#define ASSIGN_COMP_


layout(binding = 0) buffer out0 {
    float outputBuffer[];
};

void main() {

    const uint index = gl_GlobalInvocationID.x;
    outputBuffer[index] = index;
}

#endif // ASSIGN_COMP_
"#;

const SHADER_CODE_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read_write> outputBuffer: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  // Avoid accessing the buffer out of bounds
  if (global_id.x >= 1024u) {
    return;
  }

  outputBuffer[global_id.x] = f32(global_id.x);
}
"#;

#[derive(Parser)]
struct CliArgs {
    #[clap(short, long, default_value = "1024")]
    size: u32,
}

///////////////////////////////////////////////////////////////////////////////
// GPU implementation
async fn run_gpu() -> Result<()> {
    ///////////////////////////////////////////////////////
    let opts: CliArgs = CliArgs::parse();
    println!("Size: {}", opts.size);

    let size_bytes = opts.size as u64 * size_of::<f32>() as u64;

    // create a wgpu instance
    let instance = wgpu::Instance::default();
    println!("Instance: {:?}", instance);

    // get a wgpu adapter
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };

    let adapter = instance
        .request_adapter(&options)
        .await
        .ok_or(anyhow!("No adapter found"))?;

    println!("Adapter: {:?}", adapter.get_info());

    // request a wgpu device
    let desc = wgpu::DeviceDescriptor {
        label: Some("rgb2gray_device"),
        ..Default::default()
    };
    let (device, queue) = adapter.request_device(&desc, None).await?;

    println!("Device: {:?}", device);

    ///////////////////////////////////////////////////////
    // output buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    println!("Output buffer: {:?}", output_buffer);

    ///////////////////////////////////////////////////////
    // output staging buffer
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_staging"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    println!("Output staging buffer: {:?}", output_staging_buffer);

    ///////////////////////////////////////////////////////
    // shader module
    let shader_module_desc = wgpu::ShaderModuleDescriptor {
        label: Some("shader_module"),
        // source: wgpu::ShaderSource::Glsl {
        //     shader: SHADER_CODE.into(),
        //     stage: wgpu::naga::ShaderStage::Compute,
        //     defines: Default::default(),
        // },
        source: wgpu::ShaderSource::Wgsl(SHADER_CODE_WGSL.into()),
    };
    let shader_module = device.create_shader_module(shader_module_desc);

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    ///////////////////////////////////////////////////////
    // command encoder

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(4, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_staging_buffer, 0, size_bytes);

    queue.submit(Some(encoder.finish()));

    ///////////////////////////////////////////////////////
    // read output buffer
    let buffer_slice = output_staging_buffer.slice(..);

    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        {
            let data = buffer_slice.get_mapped_range();
            let d = bytemuck::cast_slice::<u8, f32>(&data);

            println!("Data: {:?}", data);
            println!("D: {:?}", d);
        }

        output_staging_buffer.unmap();
    }

    Ok(())
}

fn main() -> Result<()> {
    pollster::block_on(run_gpu())?;

    Ok(())
}
