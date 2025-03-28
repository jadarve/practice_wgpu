use std::{fs::File, vec};

use anyhow::{anyhow, Result};
use clap::Parser;
use wgpu::core::command;

const SHADER_CODE_WGSL: &str = r#"
@group(0) @binding(0)
var input_texture: texture_2d<u32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  // Avoid accessing the buffer out of bounds
  if (global_id.x >= 615u || global_id.y >= 264u) {
    return;
  }

  let rgba = textureLoad(input_texture, vec2(i32(global_id.x), i32(global_id.y)), 0);
  let gray = u32(f32(rgba.r) * 0.3 + f32(rgba.g) * 0.59 + f32(rgba.b) * 0.11);
  //let gray  = u32(rgba.r);
  textureStore(output_texture, vec2(i32(global_id.x), i32(global_id.y)), vec4(gray, gray, gray, 255));
}
"#;

#[derive(Parser)]
struct CliArgs {
    #[clap(short, long, default_value = "input.png")]
    input: String,
    #[clap(short, long, default_value = "output.png")]
    output: String,
}

///////////////////////////////////////////////////////////////////////////////
fn read_png(path: &str) -> Result<(png::OutputInfo, Vec<u8>)> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let (info, mut reader) = decoder.read_info()?;
    let mut buf = vec![0; info.buffer_size()];
    reader.next_frame(&mut buf)?;
    Ok((info, buf))
}

fn get_png_reader(path: &str) -> Result<(png::OutputInfo, png::Reader<File>)> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let (info, reader) = decoder.read_info()?;
    Ok((info, reader))
}

fn write_png(
    path: &str,
    info: png::OutputInfo,
    buf: Vec<u8>,
    row_alignment: Option<usize>,
) -> Result<()> {
    let buf_without_alignment = if let Some(row_alignment) = row_alignment {
        let row_size = info.width * info.color_type.samples() as u32;
        let size = info.height * row_size;

        let mut out = vec![0; size as usize];
        // remove the alignment padding
        for (i, row) in buf.chunks_exact(row_alignment).enumerate() {
            out[i * row_size as usize..(i + 1) * row_size as usize]
                .copy_from_slice(&row[..row_size as usize]);
        }

        out
    } else {
        buf
    };

    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, info.width, info.height);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&buf_without_alignment)?;
    Ok(())
}

///////////////////////////////////////////////////////////////////////////////
// CPU implementation
fn rgb2gray_cpu(input_buffer: &Vec<u8>) -> Vec<u8> {
    input_buffer
        .chunks_exact(3)
        .map(|pixel| {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let gray = (0.3 * r + 0.59 * g + 0.11 * b).round() as u8;
            [gray]
        })
        .flatten()
        .collect()
}

async fn run_cpu() -> Result<()> {
    let opts: CliArgs = CliArgs::parse();
    println!("Input: {}", opts.input);
    println!("Output: {}", opts.output);

    let (input_info, input_buffer) = read_png(&opts.input)?;

    println!("Input image: {:?}", input_info);

    let output_info = png::OutputInfo {
        width: input_info.width,
        height: input_info.height,
        color_type: png::ColorType::Grayscale,
        bit_depth: png::BitDepth::Eight,
        line_size: input_info.width as usize,
    };

    println!("Output image: {:?}", output_info);

    let output_buffer: Vec<u8> = rgb2gray_cpu(&input_buffer);

    println!("Output buffer: {}", output_buffer.len());

    write_png("output_cpu.png", output_info, output_buffer, None)?;

    Ok(())
}

///////////////////////////////////////////////////////////////////////////////
// CPU format conversion
fn rgb2rgba_cpu(input_buffer: &[u8], width: usize, height: usize) -> (Vec<u8>, usize) {
    let row_alignment =
        wgpu::util::align_to(4 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
    let mut rgba = vec![0u8; row_alignment * height];

    for i in 0..height {
        for j in 0..width {
            let rgb_idx = i * width * 3 + j * 3;
            let rgba_idx = i * row_alignment + j * 4;
            rgba[rgba_idx] = input_buffer[rgb_idx];
            rgba[rgba_idx + 1] = input_buffer[rgb_idx + 1];
            rgba[rgba_idx + 2] = input_buffer[rgb_idx + 2];
            rgba[rgba_idx + 3] = 255;
        }
    }

    (rgba, row_alignment)
}

///////////////////////////////////////////////////////////////////////////////
// GPU implementation
async fn run_gpu() -> Result<()> {
    ///////////////////////////////////////////////////////
    let opts: CliArgs = CliArgs::parse();
    println!("Input: {}", opts.input);
    println!("Output: {}", opts.output);

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
    // input buffer
    let (input_info, input_buffer_rgb) = read_png(&opts.input)?;

    // convert RGB to RGBA
    let (input_buffer_rgba, input_row_alignment) = rgb2rgba_cpu(
        &input_buffer_rgb,
        input_info.width as usize,
        input_info.height as usize,
    );

    let input_buffer_desc = wgpu::BufferDescriptor {
        label: Some("input_buffer"),
        size: input_buffer_rgba.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    };

    let input_buffer_device = device.create_buffer(&input_buffer_desc);
    println!("Input buffer: {:?}", input_buffer_device);

    input_buffer_device.slice(..).get_mapped_range_mut()[..].copy_from_slice(&input_buffer_rgba);
    input_buffer_device.unmap();

    ///////////////////////////////////////////////////////
    // output buffer
    let output_buffer_desc = wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: input_buffer_rgba.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };

    let output_buffer_device = device.create_buffer(&output_buffer_desc);
    println!("Output buffer: {:?}", output_buffer_device);

    ///////////////////////////////////////////////////////
    // input texture
    let input_texture_desc = wgpu::TextureDescriptor {
        label: Some("input_texture"),
        size: wgpu::Extent3d {
            width: input_info.width,
            height: input_info.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Uint,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[wgpu::TextureFormat::Rgba8Uint],
    };

    let input_texture = device.create_texture(&input_texture_desc);

    ///////////////////////////////////////////////////////
    // input texture view
    let input_texture_view_desc = wgpu::TextureViewDescriptor {
        label: Some("input_texture_view"),
        format: Some(wgpu::TextureFormat::Rgba8Uint),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
        usage: Some(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING),
    };
    let input_texture_view = input_texture.create_view(&input_texture_view_desc);

    ///////////////////////////////////////////////////////
    // output texture
    let output_texture_desc = wgpu::TextureDescriptor {
        label: Some("output_texture"),
        size: wgpu::Extent3d {
            width: input_info.width,
            height: input_info.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Uint,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[wgpu::TextureFormat::Rgba8Uint],
    };

    let output_texture = device.create_texture(&output_texture_desc);

    ///////////////////////////////////////////////////////
    // output texture view
    let output_texture_view_desc = wgpu::TextureViewDescriptor {
        label: Some("output_texture_view"),
        format: Some(wgpu::TextureFormat::Rgba8Uint),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
        usage: Some(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING),
    };
    let output_texture_view = output_texture.create_view(&output_texture_view_desc);

    ///////////////////////////////////////////////////////
    // shader module
    let shader_module_desc = wgpu::ShaderModuleDescriptor {
        label: Some("shader_module"),
        // source: wgpu::ShaderSource::Glsl {
        //     shader: shader.into(),
        //     stage: wgpu::naga::ShaderStage::Compute,
        //     defines: Default::default(),
        // },
        source: wgpu::ShaderSource::Wgsl(SHADER_CODE_WGSL.into()),
    };
    let shader_module = device.create_shader_module(shader_module_desc);

    ///////////////////////////////////////////////////////
    // compute pipeline
    let layout = wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout"),
        bind_group_layouts: &[
            &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            }),
        ],
        push_constant_ranges: &[],
    };

    let pipeline_layout = device.create_pipeline_layout(&layout);

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_texture_view),
            },
        ],
    });

    // copy input buffer to input texture
    let command_encoder_desc = wgpu::CommandEncoderDescriptor {
        label: Some("command_encoder"),
    };
    let mut command_encoder = device.create_command_encoder(&command_encoder_desc);
    command_encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &input_buffer_device,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(input_row_alignment as u32),
                rows_per_image: Some(input_info.height),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &input_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: input_info.width,
            height: input_info.height,
            depth_or_array_layers: 1,
        },
    );

    {
        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            (input_info.width + 15) / 16,
            (input_info.height + 15) / 16,
            1,
        );
    }

    command_encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer_device,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(input_row_alignment as u32),
                rows_per_image: Some(input_info.height),
            },
        },
        wgpu::Extent3d {
            width: input_info.width,
            height: input_info.height,
            depth_or_array_layers: 1,
        },
    );

    let command_buffer = command_encoder.finish();

    queue.submit([command_buffer]);

    ///////////////////////////////////////////////////////
    // read output buffer
    let buffer_slice = output_buffer_device.slice(..);

    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        {
            let data = buffer_slice.get_mapped_range();
            let d = bytemuck::cast_slice::<u8, u8>(&data);

            let output_info = png::OutputInfo {
                width: input_info.width,
                height: input_info.height,
                color_type: png::ColorType::RGBA,
                bit_depth: png::BitDepth::Eight,
                line_size: input_row_alignment as usize,
            };

            write_png(
                "output_gpu.png",
                output_info,
                data.to_vec(),
                Some(input_row_alignment),
            )?;
        }

        output_buffer_device.unmap();
    }

    Ok(())
}

fn main() -> Result<()> {
    pollster::block_on(run_cpu())?;
    pollster::block_on(run_gpu())?;

    Ok(())
}
