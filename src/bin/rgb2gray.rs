use core::hash;
use std::{fs::File, vec};

use anyhow::{anyhow, Result};
use clap::Parser;

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

fn write_png(path: &str, info: png::OutputInfo, buf: Vec<u8>) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, info.width, info.height);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&buf)?;
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

    write_png(&opts.output, output_info, output_buffer)?;

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
    };
    let input_texture_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
        format: wgpu::TextureFormat::Rg8Uint,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[wgpu::TextureFormat::R8Uint],
    };

    let output_texture = device.create_texture(&output_texture_desc);

    ///////////////////////////////////////////////////////
    // output texture view
    let output_texture_view_desc = wgpu::TextureViewDescriptor {
        label: Some("output_texture_view"),
        format: Some(wgpu::TextureFormat::R8Uint),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    };
    let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    ///////////////////////////////////////////////////////
    // shader
    // ```glsl
    // #version 450
    // layout(set = 0, binding = 0) uniform texture2D input_texture;
    // layout(set = 0, binding = 1) uniform texture2D output_texture;
    // layout(location = 0) out vec4 out_color;
    // void main() {
    //     vec4 color = texture(input_texture, gl_FragCoord.xy / textureSize(input_texture, 0));
    //     float gray = 0.3 * color.r + 0.59 * color.g + 0.11 * color.b;
    //     out_color = vec4(gray, gray, gray, 1.0);
    // }
    // ```
    let shader = r#"
        /**
RGBA2Gray.comp

Parameters
----------
in_rgba : rgba8ui uimage2D.
    input image.

out_gray : r8ui uimage2D.
    output image in gray scale.
*/

#version 450

#include <lluvia/core.glsl>
#include <lluvia/core/color.glsl>

layout(binding = 0, rgba8ui) uniform uimage2D in_rgba;
layout(binding = 1, r8ui) uniform uimage2D out_gray;

void main()
{

    const ivec2 coords  = LL_GLOBAL_COORDS_2D;
    const ivec2 imgSize = imageSize(out_gray);

    if (coords.x > imgSize.x || coords.y > imgSize.y) {
        return;
    }

    const uvec4 RGBA = imageLoad(in_rgba, coords);
    const uint  gray = color_rgba2gray(RGBA);

    imageStore(out_gray, coords, uvec4(gray));

    "#;

    ///////////////////////////////////////////////////////
    // shader module
    let shader_module_desc = wgpu::ShaderModuleDescriptor {
        label: Some("shader_module"),
        source: wgpu::ShaderSource::Glsl {
            shader: shader.into(),
            stage: wgpu::naga::ShaderStage::Compute,
            defines: Default::default(),
        },
    };
    let shader_module = device.create_shader_module(shader_module_desc);

    // copy input buffer to input texture
    let command_encoder_desc = wgpu::CommandEncoderDescriptor {
        label: Some("command_encoder"),
    };
    let mut command_encoder = device.create_command_encoder(&command_encoder_desc);
    command_encoder.copy_buffer_to_texture(
        wgpu::ImageCopyBuffer {
            buffer: &input_buffer_device,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(input_row_alignment as u32),
                rows_per_image: Some(input_info.height),
            },
        },
        wgpu::ImageCopyTexture {
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

    let command_buffer = command_encoder.finish();

    queue.submit([command_buffer]);

    // copy input buffer to device
    // let command_encoder_desc = wgpu::CommandEncoderDescriptor {
    //     label: Some("copy_input_buffer"),
    // };
    // let command_buffer = device.create_command_encoder(&command_encoder_desc);

    // command_buffer.

    Ok(())
}

fn main() -> Result<()> {
    pollster::block_on(run_gpu())?;

    Ok(())
}
