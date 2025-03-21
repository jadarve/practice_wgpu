use pollster;
use wgpu;

async fn get_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    // try to find a discrete GPU
    // let adapter = instance
    //     .enumerate_adapters(wgpu::Backends::all())
    //     .iter()
    //     .find(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu);

    // if let Some(adapter) = adapter {
    //     return Some(adapter.clone());
    // }

    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: true,
    };

    instance.request_adapter(&options).await
}

async fn run() {
    // Create an instance of the default adapter
    let instance = wgpu::Instance::default();

    // for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
    //     println!("Adapter: {:?}\n", adapter.get_info());
    // }

    if let Some(adapter) = get_adapter(&instance).await {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device_info_label"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        println!("Adapter: {:?}\n", adapter.get_info());
        println!("Features: {:?}\n", adapter.features());
    } else {
        println!("No adapter found");
    }
}

fn main() {
    pollster::block_on(run());
}
