#![cfg(target_os = "macos")]

use candle_core::DType;
use metal::{
    Buffer, ComputePassDescriptor, ComputePassDescriptorRef, CounterSampleBuffer,
    CounterSampleBufferDescriptor, Device, MTLCounterSamplingPoint, MTLResourceOptions,
    MTLStorageMode, NSRange,
};
use metal_kernels::{call_copy_blocks_metal4, Kernels};
use std::time::Instant;

const NUM_SAMPLES: u64 = 2;
const WARMUP_ITERS: usize = 20;
const MEASURE_ITERS: usize = 100;

fn new_buffer_from_slice<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let size = std::mem::size_of_val(data) as u64;
    let ptr = data.as_ptr() as *const std::ffi::c_void;
    device.new_buffer_with_data(ptr, size, MTLResourceOptions::StorageModeShared)
}

fn percentile_us(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn create_counter_sample_buffer(device: &Device) -> CounterSampleBuffer {
    let desc = CounterSampleBufferDescriptor::new();
    desc.set_storage_mode(MTLStorageMode::Shared);
    desc.set_sample_count(NUM_SAMPLES);
    let counter_sets = device.counter_sets();
    let timestamp_counter = counter_sets
        .iter()
        .find(|cs| cs.name() == "timestamp")
        .expect("No timestamp counter set found");
    desc.set_counter_set(timestamp_counter);
    device
        .new_counter_sample_buffer_with_descriptor(&desc)
        .expect("failed to create counter sample buffer")
}

fn setup_compute_pass_for_timestamps(
    compute_pass_descriptor: &ComputePassDescriptorRef,
    counter_sample_buffer: &CounterSampleBuffer,
) {
    let attachment = compute_pass_descriptor
        .sample_buffer_attachments()
        .object_at(0)
        .expect("sample attachment 0 should exist");
    attachment.set_sample_buffer(counter_sample_buffer);
    attachment.set_start_of_encoder_sample_index(0);
    attachment.set_end_of_encoder_sample_index(1);
}

fn resolve_samples_into_buffer(
    command_buffer: &metal::CommandBufferRef,
    counter_sample_buffer: &CounterSampleBuffer,
    destination_buffer: &Buffer,
) {
    let blit = command_buffer.new_blit_command_encoder();
    blit.resolve_counters(
        counter_sample_buffer,
        NSRange::new(0u64, NUM_SAMPLES),
        destination_buffer,
        0u64,
    );
    blit.end_encoding();
}

fn gpu_pass_us_from_samples(
    sample_buffer: &Buffer,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> f64 {
    let samples =
        unsafe { std::slice::from_raw_parts(sample_buffer.contents().cast::<u64>(), NUM_SAMPLES as usize) };
    let pass_start = samples[0];
    let pass_end = samples[1];
    let pass_span = pass_end.saturating_sub(pass_start) as f64;
    let gpu_span = gpu_end.saturating_sub(gpu_start) as f64;
    let cpu_span = cpu_end.saturating_sub(cpu_start) as f64;
    if pass_span <= 0.0 || gpu_span <= 0.0 || cpu_span <= 0.0 {
        return 0.0;
    }
    let ns = pass_span / gpu_span * cpu_span;
    ns / 1000.0
}

#[test]
fn copy_blocks_low_level_metrics() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let num_blocks = 256usize;
    let numel_per_block = 256usize;
    let num_pairs = 256usize;
    let dtype = DType::F32;
    let elem_bytes = dtype.size_in_bytes();
    let total = num_blocks * numel_per_block;
    let bytes_per_iter = (num_pairs * numel_per_block * elem_bytes * 2 * 2) as f64;

    let mut key_init = Vec::with_capacity(total);
    let mut value_init = Vec::with_capacity(total);
    for i in 0..total {
        key_init.push(i as f32 + 1.0);
        value_init.push(10000.0 + i as f32);
    }

    let mut map = Vec::with_capacity(num_pairs * 2);
    for i in 0..num_pairs as i64 {
        map.push(i);
        map.push((num_pairs as i64 - 1) - i);
    }

    let queue = device.new_command_queue();
    let kernels = Kernels::default();
    let pipeline = kernels
        .load_pipeline(&device, "copy_blocks_float".to_string())
        .expect("pipeline should load");

    println!(
        "pipeline: thread_execution_width={} max_threads_per_tg={}",
        pipeline.thread_execution_width(),
        pipeline.max_total_threads_per_threadgroup()
    );
    println!(
        "dispatch: tg_count={} tg_size={} bytes_per_iter={}",
        num_pairs,
        numel_per_block.min(1024),
        bytes_per_iter as u64
    );

    // Kernel mode: collect true GPU pass timestamps via counter sample buffer.
    std::env::set_var("ATTENTION_RS_METAL4_COPY_BLOCKS_MODE", "kernel");
    let counter_sampling_point = MTLCounterSamplingPoint::AtStageBoundary;
    let supports_counter_sampling = device.supports_counter_sampling(counter_sampling_point);
    println!("counter_sampling_supported={supports_counter_sampling}");

    let key_kernel = new_buffer_from_slice(&device, &key_init);
    let value_kernel = new_buffer_from_slice(&device, &value_init);
    let map_kernel = new_buffer_from_slice(&device, &map);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        call_copy_blocks_metal4(
            &device,
            cb,
            kernels,
            dtype,
            &key_kernel,
            0,
            &value_kernel,
            0,
            &map_kernel,
            0,
            num_pairs as u64,
            numel_per_block as u64,
        )
        .expect("kernel warmup should succeed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut kernel_cpu_us = Vec::with_capacity(MEASURE_ITERS);
    let mut kernel_gpu_pass_us = Vec::with_capacity(MEASURE_ITERS);

    let counter_sample_buffer = if supports_counter_sampling {
        Some(create_counter_sample_buffer(&device))
    } else {
        None
    };
    let resolved_samples = counter_sample_buffer.as_ref().map(|_| {
        device.new_buffer(
            (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    for _ in 0..MEASURE_ITERS {
        let mut cpu0 = 0u64;
        let mut gpu0 = 0u64;
        device.sample_timestamps(&mut cpu0, &mut gpu0);
        let t0 = Instant::now();

        let cb = queue.new_command_buffer();
        if let (Some(csb), Some(sample_buf)) = (counter_sample_buffer.as_ref(), resolved_samples.as_ref()) {
            let pass_desc = ComputePassDescriptor::new();
            setup_compute_pass_for_timestamps(pass_desc, csb);
            let enc = cb.compute_command_encoder_with_descriptor(pass_desc);
            call_copy_blocks_metal4(
                &device,
                enc,
                kernels,
                dtype,
                &key_kernel,
                0,
                &value_kernel,
                0,
                &map_kernel,
                0,
                num_pairs as u64,
                numel_per_block as u64,
            )
            .expect("kernel run should succeed");
            enc.end_encoding();
            resolve_samples_into_buffer(cb, csb, sample_buf);
        } else {
            call_copy_blocks_metal4(
                &device,
                cb,
                kernels,
                dtype,
                &key_kernel,
                0,
                &value_kernel,
                0,
                &map_kernel,
                0,
                num_pairs as u64,
                numel_per_block as u64,
            )
            .expect("kernel run should succeed");
        }
        cb.commit();
        cb.wait_until_completed();

        let cpu_us = t0.elapsed().as_secs_f64() * 1e6;
        kernel_cpu_us.push(cpu_us);

        let mut cpu1 = 0u64;
        let mut gpu1 = 0u64;
        device.sample_timestamps(&mut cpu1, &mut gpu1);
        if let Some(sample_buf) = resolved_samples.as_ref() {
            kernel_gpu_pass_us.push(gpu_pass_us_from_samples(sample_buf, cpu0, cpu1, gpu0, gpu1));
        }
    }

    // Tensor mode: collect CPU timing of encoded+submitted work.
    std::env::set_var("ATTENTION_RS_METAL4_COPY_BLOCKS_MODE", "tensor");
    let key_tensor = new_buffer_from_slice(&device, &key_init);
    let value_tensor = new_buffer_from_slice(&device, &value_init);
    let map_tensor = new_buffer_from_slice(&device, &map);

    for _ in 0..WARMUP_ITERS {
        let cb = queue.new_command_buffer();
        call_copy_blocks_metal4(
            &device,
            cb,
            kernels,
            dtype,
            &key_tensor,
            0,
            &value_tensor,
            0,
            &map_tensor,
            0,
            num_pairs as u64,
            numel_per_block as u64,
        )
        .expect("tensor warmup should succeed");
        cb.commit();
        cb.wait_until_completed();
    }

    let mut tensor_cpu_us = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let t0 = Instant::now();
        let cb = queue.new_command_buffer();
        call_copy_blocks_metal4(
            &device,
            cb,
            kernels,
            dtype,
            &key_tensor,
            0,
            &value_tensor,
            0,
            &map_tensor,
            0,
            num_pairs as u64,
            numel_per_block as u64,
        )
        .expect("tensor run should succeed");
        cb.commit();
        cb.wait_until_completed();
        tensor_cpu_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let kernel_cpu_p50 = percentile_us(&kernel_cpu_us, 0.50);
    let kernel_cpu_p95 = percentile_us(&kernel_cpu_us, 0.95);
    let tensor_cpu_p50 = percentile_us(&tensor_cpu_us, 0.50);
    let tensor_cpu_p95 = percentile_us(&tensor_cpu_us, 0.95);
    let tensor_vs_kernel = if kernel_cpu_p50 > 0.0 {
        tensor_cpu_p50 / kernel_cpu_p50
    } else {
        0.0
    };

    println!(
        "kernel: cpu_p50_us={kernel_cpu_p50:.2} cpu_p95_us={kernel_cpu_p95:.2}"
    );
    if !kernel_gpu_pass_us.is_empty() {
        let kernel_gpu_p50 = percentile_us(&kernel_gpu_pass_us, 0.50);
        let kernel_gpu_p95 = percentile_us(&kernel_gpu_pass_us, 0.95);
        let gb = bytes_per_iter / 1e9;
        let kernel_gpu_gbps = if kernel_gpu_p50 > 0.0 {
            gb / (kernel_gpu_p50 / 1e6)
        } else {
            0.0
        };
        println!(
            "kernel: gpu_pass_p50_us={kernel_gpu_p50:.2} gpu_pass_p95_us={kernel_gpu_p95:.2} gpu_pass_gbps={kernel_gpu_gbps:.2}"
        );
    }
    println!(
        "tensor: cpu_p50_us={tensor_cpu_p50:.2} cpu_p95_us={tensor_cpu_p95:.2} ratio_vs_kernel_p50={tensor_vs_kernel:.2}"
    );

    std::env::remove_var("ATTENTION_RS_METAL4_COPY_BLOCKS_MODE");
}
