#![cfg(target_os = "macos")]

use candle_core::DType;
use metal::{Buffer, Device, MTLResourceOptions};
use metal_kernels::{call_copy_blocks, call_copy_blocks_metal4, Kernels};

fn new_buffer_from_slice<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let size = std::mem::size_of_val(data) as u64;
    let ptr = data.as_ptr() as *const std::ffi::c_void;
    device.new_buffer_with_data(ptr, size, MTLResourceOptions::StorageModeShared)
}

fn read_buffer_as_vec_f32(buffer: &Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[test]
fn copy_blocks_metal4_matches_metal3() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let num_blocks = 3usize;
    let numel_per_block = 8usize;
    let total = num_blocks * numel_per_block;

    let mut key_init = Vec::with_capacity(total);
    let mut value_init = Vec::with_capacity(total);
    for i in 0..total {
        key_init.push(i as f32 + 1.0);
        value_init.push(1000.0 + i as f32);
    }

    // One pair: copy block 0 into block 2.
    let block_mapping: [i64; 2] = [0, 2];

    let queue = device.new_command_queue();
    let kernels = Kernels::default();

    let key_m3 = new_buffer_from_slice(&device, &key_init);
    let value_m3 = new_buffer_from_slice(&device, &value_init);
    let map_m3 = new_buffer_from_slice(&device, &block_mapping);
    let cb_m3 = queue.new_command_buffer();
    call_copy_blocks(
        &device,
        cb_m3,
        kernels,
        DType::F32,
        &key_m3,
        0,
        &value_m3,
        0,
        &map_m3,
        0,
        1,
        numel_per_block as u64,
    )
    .expect("metal3 copy_blocks should succeed");
    cb_m3.commit();
    cb_m3.wait_until_completed();

    let key_m4 = new_buffer_from_slice(&device, &key_init);
    let value_m4 = new_buffer_from_slice(&device, &value_init);
    let map_m4 = new_buffer_from_slice(&device, &block_mapping);
    let cb_m4 = queue.new_command_buffer();
    call_copy_blocks_metal4(
        &device,
        cb_m4,
        kernels,
        DType::F32,
        &key_m4,
        0,
        &value_m4,
        0,
        &map_m4,
        0,
        1,
        numel_per_block as u64,
    )
    .expect("metal4 copy_blocks should succeed");
    cb_m4.commit();
    cb_m4.wait_until_completed();

    let got_key_m3 = read_buffer_as_vec_f32(&key_m3, total);
    let got_val_m3 = read_buffer_as_vec_f32(&value_m3, total);
    let got_key_m4 = read_buffer_as_vec_f32(&key_m4, total);
    let got_val_m4 = read_buffer_as_vec_f32(&value_m4, total);

    std::env::set_var("ATTENTION_RS_METAL4_COPY_BLOCKS_MODE", "tensor_msl");
    let key_m4_tensor = new_buffer_from_slice(&device, &key_init);
    let value_m4_tensor = new_buffer_from_slice(&device, &value_init);
    let map_m4_tensor = new_buffer_from_slice(&device, &block_mapping);
    let cb_m4_tensor = queue.new_command_buffer();
    let tensor_msl_result = call_copy_blocks_metal4(
        &device,
        cb_m4_tensor,
        kernels,
        DType::F32,
        &key_m4_tensor,
        0,
        &value_m4_tensor,
        0,
        &map_m4_tensor,
        0,
        1,
        numel_per_block as u64,
    );

    assert_eq!(got_key_m4, got_key_m3, "key cache output mismatch");
    assert_eq!(got_val_m4, got_val_m3, "value cache output mismatch");
    assert!(
        tensor_msl_result.is_err(),
        "tensor_msl should fail fast until tensor argument binding exists"
    );
    std::env::remove_var("ATTENTION_RS_METAL4_COPY_BLOCKS_MODE");

}

#[test]
fn copy_blocks_metal4_tensor_perf_smoke() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let num_blocks = 256usize;
    let numel_per_block = 256usize;
    let total = num_blocks * numel_per_block;

    let mut key_init = Vec::with_capacity(total);
    let mut value_init = Vec::with_capacity(total);
    for i in 0..total {
        key_init.push(i as f32 + 1.0);
        value_init.push(10000.0 + i as f32);
    }

    let mut map = Vec::with_capacity(512);
    for i in 0..256i64 {
        map.push(i);
        map.push((255 - i) as i64);
    }

    let queue = device.new_command_queue();
    let kernels = Kernels::default();

    let run = |mode: &str| {
        std::env::set_var("ATTENTION_RS_METAL4_COPY_BLOCKS_MODE", mode);
        let key_buf = new_buffer_from_slice(&device, &key_init);
        let value_buf = new_buffer_from_slice(&device, &value_init);
        let map_buf = new_buffer_from_slice(&device, &map);

        let warmup = queue.new_command_buffer();
        call_copy_blocks_metal4(
            &device,
            warmup,
            kernels,
            DType::F32,
            &key_buf,
            0,
            &value_buf,
            0,
            &map_buf,
            0,
            (map.len() / 2) as u64,
            numel_per_block as u64,
        )
        .expect("warmup should succeed");
        warmup.commit();
        warmup.wait_until_completed();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let cb = queue.new_command_buffer();
            call_copy_blocks_metal4(
                &device,
                cb,
                kernels,
                DType::F32,
                &key_buf,
                0,
                &value_buf,
                0,
                &map_buf,
                0,
                (map.len() / 2) as u64,
                numel_per_block as u64,
            )
            .expect("run should succeed");
            cb.commit();
            cb.wait_until_completed();
        }
        start.elapsed()
    };

    let t_kernel = run("kernel");
    let t_tensor = run("tensor");
    let ratio = t_tensor.as_secs_f64() / t_kernel.as_secs_f64();
    println!(
        "copy_blocks perf-smoke: kernel={t_kernel:?} tensor_api={t_tensor:?} ratio_api={ratio:.3}"
    );

    // Phase-1 guard: detect catastrophic regressions only.
    assert!(
        ratio <= 20.0,
        "mtltensor copy path is catastrophically slower in smoke test: ratio={ratio:.3}"
    );
}
