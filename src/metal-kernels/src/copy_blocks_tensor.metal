#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

template <typename T>
[[kernel]] void copy_blocks_tensor(
    tensor<device T, dextents<uint, 1>> key_cache [[buffer(0)]],
    tensor<device T, dextents<uint, 1>> value_cache [[buffer(1)]],
    const device int64_t* block_mapping [[buffer(2)]],
    device const uint& numel_per_block [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint pair_idx = gid;

    const int64_t src_block_number = block_mapping[2 * pair_idx];
    const int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

    const uint src_block_offset = uint(src_block_number) * numel_per_block;
    const uint dst_block_offset = uint(dst_block_number) * numel_per_block;

    for (uint i = tid; i < numel_per_block; i += threads_per_threadgroup) {
        array<uint, 1> src_idx = { src_block_offset + i };
        array<uint, 1> dst_idx = { dst_block_offset + i };
        key_cache[dst_idx] = key_cache[src_idx];
        value_cache[dst_idx] = value_cache[src_idx];
    }
}

template [[host_name("copy_blocks_tensor_float")]]
[[kernel]] void copy_blocks_tensor<float>(
    tensor<device float, dextents<uint, 1>> key_cache [[buffer(0)]],
    tensor<device float, dextents<uint, 1>> value_cache [[buffer(1)]],
    const device int64_t* block_mapping [[buffer(2)]],
    device const uint& numel_per_block [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]);
