import numpy as np
import numba
from numba import cuda
import time
import pickle

BLOCK_SIZE = 1024
CACHE_BLOCK = 32768

from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

@cuda.jit
def counting_sort_kernel(arr, output, count, exp, size):
    shared_count = cuda.shared.array(10, dtype=numba.int32)
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    local_tid = cuda.threadIdx.x

    if local_tid < 10:
        shared_count[local_tid] = 0
    cuda.syncthreads()

    for block_start in range(0, size, CACHE_BLOCK):
        block_end = min(block_start + CACHE_BLOCK, size)

        if tid < block_end and tid >= block_start:
            index = (arr[tid] // exp) % 10
            cuda.atomic.add(shared_count, index, 1)
        cuda.syncthreads()

    if local_tid < 10:
        cuda.atomic.add(count, local_tid, shared_count[local_tid])
    cuda.syncthreads()

    if tid < 10 and tid > 0:
        count[tid] += count[tid - 1]
    cuda.syncthreads()

    for block_start in range(size - CACHE_BLOCK, -1, -CACHE_BLOCK):
        block_end = min(block_start + CACHE_BLOCK, size)

        if tid < block_end and tid >= block_start:
            index = (arr[tid] // exp) % 10
            pos = count[index] - 1
            output[pos] = arr[tid]
            cuda.atomic.sub(count, index, 1)
        cuda.syncthreads()

def radix_sort_gpu(arr):
    size = arr.size
    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)
    max_num = arr.max()
    threads_per_block = BLOCK_SIZE
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    start = time.time()
    exp = 1
    while max_num // exp > 0:
        d_count = cuda.to_device(np.zeros(10, dtype=np.int32))
        counting_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_output, d_count, exp, size)
        d_arr.copy_to_device(d_output)
        exp *= 10
    cuda.synchronize()
    end = time.time()

    sorted_arr = d_arr.copy_to_host()
    return sorted_arr, end - start

results = []
sizes = [2**i * 10**6 for i in range(10)]
for size in sizes:
    arr = np.random.randint(0, 10000, size, dtype=np.int32)
    sorted_gpu, gpu_time = radix_sort_gpu(arr.copy())

    flops = (size * np.log10(size)) / gpu_time
    memory_usage = arr.nbytes

    results.append({
        "size": int(size),
        "gpu_time": float(gpu_time),
        "flops": float(flops),
        "memory_usage": int(memory_usage)
    })

    print(f"Array size: {size}")
    print(f"GPU Time: {gpu_time:.6f} sec")
    print(f"Approx FLOPS: {flops:.2f}")
    print(f"Memory Usage: {memory_usage / (1024**2):.2f} MB\n")

with open("radix_sort_results.pkl", "wb") as f:
    pickle.dump(results, f)