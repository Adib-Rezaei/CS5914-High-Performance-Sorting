import numpy as np
import numba
from numba import cuda
import time
import pickle

from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

@cuda.jit
def counting_sort_kernel(arr, output, count, exp, size):
    shared_count = cuda.shared.array(10, dtype=numba.int32)

    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tx = cuda.threadIdx.x

    if tx < 10:
        shared_count[tx] = 0
    cuda.syncthreads()

    if tid < size:
        index = (arr[tid] // exp) % 10
        cuda.atomic.add(shared_count, index, 1)
    cuda.syncthreads()

    # Parallel scan (prefix sum)
    if tx == 0:
        for i in range(1, 10):
            shared_count[i] += shared_count[i - 1]
    cuda.syncthreads()

    if tid < size:
        index = (arr[tid] // exp) % 10
        pos = cuda.atomic.sub(shared_count, index, 1) - 1
        output[pos] = arr[tid]
    cuda.syncthreads()


def radix_sort_gpu(arr):
    size = arr.size
    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)
    max_num = arr.max()
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    start = time.time()
    exp = 1
    while max_num // exp > 0:
        d_count = cuda.device_array(10, dtype=np.int32)
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
