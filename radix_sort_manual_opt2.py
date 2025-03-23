import numpy as np
import numba
from numba import cuda
import time
import pickle
import math

@cuda.jit
def counting_sort_kernel(d_arr, d_output, exp):
    shared_count = cuda.shared.array(10, numba.int32)

    tid = cuda.grid(1)
    lane = cuda.threadIdx.x

    if lane < 10:
        shared_count[lane] = 0
    cuda.syncthreads()

    if tid < d_arr.size:
        digit = (d_arr[tid] // exp) % 10
        cuda.atomic.add(shared_count, digit, 1)
    cuda.syncthreads()

    if lane == 0:
        for i in range(1, 10):
            shared_count[i] += shared_count[i - 1]
    cuda.syncthreads()

    if tid < d_arr.size:
        digit = (d_arr[tid] // exp) % 10
        pos = cuda.atomic.sub(shared_count, digit, 1) - 1
        d_output[pos] = d_arr[tid]

def radix_sort_gpu(arr):
    n = arr.size
    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)

    max_element = np.max(arr)
    exp = 1

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)

    while max_element // exp > 0:
        counting_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_output, exp)
        d_arr, d_output = d_output, d_arr  # Swap references
        exp *= 10

    end_event.record()
    end_event.synchronize()
    gpu_time = cuda.event_elapsed_time(start_event, end_event) / 1000.0

    sorted_arr = d_arr.copy_to_host()
    return sorted_arr, gpu_time

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

