import numpy as np
import numba
from numba import cuda
import time
import pickle
import cupy as cp
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

@cuda.jit
def counting_sort_kernel(arr, output, count, exp, size, temp_storage):
    shared_count = cuda.shared.array(shape=10, dtype=numba.int32)
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    local_tid = cuda.threadIdx.x

    if local_tid < 10:
        shared_count[local_tid] = 0
    cuda.syncthreads()

    if tid < size:
        index = (arr[tid] // exp) % 10
        cuda.atomic.add(shared_count, index, 1)
    cuda.syncthreads()

    if local_tid < 10:
        cuda.atomic.add(count, local_tid, shared_count[local_tid])
    cuda.syncthreads()

    if tid < 10 and tid > 0:
        count[tid] += count[tid - 1]
    cuda.syncthreads()

    if tid < size:
        index = (arr[tid] // exp) % 10
        pos = cuda.atomic.sub(count, index, 1) - 1
        temp_storage[tid] = pos
    cuda.syncthreads()

    if tid < size:
        output[temp_storage[tid]] = arr[tid]

@cuda.jit
def init_rng(rng_states, seed):
    tid = cuda.grid(1)
    if tid < rng_states.shape[0]:
        cuda.random.init(rng_states, seed)

def radix_sort_gpu(arr):
    size = arr.size
    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)
    d_temp_storage = cuda.device_array(size, dtype=np.int32)
    max_num = arr.max()

    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    start = time.time()
    exp = 1
    while max_num // exp > 0:
        d_count = cuda.to_device(np.zeros(10, dtype=np.int32))
        counting_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_output, d_count, exp, size, d_temp_storage)
        d_arr, d_output = d_output, d_arr
        exp *= 10
    cuda.synchronize()
    end = time.time()

    sorted_arr = d_arr.copy_to_host()
    return sorted_arr, end - start

results = []
sizes = [2**i * 10**6 for i in range(10)]

for size in sizes:
    rng_states = create_xoroshiro128p_states(256, seed=1)
    arr = cp.random.randint(0, 10000, size, dtype=cp.int32).get()
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