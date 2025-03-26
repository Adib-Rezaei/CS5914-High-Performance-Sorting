import numpy as np
import numba
from numba import cuda
import time
import pickle
from mpi4py import MPI

from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

@cuda.jit
def counting_sort_kernel(arr, output, count, exp, size):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid < size:
        index = (arr[tid] // exp) % 10
        cuda.atomic.add(count, index, 1)
    cuda.syncthreads()

    if tid < 10 and tid > 0:
        count[tid] += count[tid - 1]
    cuda.syncthreads()

    if tid < size:
        index = (arr[tid] // exp) % 10
        pos = count[index] - 1
        output[pos] = arr[tid]
        cuda.atomic.sub(count, index, 1)
    cuda.syncthreads()

def parallel_radix_sort_gpu(arr, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_size = arr.size // size
    local_arr = np.empty(local_size, dtype=np.int32)

    comm.Scatter(arr, local_arr, root=0)

    d_arr = cuda.to_device(local_arr)
    d_output = cuda.device_array_like(local_arr)
    max_num = comm.allreduce(local_arr.max(), op=MPI.MAX)

    threads_per_block = 256
    blocks_per_grid = (local_size + threads_per_block - 1) // threads_per_block

    exp = 1
    while max_num // exp > 0:
        d_count = cuda.to_device(np.zeros(10, dtype=np.int32))
        counting_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_output, d_count, exp, local_size)
        d_arr.copy_to_device(d_output)
        exp *= 10
    cuda.synchronize()

    local_sorted = d_arr.copy_to_host()
    global_sorted = None

    if rank == 0:
        global_sorted = np.empty(arr.size, dtype=np.int32)

    comm.Gather(local_sorted, global_sorted, root=0)

    if rank == 0:
        return global_sorted
    return None

comm = MPI.COMM_WORLD
results = []
sizes = [2**i * 10**6 for i in range(10)]

for size in sizes:
    if comm.Get_rank() == 0:
        arr = np.random.randint(0, 10000, size, dtype=np.int32)
    else:
        arr = None

    start = time.time()
    sorted_gpu = parallel_radix_sort_gpu(arr, comm)
    end = time.time()
    gpu_time = end - start

    if comm.Get_rank() == 0:
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

if comm.Get_rank() == 0:
    with open("radix_sort_results.pkl", "wb") as f:
        pickle.dump(results, f)