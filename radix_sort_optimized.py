import cupy as cp
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

def parallel_radix_sort(arr):
    d_arr = cp.asarray(arr, dtype=cp.int32)

    mem_before = cp.cuda.Device(0).mem_info[0]  
    print(f"mem before: {mem_before}")

    cpu_start = time.time()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()

    sorted_arr = cp.sort(d_arr)

    end_event.record()
    end_event.synchronize()
    gpu_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)  

    cpu_end = time.time()
    cpu_time_s = cpu_end - cpu_start  

    mem_after = cp.cuda.Device(0).mem_info[0]  
    print(f"mem after: {mem_after}")
    mem_used = (mem_before - mem_after) / (1024 ** 2) 
    print(f"mem used: {mem_used}")

    n = d_arr.size
    flops = (n * cp.log2(n)).get()  

    return sorted_arr, gpu_time_ms, cpu_time_s, mem_used, flops

sizes = [2**i * 10**6 for i in range(10)]
gpu_times = []
cpu_times = []
memory_usage = []
flops_estimation = []
results = []
for n in sizes:
    print(f"Running sort for {n} elements...")
    h_arr = cp.random.randint(0, 1000000, size=n, dtype=cp.int32)
    _, gpu_time, cpu_time, mem_used, flops = parallel_radix_sort(h_arr)

    gpu_times.append(gpu_time)
    cpu_times.append(cpu_time)
    memory_usage.append(mem_used)
    flops_estimation.append(flops)

    results.append({
        "size": n,
        "gpu_time": float(gpu_time),
        "flops": float(flops),
        "memory_usage": int(mem_used)
    })

with open("radix_sort_results_v100_optimized.pkl", "wb") as f:
    pickle.dump(results, f)