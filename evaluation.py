import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_results(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

results_v100 = load_results("radix_sort_results_v100.pkl")
results_a100 = load_results("radix_sort_results_a100.pkl")
results_v100_optimized = load_results("radix_sort_results_v100_optimized.pkl")

sizes_v100 = [res["size"] for res in results_v100][1:]
gpu_times_v100 = [res["gpu_time"] for res in results_v100][1:]
memory_usages_v100 = [res["memory_usage"] / (1024**2) for res in results_v100][1:]

sizes_a100 = [res["size"] for res in results_a100][1:]
gpu_times_a100 = [res["gpu_time"] for res in results_a100][1:]
memory_usages_a100 = [res["memory_usage"] / (1024**2) for res in results_a100][1:]

sizes_v100_optimized = [res["size"] for res in results_v100_optimized][1:]
gpu_times_v100_optimized = [res["gpu_time"] / 1000 for res in results_v100_optimized][1:]
memory_usages_v100_optimized = [res["memory_usage"] / (1024**2) for res in results_v100_optimized][1:]

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].plot(sizes_v100, gpu_times_v100, label="V100 GPU Time", marker="s", linestyle="--", color="tab:blue")
axes[0].plot(sizes_a100, gpu_times_a100, label="A100 GPU Time", marker="o", linestyle="-", color="tab:cyan")
axes[0].plot(sizes_v100_optimized, gpu_times_v100_optimized, label="V100 Optimized GPU Time", marker="^", linestyle=":", color="tab:red")
axes[0].set_xlabel("Array Size")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
axes[0].set_xticks(sizes_v100)
axes[0].set_xticklabels([f"{s:.1e}" for s in sizes_v100], rotation=45)
axes[0].legend()
axes[0].set_title("GPU Execution Time for Radix Sort")
axes[0].grid(True)

axes[1].plot(sizes_v100, memory_usages_v100, label="V100 Memory Usage (MB)", marker="d", linestyle="--", color="tab:green")
axes[1].plot(sizes_a100, memory_usages_a100, label="A100 Memory Usage (MB)", marker="p", linestyle="-", color="tab:olive")
axes[1].plot(sizes_v100_optimized, memory_usages_v100_optimized, label="V100 Optimized Memory Usage (MB)", marker="h", linestyle=":", color="tab:orange")
axes[1].set_xlabel("Array Size")
axes[1].set_ylabel("Memory Usage (MB)")
axes[1].set_xscale("log")
axes[0].set_yscale("log")
axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
axes[1].set_xticks(sizes_v100)
axes[1].set_xticklabels([f"{s:.1e}" for s in sizes_v100], rotation=45)
axes[1].legend()
axes[1].set_title("Memory Usage for Radix Sort")
axes[1].grid(True)

plt.tight_layout()
plt.show()
