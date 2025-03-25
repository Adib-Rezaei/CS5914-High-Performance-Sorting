import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_results(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

filenames = {
    "Baseline": "radix_sort_results_v100.pkl",
    "Opt1 256 Threads": "radix_sort_results_v100_opt1.pkl",
    "Baseline 512 Threads": "radix_sort_results_v100_baseline_512t.pkl",
    "Opt1 512 Threads": "radix_sort_results_v100_opt1_512t.pkl",
    "Opt1 1024 Threads": "radix_sort_results_v100_opt1_1024t.pkl",
    "Opt2 256 Threads": "radix_sort_results_v100_opt1_try_256t.pkl",
    "Opt2 1024 Threads": "radix_sort_results_v100_opt1_try_1024t.pkl",
    "Optimized": "radix_sort_results_v100_optimized_new.pkl"
}

results = {key: load_results(file) for key, file in filenames.items()}

sizes, gpu_times, memory_usages = {}, {}, {}
for key, res in results.items():
    sizes[key] = [r["size"] for r in res][1:]
    gpu_times[key] = [r["gpu_time"] / (1000 if "Optimized" in key else 1) for r in res][1:]
    memory_usages[key] = [r["memory_usage"] / (1024**2) for r in res][1:]

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
styles = {
    "Baseline": ("s", "--", "tab:blue"),
    "Baseline 512 Threads": (">", "-", "tab:brown"),
    "Opt1 256 Threads": ("o", "-", "tab:cyan"),
    "Opt1 512 Threads": ("<", "-", "tab:purple"),
    "Opt1 1024 Threads": ("v", "-", "tab:gray"),
    "Opt2 256 Threads": ("H", "-", "tab:olive"),
    "Opt2 1024 Threads": ("*", "-", "tab:orange"),
    "Optimized": ("^", ":", "tab:red")
}

for key, (marker, linestyle, color) in styles.items():
    axes[0].plot(sizes[key], gpu_times[key], label=f"{key} GPU Time", marker=marker, linestyle=linestyle, color=color)
    axes[1].plot(sizes[key], memory_usages[key], label=f"{key} Memory Usage (MB)", marker=marker, linestyle=linestyle, color=color)

axes[0].set_yscale("log")

for ax, ylabel, title in zip(axes, ["Time (seconds)", "Memory Usage (MB)"],
                             ["GPU Execution Time for Radix Sort", "Memory Usage for Radix Sort"]):
    ax.set_xlabel("Array Size")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(sizes["Baseline"]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.legend()
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.show()
