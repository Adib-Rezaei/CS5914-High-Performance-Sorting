import numpy as np
import time
import psutil
import os
import pickle

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n  # Output array
    count = [0] * 10  # Count array for digits (0-9)

    # Count occurrences of digits in the current place value (exp)
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    # Update count[i] to store the position of the digit in the output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1

    # Copy the output array back to the original array
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    # Find the maximum number to determine the number of digits
    max_num = max(arr)
    exp = 1  # Initialize exponent to start sorting from the least significant digit
    
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10  # Move to the next digit place

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Create arrays of different sizes and sort them
results = []
sizes = [2**i * 10**6 for i in range(10)]
for size in sizes:
    arr = np.random.randint(0, 10000, size, dtype=np.int32)
    
    start_time = time.time()
    radix_sort(arr)
    end_time = time.time()
    
    cpu_time = end_time - start_time
    approx_flops = size * np.log10(max(arr)) if len(arr) > 0 else 0  # Approximate FLOPS calculation
    memory_usage = get_memory_usage()
    
    print(f"Sorted array size: {size}, CPU Time: {cpu_time:.4f} sec, Approx FLOPS: {approx_flops:.2f}, Memory Usage: {memory_usage:.2f} MB")

    results.append({
        "size": int(size),
        # "cpu_time": float(cpu_time),
        "gpu_time": float(cpu_time),
        "flops": float(approx_flops),
        "memory_usage": int(memory_usage)
    })

with open("radix_sort_results.pkl", "wb") as f:
    pickle.dump(results, f)


















# import numpy as np

# def counting_sort(arr, exp):
#     n = len(arr)
#     output = [0] * n  # Output array
#     count = [0] * 10  # Count array for digits (0-9)

#     # Count occurrences of digits in the current place value (exp)
#     for i in range(n):
#         index = (arr[i] // exp) % 10
#         count[index] += 1

#     # Update count[i] to store the position of the digit in the output array
#     for i in range(1, 10):
#         count[i] += count[i - 1]

#     # Build the output array
#     i = n - 1
#     while i >= 0:
#         index = (arr[i] // exp) % 10
#         output[count[index] - 1] = arr[i]
#         count[index] -= 1
#         i -= 1

#     # Copy the output array back to the original array
#     for i in range(n):
#         arr[i] = output[i]

# def radix_sort(arr):
#     # Find the maximum number to determine the number of digits
#     max_num = max(arr)
#     exp = 1  # Initialize exponent to start sorting from the least significant digit
    
#     while max_num // exp > 0:
#         counting_sort(arr, exp)
#         exp *= 10  # Move to the next digit place

# # Create arrays of different sizes and sort them
# sizes = [2**i * 10**6 for i in range(10)]
# for size in sizes:
#     arr = np.random.randint(0, 10000, size, dtype=np.int32)
#     radix_sort(arr)
#     print("Sorted array size:", size)













# def counting_sort(arr, exp):
#     n = len(arr)
#     output = [0] * n  # Output array
#     count = [0] * 10  # Count array for digits (0-9)

#     # Count occurrences of digits in the current place value (exp)
#     for i in range(n):
#         index = (arr[i] // exp) % 10
#         count[index] += 1

#     # Update count[i] to store the position of the digit in the output array
#     for i in range(1, 10):
#         count[i] += count[i - 1]

#     # Build the output array
#     i = n - 1
#     while i >= 0:
#         index = (arr[i] // exp) % 10
#         output[count[index] - 1] = arr[i]
#         count[index] -= 1
#         i -= 1

#     # Copy the output array back to the original array
#     for i in range(n):
#         arr[i] = output[i]

# def radix_sort(arr):
#     # Find the maximum number to determine the number of digits
#     max_num = max(arr)
#     exp = 1  # Initialize exponent to start sorting from the least significant digit
    
#     while max_num // exp > 0:
#         counting_sort(arr, exp)
#         exp *= 10  # Move to the next digit place

# # Example usage:
# arr = [170, 45, 75, 90, 802, 24, 2, 66]
# radix_sort(arr)
# print("Sorted array:", arr)
