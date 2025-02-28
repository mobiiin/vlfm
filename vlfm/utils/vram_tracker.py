import torch
from collections import defaultdict

class GPUMemoryTracker:
    def __init__(self):
        self.tensor_sizes = defaultdict(list)  # Stores sizes of tensors over time
        self.tensor_memory = defaultdict(list)  # Stores memory usage of tensors over time

    def track_tensor(self, name, tensor):
        """
        Track the size and memory usage of a tensor.
        """
        if tensor.is_cuda:
            size = tensor.element_size() * tensor.numel()  # Size in bytes
            self.tensor_sizes[name].append(size / 1e6)  # Convert to MB
            self.tensor_memory[name].append(torch.cuda.memory_allocated() / 1e6)  # Convert to MB

    def print_summary(self):
        """
        Print a summary of tensor sizes and memory usage over time.
        """
        print("Tensor Size Summary (MB):")
        for name, sizes in self.tensor_sizes.items():
            print(f"{name}: {sizes}")

        print("\nTensor Memory Usage Summary (MB):")
        for name, memory in self.tensor_memory.items():
            print(f"{name}: {memory}")