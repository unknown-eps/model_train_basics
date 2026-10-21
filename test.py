import torch

# arange is deterministic: it counts 0, 1, 2, ... so no seed is needed here.
x = torch.arange(24).reshape(2, 3, 4)
print("Original tensor (2, 3, 4):")
print(x)
print("Shape:", x.shape)
print("Strides:", x.stride())

# transpose(0, 1) swaps dim 0 and dim 1: (2, 3, 4) -> (3, 2, 4).
# Element [i, j, k] moves to position [j, i, k].
x_t = x.transpose(0, 1)
print("\nTransposed tensor (3, 2, 4):")
print(x_t)
print("Shape:", x_t.shape)
print("Strides:", x_t.stride())

# A transpose is a view: same storage, only the strides changed.
print("\nShares storage with original:", x_t.data_ptr() == x.data_ptr())
print("Contiguous:", x_t.is_contiguous())
