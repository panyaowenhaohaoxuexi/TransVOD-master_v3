import torch
from util.misc_multi_3m import nested_tensor_from_tensor_list

B, K, H, W = 2, 3, 128, 128
samples = [torch.randn(9*(1+K), H, W) for _ in range(B)]
nt = nested_tensor_from_tensor_list(samples)  # split 默认 True
print(nt.tensors.shape)  # 期望 torch.Size([B*(1+K), 9, H, W]) -> [8, 9, 128, 128]
print(nt.mask.shape)     # 期望 torch.Size([8, 128, 128])
