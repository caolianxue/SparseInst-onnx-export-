import torch.nn.functional as F
import torch

a = torch.rand([1,2,2])
print(a)
print(a.sum([1,2]))