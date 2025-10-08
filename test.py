import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import torch
x = torch.randn(10000, 10000, device='cuda')
print(x.sum())
