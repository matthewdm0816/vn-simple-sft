import torch
for _ in range(500000):
    print(" ---- ---- ")
    x = torch.randn(2088, 1536, dtype=torch.float, device="cuda")
    y = torch.mean(x, 1)
