import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def test_ddp(rank, world_size):
    setup(rank, world_size)

    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 10).to(rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    cleanup()
    print(f"DDP test completed on rank {rank}")

def test_amp():
    model = resnet18().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()

    for _ in range(10):
        optimizer.zero_grad()
        with autocast():
            output = model(torch.randn(32, 3, 224, 224).cuda())
            loss = F.mse_loss(output, torch.randn_like(output))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print("Mixed precision training test completed")

def test_torchscript():
    model = SimpleModel()
    example_input = torch.randn(10, 10)
    
    # Tracing
    traced_model = torch.jit.trace(model, example_input)
    traced_output = traced_model(example_input)
    
    # Scripting
    scripted_model = torch.jit.script(model)
    scripted_output = scripted_model(example_input)

    print("TorchScript test completed")

def test_dataparallel():
    model = resnet18()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()

    input_tensor = torch.randn(32, 3, 224, 224).cuda()
    output = model(input_tensor)

    print("DataParallel test completed")

# Custom CUDA kernel
custom_kernel = load(
    name="custom_kernel",
    sources=["custom_kernel.cpp", "custom_kernel_cuda.cu"],
    verbose=True
)

def test_custom_cuda_kernel():
    a = torch.randn(100, device='cuda')
    b = torch.randn(100, device='cuda')
    c = custom_kernel.add(a, b)
    print("Custom CUDA kernel test completed")

if __name__ == "__main__":
    # Test DistributedDataParallel
    world_size = torch.cuda.device_count()
    mp.spawn(test_ddp, args=(world_size,), nprocs=world_size, join=True)

    # Test Mixed Precision Training
    test_amp()

    # Test TorchScript
    test_torchscript()

    # Test DataParallel
    test_dataparallel()

    # Test Custom CUDA Kernel
    test_custom_cuda_kernel()
