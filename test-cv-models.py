from six.moves import urllib

proxy = urllib.request.ProxyHandler({'http': 'localhost:17893', 'https': 'localhost:17893'})
# construct a new opener using your proxy settings
opener = urllib.request.build_opener(proxy)
# install the openen on the module-level
urllib.request.install_opener(opener)

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.models import resnet18, efficientnet_b0 # , deeplabv3_resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Image Classification

# 1.1 Simple: LeNet-5 on MNIST
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.reshape(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_lenet5():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=32)

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    print("LeNet-5 training completed")

# 1.2 Intermediate: ResNet-18 on CIFAR-10
def train_resnet18():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=32)

    model = resnet18(pretrained=False, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    print("ResNet-18 training completed")

# 1.3 Complex: EfficientNet-B0 on ImageNet
def test_efficientnet():
    model = efficientnet_b0(pretrained=True).to(device)
    model.eval()

    # Simulating an ImageNet input
    dummy_input = torch.randn(32, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print("EfficientNet-B0 inference completed")

# 2. Object Detection: YOLO v5
def test_yolov5():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    
    # Simulating a COCO input
    dummy_input = torch.randn(32, 3, 640, 640).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print("YOLO v5 inference completed")

# 3. Semantic Segmentation: DeepLabV3
def test_deeplabv3():
    model = deeplabv3_resnet50(pretrained=True, progress=True).to(device)
    model.eval()

    # Simulating a Pascal VOC input
    dummy_input = torch.randn(1, 3, 513, 513).to(device)
    with torch.no_grad():
        output = model(dummy_input)['out']

    print("DeepLabV3 inference completed")

# 4. Image Generation: StyleGAN2
def test_stylegan2():
    model = torch.hub.load('nvidia/StyleGAN2-ADA-PyTorch', 'synthesize').to(device)
    
    # Generate a random latent vector
    latent = torch.randn(1, 512).to(device)
    with torch.no_grad():
        image = model(latent)

    print("StyleGAN2 inference completed")

if __name__ == "__main__":
    train_lenet5()
    train_resnet18()
    test_efficientnet()
    test_yolov5()
    # test_deeplabv3()
    # test_stylegan2()
