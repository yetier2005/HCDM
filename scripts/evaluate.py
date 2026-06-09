#!/usr/bin/env python3
"""
Evaluate distilled dataset by training a classifier on it and testing on real data.

Usage:
    python scripts/evaluate.py --distilled ./outputs/hcdm_cifar10/distilled_data.pt \
                               --dataset cifar10 --arch resnet18 --epochs 1000
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_test_loader(dataset_name: str, batch_size: int = 128):
    """Build test dataloader for a dataset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # ResNet standard
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "imagenet1k":
        ds = datasets.ImageNet(root="./data/imagenet", split="val", transform=transform)
    elif dataset_name in ("imagewoof", "imagenette"):
        ds = datasets.ImageFolder(root=f"./data/{dataset_name}/val", transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def build_model(arch: str, num_classes: int):
    """Build a classifier."""
    if arch == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
    elif arch == "resnet10":
        # Lightweight for fast eval
        from torchvision.models.resnet import BasicBlock
        model = ResNetSmall(BasicBlock, [2, 2, 2], num_classes=num_classes)
    elif arch == "convnet":
        model = ConvNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


class ConvNet(nn.Module):
    """Simple ConvNet used in DD papers for CIFAR evaluation."""
    def __init__(self, num_classes=10, img_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNetSmall(nn.Module):
    """Small ResNet for fast evaluation."""
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        from torchvision.models.resnet import BasicBlock
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled dataset")
    parser.add_argument("--distilled", "-d", type=str, required=True,
                       help="Path to distilled_data.pt")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="Dataset name")
    parser.add_argument("--arch", type=str, default="convnet",
                       choices=["convnet", "resnet10", "resnet18"],
                       help="Classifier architecture")
    parser.add_argument("--epochs", type=int, default=1000,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--repeat", type=int, default=3,
                       help="Number of repeated evaluations")
    parser.add_argument("--real-baseline", action="store_true",
                       help="Also train on full real dataset for comparison")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load distilled data
    print(f"Loading distilled data from {args.distilled}...")
    data = torch.load(args.distilled, map_location="cpu")
    images = data["images"]  # [N, 3, H, W] in [-1, 1]
    labels = data["labels"]  # [N]
    ipc = data.get("ipc", len(images) // len(labels.unique()))
    print(f"  IPC: {ipc}")
    print(f"  Total images: {len(images)}")
    print(f"  Classes: {len(labels.unique())}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")

    num_classes = len(labels.unique())

    # Build distilled data loader
    ds_distilled = TensorDataset(images, labels)
    train_loader = DataLoader(ds_distilled, batch_size=min(args.batch_size, len(images)),
                              shuffle=True, drop_last=False)

    # Build test loader
    test_loader = build_test_loader(args.dataset)

    # Train and evaluate multiple times
    results = []
    print(f"\nTraining {args.arch} on distilled data ({args.epochs} epochs, {args.repeat} repeats)...")
    print("=" * 60)

    for r in range(args.repeat):
        model = build_model(args.arch, num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        best_acc = 0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc

            if (epoch + 1) % 200 == 0:
                print(f"  Repeat {r+1}, Epoch {epoch+1}: train_acc={train_acc:.4f}, "
                      f"test_acc={test_acc:.4f} (best={best_acc:.4f})")

        results.append(best_acc)
        print(f"  Repeat {r+1} best: {best_acc*100:.2f}%")

    # Summary
    results_t = torch.tensor(results)
    print("\n" + "=" * 60)
    print(f"Results ({args.arch}, IPC={ipc}, {args.dataset}):")
    print(f"  Mean:  {results_t.mean()*100:.2f}%")
    print(f"  Std:   {results_t.std()*100:.2f}%")
    print(f"  Max:   {results_t.max()*100:.2f}%")
    print(f"  All:   {[f'{r*100:.1f}%' for r in results]}")

    # Comparison with survey benchmarks
    print("\n" + "=" * 60)
    print("Reference (from survey Table I, ResNet-18):")
    if args.dataset == "cifar10":
        print(f"  Full dataset:     89.9%")
        print(f"  Minimax (CVPR'24): IPC=10 → 72.8%")
        print(f"  D3HR (ICML'25):    IPC=10 → high")
        print(f"  DAP (ICLR'26):     IPC=10 → 63.2%")
        print(f"  CoDA (ICLR'26):    IPC=10 → 64.9%")
    elif args.dataset == "cifar100":
        print(f"  Full dataset:     71.6%")
        print(f"  DACE (CVPR'24):    IPC=10 → 57.7%")
        print(f"  DC3 (TMLR'25):     IPC=10 → 57.4%")
        print(f"  DAP (ICLR'26):     IPC=10 → 50.6%")
        print(f"  EDITS (CVPR'25):   IPC=10 → 52.8%")

    # Real baseline (optional)
    if args.real_baseline:
        print(f"\n[Optional] Training on full real dataset...")
        real_ds = datasets.CIFAR10(root="./data", train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5]*3, [0.5]*3)]))
        real_loader = DataLoader(real_ds, batch_size=128, shuffle=True, num_workers=4)
        model = build_model(args.arch, num_classes).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        for ep in range(200):
            train_epoch(model, real_loader, opt, device)
            sched.step()
        real_acc = evaluate(model, test_loader, device)
        print(f"  Full real data ({args.arch}): {real_acc*100:.2f}%")


if __name__ == "__main__":
    main()
