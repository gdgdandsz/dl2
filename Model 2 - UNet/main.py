import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchvision.transforms as transforms
#import wandb

from models import UNet
from datasets import SegmentationDataSet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# 数据增强
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
])

# 创建数据集和数据加载器
train_set_path = '/scratch/xz3645/test/dl/Dataset_Student/train/video_'
val_set_path = '/scratch/xz3645/test/dl/Dataset_Student/val/video_'

train_data_dir = [f"{train_set_path}{i:05d}" for i in range(0, 1000)]
val_data_dir = [f"{val_set_path}{i:05d}" for i in range(1000, 2000)]

train_dataset = SegmentationDataSet(train_data_dir, transform=data_transforms)
val_dataset = SegmentationDataSet(val_data_dir, transform=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 模型、损失函数和优化器
model = UNet(n_channels=3, n_classes=49, bilinear=True).to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Training Loop with Early Stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(40):
    model.train()
    train_loss = []
    for data, targets in tqdm(train_dataloader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)

    model.eval()
    val_loss = []
    with torch.no_grad():
        for data, targets in tqdm(val_dataloader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            val_loss.append(loss.item())
    
    avg_val_loss = np.mean(val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'unet.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break
