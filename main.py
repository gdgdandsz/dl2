import os
import torch
import numpy as np
import torchmetrics
from torchmetrics import JaccardIndex
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import optuna


from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Define the dataset
class SegmentationDataSet(Dataset):

    def __init__(self, video_dir, transform=None):
        self.transform = transform
        self.images, self.masks = [], []
        for i in video_dir:
            imgs = os.listdir(i)
            self.images.extend([i + '/' + img for img in imgs if not img.startswith(
                'mask')])  # /content/gdrive/MyDrive/Dataset_Studentnew/Dataset_Student/train/video_
        # print(self.images[1000])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        x = self.images[index].split('/')
        image_name = x[-1]
        mask_index = int(image_name.split("_")[1].split(".")[0])
        x = x[:-1]
        mask_path = '/'.join(x)
        mask = np.load(mask_path + '/mask.npy')
        try:
            mask = mask[mask_index, :, :]
        except IndexError:  # Index is out of the bounds of the array
            mask = mask[-1, :, :]  # Use the last mask if the index is out of range
        if self.transform:
            img = self.transform(img)  # 应用transform

        mask = torch.tensor(mask, dtype=torch.long)  # 确保掩码也转换为tensor


        return img, mask

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(self, out_channels=49, features=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = encoding_block(3, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block(features[2], features[3])
        self.conv5 = encoding_block(features[3] * 2, features[3])
        self.conv6 = encoding_block(features[3], features[2])
        self.conv7 = encoding_block(features[2], features[1])
        self.conv8 = encoding_block(features[1], features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3], features[3] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(DEVICE)
    # print(model)
    y_preds_list = []
    y_trues_list = []
    #ious = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            # print(x.shape)
            # plt.imshow(x.cpu()[0])
            x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)

            y_preds_list.append(preds)
            y_trues_list.append(y)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            thresholded_iou = batch_iou_pytorch(SMOOTH, preds, y)
            ious.append(thresholded_iou)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

            print(dice_score)
            print(x.cpu()[0])
            

    mean_thresholded_iou = sum(ious)/len(ious)

    y_preds_concat = torch.cat(y_preds_list, dim=0)
    y_trues_concat = torch.cat(y_trues_list, dim=0)
    print("IoU over val: ", mean_thresholded_iou)

    print(len(y_preds_list))
    print(y_preds_concat.shape)

    jac_idx = jaccard(y_trues_concat, y_preds_concat)

    print(f"Jaccard Index {jac_idx}")

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")


def batch_iou_pytorch(SMOOTH, outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = unet_model().to(device)
    optimizer = Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-1, log=True), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
    train_dataset = SegmentationDataSet(train_data_dir,transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = SegmentationDataSet(val_data_dir,transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 30
    best_iou = 0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        jaccard = JaccardIndex(num_classes=49, task="multiclass").to(device)
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.permute(0, 3, 1, 2).float().to(device)
                masks = masks.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                jaccard.update(preds, masks)
        iou_score = jaccard.compute()
        scheduler.step(iou_score)

        if iou_score > best_iou:
            best_iou = iou_score
            best_model = copy.deepcopy(model)
            torch.save(best_model, 'unet.pt')  # Saving the entire model

    return best_iou

if __name__ == "__main__":
    train_data_dir = [f'/scratch/xz3645/test/dl/Dataset_Student/train/video_{i:05d}' for i in range(0, 1000)]
    val_data_dir = [f'/scratch/xz3645/test/dl/Dataset_Student/val/video_{i:05d}' for i in range(1000, 2000)]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print("  IOU: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

