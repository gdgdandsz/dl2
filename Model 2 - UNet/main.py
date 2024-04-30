import os
import torch
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm



from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

from torchvision.transforms import ToTensor
from torchvision import transforms

import torch.nn.functional as F
from torch import nn
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

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.iou_loss = IoULoss()

    def forward(self, inputs, targets):
        #loss_ce=0
        loss_ce = self.ce_loss(inputs, targets)
        # 使用softmax处理预测结果，以计算IoU损失
        inputs_soft = F.softmax(inputs, dim=1)
        loss_iou = self.iou_loss(inputs_soft, F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2))
        # 组合损失，权重可以根据需要调整
        return 0.8 * loss_ce + 0.2 * loss_iou

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum(dim=(2, 3))
        union = (preds + targets).sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        # 返回1 - IoU以作为损失
        return 1 - iou.mean()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":

    # cfg = {
    # "train": {"learning_rate":1e-4, "epochs":1}

    # }

    # wandb.init(project='unet-seg', config=cfg)
    transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor，自动将[0, 255]范围内的数据标准化至[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
    
    
    train_set_path = '/scratch/xz3645/test/dl/Dataset_Student/train/video_' #Change this to your train set path
    val_set_path = '/scratch/xz3645/test/dl/Dataset_Student/val/video_' #Change this to your validation path

    train_data_dir = [f"{train_set_path}{i:05d}" for i in range(0, 1000)]
    train_dataset = SegmentationDataSet(train_data_dir)#, transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_data_dir = [f"{val_set_path}{i:05d}" for i in range(1000, 2000)]
    val_dataset = SegmentationDataSet(val_data_dir)#, transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = unet_model().to(DEVICE)
    best_model = None
    # summary(model, (3, 256, 256))

    # hyperparameters

    LEARNING_RATE = 0.0005
    num_epochs = 100
    max_patience = 3
    epochs_no_improve = 0
    early_stop = False
    SMOOTH = 1e-6

    # loss criterion, optimizer

    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    # 初始
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    scaler = torch.cuda.amp.GradScaler()
    best_iou = 0.0 
    # Train loop
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader)
        for idx, (data, targets) in enumerate(loop):
            data = data.permute(0, 3, 1, 2).to(torch.float16).to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())

        val_losses = []
        last_val_loss = 1000000
        model.eval()
        mean_thresholded_iou = []
        ious = []
        last_iou = 0
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_dataloader)):
                x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(DEVICE)
                y = y.to(DEVICE)
                y = y.type(torch.long)
                # forward
                with torch.cuda.amp.autocast():
                    preds = model(data)
                    vloss = loss_fn(preds, y)

                # print(x.shape, y.shape, preds.shape)
                # vloss = loss_fn(preds, y)
                val_losses.append(vloss.item())

                # wandb.log({f'val_loss_epoch_{epoch}':vloss})
                # class_labels = {}

                # if (i+1)%5 == 0:
                #    wandb.log(
                #        {f"image_epoch_{epoch}_step_{i}" : wandb.Image(x, mask={
                #                f'pred_masks_epoch_{epoch}_step_{i}': 
                #                    {"pred_masks":preds, "class_labels":class_labels},
                #               f'true_masks_epoch_{epoch}_step_{i}':
                #                    {"true_masks":y, "class_labels":class_labels}
                #            }) 
                #        })
                preds_arg = torch.argmax(softmax(preds), axis=1)

                thresholded_iou = batch_iou_pytorch(SMOOTH, preds_arg, y)
                ious.append(thresholded_iou)
        model.eval()
        jaccard = torchmetrics.JaccardIndex(num_classes=49, task="multiclass").to(DEVICE)
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader):
                images = images.float().permute(0, 3, 1, 2).to(DEVICE)  # 确保图像是float类型并且维度正确
                masks = masks.to(DEVICE)  # Send masks to devic
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                jaccard.update(preds, masks)
            
        iou_score = jaccard.compute()
        print(f"Validation IoU: {iou_score}")
            
        mean_thresholded_iou = sum(ious) / len(ious)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch: {epoch}, avg IoU: {mean_thresholded_iou}, avg val loss: {avg_val_loss}")
            
        scheduler.step(iou_score)

        if iou_score > best_iou:
            best_model = model
            best_iou=iou_score
            torch.save(best_model, 'unet.pt')
            last_val_loss = avg_val_loss
            epochs_no_improve = 0
            print('best_iou model saved',best_iou)
        else:
            epochs_no_improve += 1

        if epochs_no_improve > max_patience and epoch > 10:
            early_stop = True
            print("Early Stopping")

    #check_accuracy(val_dataloader, best_model)

# if __name__=='__main__':
#    sweep_configuration = {
#           "method": "grid",
#           "parameters":{
#                "learning_rate": {"values":[1e-4]}
#        }
#    }

# Start the sweep
#    sweep_id = wandb.sweep(
#        sweep=sweep_configuration,
#        project='unet-seg',
#        )

#    wandb.agent(sweep_id, function=main, count=1)
