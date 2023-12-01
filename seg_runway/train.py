import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from net.unet import UNet
from dataloader import UnetDataset
from utils.transform import train_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_data_path = "./datasets/train"
weight_path_dir = './pth'
train_image_save_path = './train_image_show'

train_epochs = 1000

if __name__ == '__main__':
    print("generate Unet model")
    net = UNet(in_channels=3, out_channels=1, init_features=8, WithActivateLast=False).to(device)
    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()
    train_data_loader = DataLoader(UnetDataset(train_data_path, train_transform), batch_size=32, shuffle=True)
    print("start train")

    train_losses = []

    plt.figure(figsize=(10, 5))
    plt.xlabel('Batch Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.grid(True)

    for epoch in range(train_epochs):
        for i, (image, segment_image) in enumerate(tqdm(train_data_loader)):  # 使用 tqdm 包装数据加载器
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)

            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_losses.append(train_loss.item())

        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.pause(0.1)
        plt.draw()  # 更新图形

        if epoch % 5 == 0:
            torch.save(net.state_dict(), weight_path_dir + f'/2023_epoch_{epoch}_loss_{train_loss.item()}.pth')
