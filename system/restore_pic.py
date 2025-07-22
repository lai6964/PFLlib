import copy

import torch

from main import *
from utils.data_utils import read_client_data

from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def get_datalodaer(dataset, id):
    train_data = read_client_data(dataset, id, is_train=True)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

def recover_image(tensor):
    # 1. 克隆张量以避免修改原始数据
    tensor = tensor.clone().detach()

    # 2. 逆归一化：x = (x * std) + mean
    # 此处 std=0.5, mean=0.5 → x = x * 0.5 + 0.5
    # 等价于 (x + 1) / 2
    tensor = tensor * 0.5 + 0.5

    # 3. 转换维度: (C, H, W) → (H, W, C)
    img = tensor.permute(1, 2, 0)

    # 4. 转换为 NumPy 数组
    img = img.numpy()

    return img

def show_eg():
    # 定义逆向转换操作
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),  # 逆向归一化
        transforms.ToPILImage()  # 转换为PIL Image
    ])
    dataloader = get_datalodaer('Cifar100', 0)
    images, labels = next(iter(dataloader))
    # 显示原始图像和恢复后的图像
    fig, ax = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(4):
        # 显示原始图像
        restored_img1 = inverse_transform(images[i])
        ax[0, i].imshow(restored_img1)
        # ax[0, i].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        ax[0, i].set_title(f'Label: {labels[i]}')
        ax[0, i].axis('off')

        # 显示恢复后的图像
        restored_img = recover_image(images[i])
        ax[1, i].imshow(restored_img)
        ax[1, i].axis('off')
    plt.show()

def plot(images, respic):
    fig, ax = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(4):
        # 显示原始图像
        restored_img1 = recover_image(images[i])
        ax[0, i].imshow(restored_img1)
        # ax[0, i].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        # ax[0, i].set_title(f'Label: {labels[i]}')
        ax[0, i].axis('off')

        # 显示恢复后的图像
        restored_img = recover_image(respic[i])
        ax[1, i].imshow(restored_img)
        ax[1, i].axis('off')
    plt.show()

def train(model, dataloader):
    model.to(device)
    opt = torch.optim.SGD(model.decoder.parameters(), lr=0.005)

    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    for epoch in range(100):
        for images,labels in dataloader:
            images = images.to(device)
            embedding = model.base(images)
            reconstructed_image = model.decoder(embedding)
            loss = nn.MSELoss()(images, reconstructed_image)

            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch%10==9:
            plot(images.cpu(),reconstructed_image.cpu())
    torch.save(model,"cifar100.pt")



def plot_mine(images, respic, resimg):
    fig, ax = plt.subplots(2, 3, figsize=(3, 2))
    for i in range(2):
        # 显示原始图像
        restored_img1 = recover_image(images[i])
        ax[i, 0].imshow(restored_img1)
        ax[i, 0].axis('off')

        # 显示恢复后的图像
        restored_img = recover_image(respic[i])
        ax[i, 1].imshow(restored_img)
        ax[i, 1].axis('off')
        # 显示恢复后的图像
        restored_img = recover_image(resimg[i])
        ax[i, 2].imshow(restored_img)
        ax[i, 2].axis('off')
    plt.savefig(f"tmp.pdf",
                bbox_inches='tight',  # 关键：让 savefig 计算紧凑边界
                pad_inches=0)  # 可选：不留额外边距（默认 0.1）
    plt.show()


if __name__ == '__main__':

    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = 'Cifar100'
    algo = 'FedDFCC'
    id = 10


    dataloader = get_datalodaer(dataset,id)
    model1 = torch.load("models/Cifar100_agg/{}_client_{}.pt".format(algo, id), weights_only=False)
    model2 = torch.load("cifar100.pt", weights_only=False)

    # train(tmp_model,dataloader)

    with torch.no_grad():
        for i in range(20):
            images, labels = next(iter(dataloader))
        rep = model1.base(images.to(device))
        reconstructed_image1 =model1.decoder(rep)

        rep = model2.base(images.to(device))
        reconstructed_image2 =model2.decoder(rep)

        oripic = images.cpu()[1:3]
        respic = reconstructed_image1.cpu()[1:3]
        resimg = reconstructed_image2.cpu()[1:3]

        plot_mine(oripic, respic, resimg)


