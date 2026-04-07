import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork, PatchGANDiscriminator
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms.functional as TF
import random

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.
    """
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    actual_num_images = min(num_images, inputs.size(0))
    
    for i in range(actual_num_images):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

# 注意：确保在顶部导入了新的类
# from FCN_network import FullyConvNetwork, PatchGANDiscriminator

def train_one_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion_GAN, criterion_L1, device, epoch, num_epochs):
    generator.train()
    discriminator.train()
    
    running_loss_G = 0.0
    running_loss_D = 0.0
    lambda_L1 = 100

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # 1. 训练生成器 (Generator) - 每个 Step 都训练
        # ------------------------------------------
        optimizer_G.zero_grad()
        
        fake_images = generator(image_rgb)
        
        # 构建 PatchGAN 的真标签 (用于骗过判别器)
        # 这里可以使用 Label Smoothing，将 1.0 改为 0.9
        valid = torch.full((image_rgb.size(0), 1, 30, 30), 0.9, device=device)
        
        pred_fake = discriminator(image_rgb, fake_images)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_images, image_semantic)
        
        loss_G = loss_GAN + lambda_L1 * loss_L1
        loss_G.backward()
        optimizer_G.step()

        # 2. 策略性训练判别器 (Discriminator) - 每 4 个 Step 才练一次
        # -------------------------------------------------------
        # 【修改位置在此】：加入 if i % 4 == 0 判断
        if i % 4 == 0:
            optimizer_D.zero_grad()
            
            # 判别真图
            pred_real = discriminator(image_rgb, image_semantic)
            loss_real = criterion_GAN(pred_real, valid)
            
            # 判别假图 (Label Smoothing: 0.0 改为 0.1)
            fake_label = torch.full((image_rgb.size(0), 1, 30, 30), 0.1, device=device)
            pred_fake_d = discriminator(image_rgb, fake_images.detach())
            loss_fake = criterion_GAN(pred_fake_d, fake_label)
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            running_loss_D = loss_D.item() # 更新记录用的 Loss

        # 3. 记录数据
        running_loss_G += loss_G.item()
        
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_images, 'train_results', epoch)

        # 每 50 个 Step 打印一次
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_D: {running_loss_D:.4f}, Loss_G: {loss_G.item():.4f}')

def validate(generator, dataloader, criterion_L1, device, epoch, num_epochs):
    generator.eval() 
    
    val_loss = 0.0

    for m in generator.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            outputs = generator(image_rgb)
            loss = criterion_L1(outputs, image_semantic)
            val_loss += loss.item()

            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation L1 Loss: {avg_val_loss:.4f}')

def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Training on device: {device}") 

    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 实例化双网络
    generator = FullyConvNetwork().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    
    # 损失函数: L1 负责大体结构，BCEWithLogitsLoss 负责对抗真假判定
    criterion_L1 = nn.L1Loss()
    criterion_GAN = nn.BCEWithLogitsLoss()
    
    # 两个独立的优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # 【修改点 2】：学习率调度器，由于总轮次减少，将 step_size 从 100 提前到 50
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)

    # 【修改点 3】：总训练轮次从 300 大幅缩减到 100
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, optimizer_G, optimizer_D, criterion_GAN, criterion_L1, device, epoch, num_epochs)
        validate(generator, val_loader, criterion_L1, device, epoch, num_epochs)

        scheduler_G.step()
        scheduler_D.step()

        # 【修改点 4】：模型保存频率从每 50 轮保存一次，改为每 10 轮保存一次，防止意外死机
        if (epoch + 1) % 10 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()