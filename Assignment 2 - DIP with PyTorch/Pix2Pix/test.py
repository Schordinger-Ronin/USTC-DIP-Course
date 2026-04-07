import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork

def tensor_to_image(tensor):
    """
    将 PyTorch Tensor 转换为可供 OpenCV 保存的 NumPy 数组，
    并处理好 RGB 到 BGR 的色彩转换。
    """
    image = tensor.cpu().detach().numpy()
    # (C, H, W) -> (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # 反归一化 [-1, 1] -> [0, 1]
    image = (image + 1) / 2
    # 缩放至 [0, 255]
    image = (image * 255).astype(np.uint8)
    # OpenCV 保存时需要转回 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def main():
    # 1. 硬件设备配置
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Testing on device: {device}")

    # 2. 初始化网络并加载 300 轮的权重
    generator = FullyConvNetwork().to(device)
    weight_path = 'checkpoints/generator_epoch_300.pth'
    
    if not os.path.exists(weight_path):
        print(f"Error: 找不到权重文件 {weight_path}")
        return
        
    generator.load_state_dict(torch.load(weight_path, map_location=device))
    print("Successfully loaded model weights!")

    # 定义计算 L1 Loss 的工具
    criterion_L1 = nn.L1Loss()

    # 3. 设置为评估模式，但强行开启 Dropout
    generator.eval()
    for m in generator.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # 4. 准备测试数据集
    test_dataset = FacadesDataset(list_file='test_list.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    save_dir = 'test_results'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start testing {len(test_dataset)} images...")

    # 用来记录总 loss 的变量
    total_test_loss = 0.0

    # 5. 开始推理
    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(test_loader):
            # 将输入和目标图都送入 GPU
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            
            # 前向传播，生成图像
            outputs = generator(image_rgb)
            
            # 计算这张图的 L1 Loss 并累加
            loss = criterion_L1(outputs, image_semantic)
            total_test_loss += loss.item()
            
            # 将 Tensor 转换回图片
            input_img_np = tensor_to_image(image_rgb[0])
            target_img_np = tensor_to_image(image_semantic[0])
            output_img_np = tensor_to_image(outputs[0])
            
            # 横向拼接：输入(条件图) | 真实目标图 | 模型生成图
            comparison = np.hstack((input_img_np, target_img_np, output_img_np))
            
            # 保存图片
            save_path = os.path.join(save_dir, f'test_result_{i + 1:03d}.png')
            cv2.imwrite(save_path, comparison)
            
            # 打印进度和当前的 loss
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"Processed [{i + 1}/{len(test_loader)}] images, Current L1 Loss: {loss.item():.4f}")

    # 6. 计算并打印整个测试集的平均 Loss
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\nTesting finished!")
    print(f"=========================================")
    print(f"Average Test L1 Loss: {avg_test_loss:.4f}")
    print(f"=========================================")
    print(f"All results are saved in the '{save_dir}' folder.")

if __name__ == '__main__':
    main()