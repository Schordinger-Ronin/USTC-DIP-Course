import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ==========================================
        # Encoder (编码器)
        # ==========================================
        # Input: (3, 256, 256)
        self.enc1 = self.conv_block(3, 64)       # Output: (64, 128, 128)
        self.enc2 = self.conv_block(64, 128)     # Output: (128, 64, 64)
        self.enc3 = self.conv_block(128, 256)    # Output: (256, 32, 32)
        self.enc4 = self.conv_block(256, 512)    # Output: (512, 16, 16)
        
        # Bottleneck (瓶颈层)
        self.bottleneck = self.conv_block(512, 512) # Output: (512, 8, 8)

        # ==========================================
        # Decoder (解码器) - 带有 Skip Connections 和 Dropout
        # ==========================================
        # 核心优化：在最深处的前三个解码层加入 Dropout，防止过拟合并提供随机性
        self.dec4 = self.deconv_block(512, 512, use_dropout=True)       # Output: (512, 16, 16)
        self.dec3 = self.deconv_block(512 + 512, 256, use_dropout=True) # Output: (256, 32, 32)
        self.dec2 = self.deconv_block(256 + 256, 128, use_dropout=True) # Output: (128, 64, 64)
        
        # 越靠近输出层，特征越具象，不再使用 Dropout
        self.dec1 = self.deconv_block(128 + 128, 64, use_dropout=False) # Output: (64, 128, 128)
        
        # Final Layer (输出层)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # 保持输出在 [-1, 1] 之间
        ) # Output: (3, 256, 256)

        # 应用权重初始化
        self.apply(self.init_weights)

    # 核心优化：Pix2Pix 官方的权重初始化方法
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # 卷积层权重初始化为均值 0，标准差 0.02 的正态分布
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm 层权重初始化为均值 1.0，标准差 0.02，偏置为 0
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels):
        # Encoder 使用 LeakyReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def deconv_block(self, in_channels, out_channels, use_dropout=False):
        # 动态构建 Decoder 的序列
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        
        # 如果启用 Dropout，按照 Pix2Pix 论文规范添加 50% 的 Dropout
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        # 激活函数使用普通的 ReLU
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码阶段
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # 瓶颈层
        b = self.bottleneck(e4)
        
        # 解码阶段：按通道拼接 (dim=1)
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1) 
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1) 
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1) 
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1) 
        
        # 输出
        out = self.final(d1)
        return out
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        """
        in_channels 默认为 6，因为 cGAN 的判别器需要同时观察【输入原图(3)】和【输出图(3)】的拼接结果
        """
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True, stride=2):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1, bias=not normalization)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 第一层不使用 BatchNorm
            *discriminator_block(in_channels, 64, normalization=False), # Output: (64, 128, 128)
            *discriminator_block(64, 128),                              # Output: (128, 64, 64)
            *discriminator_block(128, 256),                             # Output: (256, 32, 32)
            # 为了维持 Patch 尺寸，倒数第二层步长设为 1
            *discriminator_block(256, 512, stride=1),                   # Output: (512, 31, 31)
            # 输出层：将通道数映射为 1 (真假概率图)
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=True)      # Output: (1, 30, 30)
        )
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_img, target_img):
        # 将输入图和目标图在通道维度(dim=1)上拼接起来，作为条件输入
        img_input = torch.cat((input_img, target_img), dim=1)
        return self.model(img_input)