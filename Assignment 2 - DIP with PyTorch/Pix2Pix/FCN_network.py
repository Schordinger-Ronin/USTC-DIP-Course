import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
        # Encoder
        # Input: (3, 256, 256)
        self.enc1 = self.conv_block(3, 64)       # Output: (64, 128, 128)
        self.enc2 = self.conv_block(64, 128)     # Output: (128, 64, 64)
        self.enc3 = self.conv_block(128, 256)    # Output: (256, 32, 32)
        self.enc4 = self.conv_block(256, 512)    # Output: (512, 16, 16)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 512) # Output: (512, 8, 8)

        # Decoder - with Skip Connections and Dropout
        # Add Dropout to the first three innermost decoding layers to prevent overfitting and provide randomness
        self.dec4 = self.deconv_block(512, 512, use_dropout=True)       # Output: (512, 16, 16)
        self.dec3 = self.deconv_block(512 + 512, 256, use_dropout=True) # Output: (256, 32, 32)
        self.dec2 = self.deconv_block(256 + 256, 128, use_dropout=True) # Output: (128, 64, 64)
        
        # No longer use Dropout
        self.dec1 = self.deconv_block(128 + 128, 64, use_dropout=False) # Output: (64, 128, 128)
        
        # Final Layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Keep the output between [-1, 1]
        ) # Output: (3, 256, 256)

        # Apply weight initialization
        self.apply(self.init_weights)

    # Weight initialization method for Pix2Pix
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize convolutional layer weights with a normal distribution of mean 0 and standard deviation 0.02
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm layer weights with mean 1.0, standard deviation 0.02, and bias 0
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels):
        # Encoder uses LeakyReLU
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def deconv_block(self, in_channels, out_channels, use_dropout=False):
        # Dynamically build the sequence for the Decoder
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        
        # If Dropout is enabled, add 50% Dropout
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        # Use standard ReLU for the activation function
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoding stage
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck layer
        b = self.bottleneck(e4)
        
        # Decoding stage: Concatenate along the channel dimension (dim=1)
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1) 
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1) 
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1) 
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1) 
        
        # Output
        out = self.final(d1)
        return out
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        """
        in_channels defaults to 6 because the cGAN discriminator needs to observe 
        the concatenated result of both the [input original image (3)] and the [output image (3)]
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
            # The first layer does not use BatchNorm
            *discriminator_block(in_channels, 64, normalization=False), # Output: (64, 128, 128)
            *discriminator_block(64, 128),                              # Output: (128, 64, 64)
            *discriminator_block(128, 256),                             # Output: (256, 32, 32)
            # To maintain the Patch size, the stride of the penultimate layer is set to 1
            *discriminator_block(256, 512, stride=1),                   # Output: (512, 31, 31)
            # Output layer: Map the number of channels to 1 (real/fake probability map)
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
        # Concatenate the input image and the target image along the channel dimension (dim=1) as a conditional input
        img_input = torch.cat((input_img, target_img), dim=1)
        return self.model(img_input)