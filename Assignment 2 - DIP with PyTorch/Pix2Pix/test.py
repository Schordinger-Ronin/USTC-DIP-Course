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
    Convert a PyTorch Tensor to a NumPy array that can be saved by OpenCV,
    and handle the RGB to BGR color conversion.
    """
    image = tensor.cpu().detach().numpy()
    # (C, H, W) -> (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize [-1, 1] -> [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255]
    image = (image * 255).astype(np.uint8)
    # OpenCV requires converting back to BGR for saving
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def main():
    # Hardware device configuration
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Testing on device: {device}")

    # Initialize the network and load the weights of the 300th epoch
    generator = FullyConvNetwork().to(device)
    weight_path = 'checkpoints/generator_epoch_300.pth'
    
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found {weight_path}")
        return
        
    generator.load_state_dict(torch.load(weight_path, map_location=device))
    print("Successfully loaded model weights!")

    # Define the tool for calculating L1 Loss
    criterion_L1 = nn.L1Loss()

    # Set to evaluation mode, but forcefully enable Dropout
    generator.eval()
    for m in generator.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    # Prepare the test dataset
    test_dataset = FacadesDataset(list_file='test_list.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    save_dir = 'test_results'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start testing {len(test_dataset)} images...")

    # Variable to record the total loss
    total_test_loss = 0.0

    # Start inference
    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(test_loader):
            # Send both input and target images to the GPU
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            
            # Forward pass to generate images
            outputs = generator(image_rgb)
            
            # Calculate the L1 Loss for this image and accumulate
            loss = criterion_L1(outputs, image_semantic)
            total_test_loss += loss.item()
            
            # Convert the Tensor back to an image
            input_img_np = tensor_to_image(image_rgb[0])
            target_img_np = tensor_to_image(image_semantic[0])
            output_img_np = tensor_to_image(outputs[0])
            
            # Horizontal concatenation: Input (condition image) | Real target image | Model generated image
            comparison = np.hstack((input_img_np, target_img_np, output_img_np))
            
            # Save the image
            save_path = os.path.join(save_dir, f'test_result_{i + 1:03d}.png')
            cv2.imwrite(save_path, comparison)
            
            # Print the progress and current loss
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"Processed [{i + 1}/{len(test_loader)}] images, Current L1 Loss: {loss.item():.4f}")

    # Calculate and print the average Loss of the entire test set
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\nTesting finished!")
    print(f"Average Test L1 Loss: {avg_test_loss:.4f}")
    print(f"All results are saved in the '{save_dir}' folder.")

if __name__ == '__main__':
    main()