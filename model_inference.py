#!/usr/bin/env python3
"""
Name: Aayush Sanghvi, Rahul Sha
Class: CS 7180 Advanced Perception
Date: 17th September 2025
Purpose: Implements a Super-Resolution Convolutional Neural Network (SRCNN) to enhance image resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
class SuperResolution: Defined the Super-Resolution CNN model.

Architecture Overview:
  1. layer_1 : Learns 128 different 9x9x3 filters to extract low-level features followed by ReLU for non-linear activation.
  2. layer_2 : Maps the 128-channel feature maps into a 64-channel layer followed by ReLU for non-linear feature transformation.
  3. layer_3 : Reconstructs the final high-resolution RGB image from the feature representation.

Output:
    x : Super-resolved RGB image (B, 3, H, W)
    y : Intermediate 64-channel feature map for visualization or auxiliary loss

"""
class SuperResolution(nn.Module):
    def __init__(self, spatial: list = [9, 5, 5], filter: list = [128, 64], num_channels: int = 3):
        super().__init__()
        self.layer_1 = nn.Conv2d(num_channels, filter[0], spatial[0], padding = spatial[0] // 2)
        self.layer_2 = nn.Conv2d(filter[0], filter[1], spatial[1], padding = spatial[1] // 2)
        self.layer_3 = nn.Conv2d(filter[1], num_channels, spatial[2], padding = spatial[2] // 2)
        self.relu = nn.ReLU()

    def forward(self, image_batch):
        x = self.layer_1(image_batch)
        x = self.relu(x)
        x = self.layer_2(x)
        y = self.relu(x)
        x = self.layer_3(y)
        return x, y 

"""
def execute: Upscales an input image tensor by a scaling factor and prepares patches with defined patch size.

Arguments:
    input_image: Input image tensor of shape (C, H, W).
    model: Trained super-resolution model.
    fs: Patch size for scanning the image.
    scale: Scaling factor for upscaling the image.

Output:
    Image Tensor: Super-resolved image tensor.
"""
def execute(input_image: torch.Tensor,
            model: torch.nn.Module,
            fs: int = 33,
            scale: float = 2.0) -> torch.Tensor:

    # Extract dimensions and prepare scaled image
    c, h, w = input_image.shape
    output_height, output_width = int(h * scale), int(w * scale)
    image = F.interpolate(input_image.unsqueeze(0),
                          size=(output_height, output_width),
                          mode="bicubic",
                          align_corners=False).to(device)

    # Prepare output image tensor
    c, h, w = image.shape[1:]
    output = torch.zeros_like(image, device="cpu")

    # Scan image in patches and apply super resolution model
    for i in tqdm(range(0, h, fs), desc="Scanning", ncols=100):
        for j in range(0, w, fs):
            # Ensure patch stays within bounds
            h_end = min(i + fs, h)
            w_end = min(j + fs, w)
            patch = image[:, :, i:h_end, j:w_end]

            # Model expects a batch; output first item and clamp to [0,1]
            result = model(patch)[0].clamp(0, 1).cpu()
            output[:, :, i:h_end, j:w_end] = result

    print(f"Channels = {c}, Image Shape = {w} x {h}")
    return output

"""
def transform: Load image and convert to a normalized tensor

Arguments:
    path: Path to the input image.

Output:
    Image Tensor: Transformed image tensor.
"""
def transform_image(path: str) -> torch.Tensor:

    image = Image.open(path).convert("RGB")
    return transforms.ToTensor()(image)

"""
def run_super_resolution: High-level function to run super-resolution on an input image using a trained model.

Arguments:
    image_path: Path to the input image.
    model_weights: Path to the trained model weights.
    scale: Scaling factor for upscaling the image.
    output_path: Path to save the super-resolved image.
    patch_size: Patch size for scanning the image.

Output:
    output_path: Path where the super-resolved image is saved.
"""
def run_super_resolution(image_path: str,
                         model_weights: str = "saved/isr_best.pth",
                         scale: float = 2.0,
                         output_path: str = None,
                         patch_size: int = 33) -> str:

    # Load model
    model = SuperResolution()
    map_loc = None if torch.cuda.is_available() else {"cuda:0": "cpu"}
    model.load_state_dict(torch.load(model_weights, map_location=map_loc))
    model.to(device).eval()

    # Prepare image tensor
    img_tensor = transform_image(image_path)

    # Super-resolve
    result = execute(img_tensor, model, fs=patch_size, scale=scale)

    # Save
    to_pil = transforms.ToPILImage()
    out_img = to_pil(result.squeeze())
    if output_path is None:
        root, ext = image_path.rsplit(".", 1)
        output_path = f"{root}_upscaled.{ext}"
    out_img.save(output_path)
    print(f"Image written to {output_path}")
    return output_path


if __name__ == "__main__":
    run_super_resolution(
        image_path="/home/aayush/Code/Advance Perception/image-super-resolution/results/northern_lights.jpg"
    )
