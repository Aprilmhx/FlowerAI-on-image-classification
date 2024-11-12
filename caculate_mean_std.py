from PIL import Image
import numpy as np
from pathlib import Path

def load_and_preprocess_image(file_path):
    with Image.open(file_path) as img:
        img = img.convert('RGB')  # Ensure it's in RGB format
        im = np.asarray(img).astype(float) / 255.  # Convert to numpy array and scale
    return im

image_files_dir = Path(r"isic2019_all")
files = list(image_files_dir.rglob("*.jpg"))
num_samples = len(files)
if num_samples == 0:
    raise ValueError("No image files found in the specified directory.")

# Calculating mean
mean = np.zeros(3)
std = np.zeros(3)
for file in files:
    im = load_and_preprocess_image(file)    #(450, 600, 3),...
    mean_temp = np.mean(im, axis=(0, 1))
    mean += mean_temp  # Average over height and width
    
    std_temp = np.sqrt(np.sum((im - mean_temp)**2, axis=(0, 1))/(im.shape[0] * im.shape[1]))
    std += std_temp
mean /= num_samples    
std = std / num_samples  # Normalize by total number of pixels

print("Mean of the images:", mean)
print("Standard deviation of the images:", std)