# Define tile and image dimensions
tile_size = 256
image_size = 256
n_tiles = 36

# Batch size and number of worker threads for data loading
batch_size = 1
num_workers = 1 

# Debugging flag
debug = False

# Import necessary libraries
import os
import sys
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
# from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm as tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import multiprocessing

# Configurations for data directories and files
data_dir = "C:/University of Limerick/AI_ML/code_exps/code_500/input/prostate-cancer-grade-assessment"
out_dir = "C:/University of Limerick/AI_ML/code_exps/code_500/input/"
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'train_images')

# Print the image folder path for debugging purposes
print(f"Image folder path: {image_folder}")

# Function to extract tiles from an image
def get_tiles(img, mode=0, transform=None):
    result = []
    h, w, c = img.shape

    # Calculate padding to make image dimensions multiples of tile_size
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    # Pad the image with white color (constant value 255)
    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]], constant_values=255)

    # Reshape and reorder the padded image into tiles
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    ).astype(np.float32)  # Use uint8 instead of float32 to reduce memory usage

    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    # If using float32, remember to normalize pixel values appropriately if needed
    # img3 /= 255.0  # Normalizing to range [0, 1]

    # Count tiles with information (not all white)
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255).sum()

    # If there are fewer tiles than needed, pad with white tiles
    if len(img3) < n_tiles:
        img3 = np.pad(img3, [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]], constant_values=255)

    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]

    for i in range(len(img3)):
        result.append({"img": img3[i], "idx": i})

    if transform is not None:
        for i in range(len(result)):
            result[i]["img"] = transform(image=result[i]["img"])["image"]
    return result


class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                 show=False
                ):
        """
        Args:
            df (DataFrame): DataFrame containing image metadata
            image_size (int): Size of the images
            n_tiles (int): Number of tiles per image
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform
        self.show = show

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset
        """
        return self.df.shape[0]


    def __getitem__(self, index):
        print(f"Fetching item index: {index}")
        row = self.df.iloc[index]
        img_id = row.image_id
        if not os.path.isfile(os.path.join(image_folder, f'{img_id}.tiff')):
            pass
        
        # Load images as tiles
        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        multi_image = skimage.io.MultiImage(tiff_file)
        if len(multi_image) > 1:
            image = multi_image[1]  # load mid-resolution if available
        else:
            image = multi_image[0]  # fallback to the first image if only one is available
        
        tiles = get_tiles(image, self.tile_mode, self.transform)  # Adjusted to handle single return value

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        # Concat tiles into a single image
        imgs = np.zeros((self.image_size * self.n_tiles // 6, self.image_size * self.n_tiles // 6, 3), dtype=np.uint8)
        for h in range(self.n_tiles // 6):
            for w in range(self.n_tiles // 6):
                this_img = tiles[idxes[h * (self.n_tiles // 6) + w]]["img"]
                imgs[h * self.image_size:(h + 1) * self.image_size, w * self.image_size:(w + 1) * self.image_size, :] = this_img

        if self.transform is not None:
            imgs = self.transform(image=imgs)["image"]
        return imgs, row.isup_grade, img_id



if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("Creating output directory...")
    # Create output directory for processed tiles
    os.makedirs(os.path.join(out_dir, "train_{}_{}".format(image_size, n_tiles)), exist_ok=True)

    print("Initializing dataset and DataLoader...")
    # Initialize dataset and dataloader for image processing
    dataset_show = PANDADataset(df_train, image_size, n_tiles, transform=None)
    train_loader = DataLoader(dataset_show, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    print("Starting data processing...")
    # Process data using a progress bar
    # Generate npz files
    bar = tqdm(train_loader)
    cnt = 0
    for (data, target, id) in bar:
        cnt += 1
        if cnt == 10 and debug:
            break
        np.savez_compressed(os.path.join(out_dir, "train_{}_{}".format(image_size, n_tiles), f"{id[0]}.npz"), data)


"""
# Debugging section for loading and visualizing an image
# debug data load
file = "./train_256_36/004dd32d9cd167d9cc31c13b704498af.npz"
images = np.load(file)["arr_0"]
images = images.transpose(2, 0, 1)

img = images/255
img = np.transpose(img)
print(img.shape)
plt.imshow(1. - img)
"""