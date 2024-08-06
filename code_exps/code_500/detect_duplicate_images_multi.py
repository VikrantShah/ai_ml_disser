import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import cv2
import imagehash
from tqdm.notebook import tqdm as tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import multiprocessing

# Get paths of all images with a specific naming pattern
paths = sorted(glob.glob("C:/University of Limerick/AI_ML/code_exps/code_500/input/prostate-cancer-grade-assessment/train_images_jpeg/*_0.jpeg"))
print(f"Number of image paths loaded: {len(paths)}")  # Debugging statement

# Extract unique image IDs from the file names
imgids = [Path(p).stem.split('_')[0] for p in paths]
print(f"Number of image IDs: {len(imgids)}")  # Debugging statement
print(f"Number of unique image IDs: {len(set(imgids))}")  # Debugging statement

# Define a list of image hashing functions from the imagehash library
funcs = [
    imagehash.average_hash,
    imagehash.phash,
    imagehash.dhash,
    imagehash.whash,
    # Uncomment the next line to use the 'db4' mode of whash
    # lambda x: imagehash.whash(x, mode='db4'),
]

def compute_hash(path):
    """
    Compute the hashes for a given image path using the defined hashing functions.

    Parameters:
    path (str): The file path of the image.

    Returns:
    tuple: A tuple containing the path and its computed hash array.
    """
    try:
        image = cv2.imread(path)  # Read the image using OpenCV
        image = Image.fromarray(image)  # Convert the image to a PIL Image
        # Compute the hash for each function and return the result
        hashes = np.array([f(image).hash for f in funcs]).reshape(256)
        return path, hashes
    except Exception as e:
        print(f"Failed to compute hash for {path}: {e}")
        return path, None

# Use multiprocessing to compute hashes in parallel
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    results = list(tqdm(pool.imap(compute_hash, paths), total=len(paths)))

# Filter out failed results
hashes = [hash for path, hash in results if hash is not None]
paths = [path for path, hash in results if hash is not None]

# Convert the list of hashes to a PyTorch tensor
hashes = torch.Tensor(np.array(hashes).astype(int))
print(f"Hashes tensor shape: {hashes.shape}")  # Debugging statement

# Calculate similarity scores between each pair of images
sims = np.array([(hashes[i] == hashes).sum(dim=1).numpy()/256 for i in range(hashes.shape[0])])
print(f"Similarity matrix shape: {sims.shape}")  # Debugging statement

# Set a threshold to identify duplicate images
threshold = 0.90
# Find pairs of images with similarity scores greater than the threshold
duplicates = np.where(sims > threshold)
print(f"Number of duplicate pairs found: {len(duplicates[0])}")  # Debugging statement

# Initialize a dictionary to store pairs of duplicate images
pairs = {}
for i, j in zip(*duplicates):
    if i == j:
        continue  # Skip if the pair consists of the same image

    # Get the paths of the duplicate images
    path1 = paths[i]
    path2 = paths[j]
    print(f"Duplicate pair found: {path1}, {path2}")  # Debugging statement

    # Read and display the duplicate images side by side using matplotlib
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    if image1.shape[0] > image1.shape[1] / 2:
        fig, ax = plt.subplots(figsize=(20, 20), ncols=2)
    elif image1.shape[1] > image1.shape[0] / 2:
        fig, ax = plt.subplots(figsize=(20, 20), nrows=2)
    else:
        fig, ax = plt.subplots(figsize=(20, 30), nrows=2)
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    plt.show()

print("The duplicates are as follows:")
print(duplicates)

# Create a graph to group connected components of duplicate pairs
g1 = nx.Graph()
for i, j in tqdm(zip(*duplicates)):
    g1.add_edge(i, j)

# Get the groups of duplicates
duplicates_groups = list(list(x) for x in nx.connected_components(g1))
print(f"Number of duplicate groups: {len(duplicates_groups)}")  # Debugging statement

# Define a dictionary to store information about each image and its group
df_dict = {
    "image_id": list(),
    "group_id": list(),
    "index_in_group": list(),
}

# Populate the dictionary with image IDs, group IDs, and indices in group
for group_idx, group in enumerate(duplicates_groups):
    for indx, indx_path in enumerate(group):
        p = Path(paths[indx_path])
        img_id = p.stem.split('_')[0]
        assert len(img_id) == 32  # Ensure the image ID has a length of 32

        df_dict["image_id"].append(img_id)
        df_dict["group_id"].append(group_idx)
        df_dict["index_in_group"].append(indx)

# Create a DataFrame from the dictionary
df = pd.DataFrame(df_dict)
# Display the first few rows of the DataFrame
display(df.head())

print(f"Number of rows in DataFrame: {len(df)}")  # Debugging statement
print(f"Number of unique image IDs in DataFrame: {len(df.image_id.unique())}")  # Debugging statement

# Save the DataFrame to a CSV file
df.to_csv("duplicate_imgids_imghash_thres_090.csv", index=False)
print("Saved the DataFrame to CSV file: duplicate_imgids_imghash_thres_090.csv")  # Debugging statement
