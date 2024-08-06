import numpy as np
import pandas as pd
import glob
import imagehash
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Increase the decompression bomb warning limit
Image.MAX_IMAGE_PIXELS = None

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
]

# Initialize an empty list to store the hashes of images
hashes = []

# Compute the hash for each image using the defined hashing functions
for path in tqdm(paths, total=len(paths)):
    pil_image = Image.open(path)  # Open the image using PIL
    # Compute the hash for each function and store it in the hashes list
    image_hashes = [f(pil_image).hash.flatten() for f in funcs]
    hashes.append(np.concatenate(image_hashes))

# Convert the list of hashes to a PyTorch tensor
hashes = torch.tensor(np.array(hashes, dtype=np.uint8))
print(f"Hashes tensor shape: {hashes.shape}")  # Debugging statement

# Calculate similarity scores between each pair of images
sims = torch.cdist(hashes.float(), hashes.float(), p=0) / 256
print(f"Similarity matrix shape: {sims.shape}")  # Debugging statement

# Identify duplicates based on a similarity threshold
duplicates = (sims > 0.90).nonzero(as_tuple=False)
duplicates = [(i.item(), j.item()) for i, j in duplicates if i != j]
duplicates = list(set([tuple(sorted(d)) for d in duplicates]))
print(f"Number of duplicate pairs found: {len(duplicates)}")  # Debugging statement

# Visualize some of the duplicates
for count, (i, j) in enumerate(duplicates[:5]):
    image1 = Image.open(paths[i])
    image2 = Image.open(paths[j])
    fig, ax = plt.subplots(figsize=(20, 20), ncols=2)
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    plt.show()

print("The duplicates are as follows:")
print(duplicates)

# Create a graph to group connected components of duplicate pairs
g1 = nx.Graph()
g1.add_edges_from(duplicates)

# Get the groups of duplicates
duplicates_groups = [list(x) for x in nx.connected_components(g1)]
print(f"Number of duplicate groups: {len(duplicates_groups)}")  # Debugging statement

# Define a dictionary to store information about each image and its group
df_dict = {
    "image_id": [],
    "group_id": [],
    "index_in_group": [],
}

# Populate the dictionary with image IDs, group IDs, and indices in group
for group_idx, group in enumerate(duplicates_groups):
    for indx in group:
        p = Path(paths[indx])
        img_id = p.stem.split('_')[0]
        assert len(img_id) == 32  # Ensure the image ID has a length of 32

        df_dict["image_id"].append(img_id)
        df_dict["group_id"].append(group_idx)
        df_dict["index_in_group"].append(indx)

# Create a DataFrame from the dictionary
df = pd.DataFrame(df_dict)
# Display the first few rows of the DataFrame
print(df.head())

print(f"Number of rows in DataFrame: {len(df)}")  # Debugging statement
print(f"Number of unique image IDs in DataFrame: {len(df.image_id.unique())}")  # Debugging statement

# Save the DataFrame to a CSV file
df.to_csv("C:/University of Limerick/AI_ML/code_exps/code_500/input/duplicate_imgids_imghash_thres_090.csv", index=False)
print("Saved the DataFrame to CSV file: C:/University of Limerick/AI_ML/code_exps/code_500/input/duplicate_imgids_imghash_thres_090.csv")  # Debugging statement
