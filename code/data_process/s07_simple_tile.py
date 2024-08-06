# Import necessary libraries
import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.io
from tqdm import tqdm


"""
https://www.kaggle.com/iafoss/panda-16x128x128-tiles
https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/155424
"""

# Define a function to extract tiles from an image
def get_tiles(img, tile_size, n_tiles, mask, mode=0):
    # Determine the size of tiles and calculate necessary padding
    t_sz = tile_size
    h, w, c = img.shape
    pad_h = (t_sz - h % t_sz) % t_sz + ((t_sz * mode) // 2)
    pad_w = (t_sz - w % t_sz) % t_sz + ((t_sz * mode) // 2)

    # Apply padding to the image and reshape it into tiles
    img2 = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img3 = img2.reshape(img2.shape[0] // t_sz, t_sz, img2.shape[1] // t_sz, t_sz, 3)
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    # Process mask similarly
    mask3 = None
    if mask is not None:
        mask2 = np.pad(
            mask,
            [
                [pad_h // 2, pad_h - pad_h // 2],
                [pad_w // 2, pad_w - pad_w // 2],
                [0, 0],
            ],
            constant_values=0,
        )
        mask3 = mask2.reshape(
            mask2.shape[0] // t_sz, t_sz, mask2.shape[1] // t_sz, t_sz, 3
        )
        mask3 = mask3.transpose(0, 2, 1, 3, 4).reshape(-1, t_sz, t_sz, 3)

    # Calculate the number of tiles with significant information
    n_tiles_with_info = (
        img3.reshape(img3.shape[0], -1).sum(1) < t_sz ** 2 * 3 * 255
    ).sum()
    # Add more padding if the number of tiles is less than n_tiles
    if len(img3) < n_tiles:
        img3 = np.pad(
            img3,
            [[0, n_tiles - len(img3)], [0, 0], [0, 0], [0, 0]],
            constant_values=255,
        )
        if mask is not None:
            mask3 = np.pad(
                mask3,
                [[0, n_tiles - len(mask3)], [0, 0], [0, 0], [0, 0]],
                constant_values=0,
            )
    # Sort tiles by the sum of pixel values and select the best n_tiles
    idxs = np.argsort(img3.reshape(img3.shape[0], -1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    if mask is not None:
        mask3 = mask3[idxs]
    return img3, mask3, n_tiles_with_info >= n_tiles


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--tile-num", type=int, default=64)
    arg("--tile-size", type=int, default=192)
    arg("--res-level", type=int, default=1, help="0:High, 1:Middle, 2:Low")
    arg("--resize", type=int, default=None)
    arg("--mode", type=int, default=0)
    return parser.parse_args()

# Main function that sets up the environment and processes all images
def main():
    # Set up command line arguments
    args = make_parse()
    n_tiles = args.tile_num
    tile_size = args.tile_size
    res_level = args.res_level
    resize_size = args.resize
    mode = args.mode

    root = Path("D:/input/prostate-cancer-grade-assessment")
    img_dir = root / "train_images"
    mask_dir = root / "train_label_masks"
    train = pd.read_csv(root / "train.csv")
    out_root = Path("C:/University of Limerick/AI_ML/code/input/prostate-cancer-grade-assessment")

    # Set up output directories and file paths
    out_dir = (
        out_root / f"numtile-{n_tiles}-tilesize-{tile_size}-res-{res_level}-mode-{mode}"
    )
    out_dir.mkdir(exist_ok=True)
    out_train_zip = str(out_dir / "train.zip")
    out_mask_zip = str(out_dir / "mask.zip")
    x_tot, x2_tot = [], []

    # Process images and save them into zip files
    with zipfile.ZipFile(out_train_zip, "w") as img_out, zipfile.ZipFile(
        out_mask_zip, "w"
    ) as mask_out:
        for img_id in tqdm(train.image_id):
            img_path = str(img_dir / (img_id + ".tiff"))
            mask_path = mask_dir / (img_id + "_mask.tiff")

            # Load the image and mask at a specified resolution level
            # RGB
            # image = skimage.io.MultiImage(img_path)[res_level]
            # Check if the desired resolution level is available
            try:
                image = skimage.io.MultiImage(str(img_path))
                if res_level >= len(image):
                    print(f"Requested resolution level {res_level} is not available.")
                    print(f"Using the highest available level {len(image) - 1}.")
                    res_level = len(image) - 1
                image = image[res_level]
            except Exception as e:
                print(f"Error reading image at resolution level {res_level}: {e}")
                return

            mask = None
            if mask_path.exists():
                try:
                    mask = skimage.io.MultiImage(str(mask_path))[res_level]
                except Exception as e:
                    print(f"Error reading mask at resolution level {res_level}: {e}")
                    return

            tiles, masks, _ = get_tiles(
                img=image, tile_size=tile_size, n_tiles=n_tiles, mask=mask, mode=mode
            )

            # Optionally resize tiles for uniformity
            if resize_size is not None:
                tiles = [cv2.resize(t, (resize_size, resize_size)) for t in tiles]
                masks = [cv2.resize(m, (resize_size, resize_size)) for m in masks]

            # Save processed tiles and masks into zip files
            for idx, img in enumerate(tiles):
                # RGB
                x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
                x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))

                # if read with PIL RGB turns into BGR
                # We get CRC error when unzip if not cv2.imencode
                img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f"{img_id}_{idx}.png", img)

                # mask[:, :, 0] has value in {0, 1, 2, 3, 4, 5}, other mask is 0 only
                if mask is not None:
                    mask = masks[idx]
                    mask = cv2.imencode(".png", mask[:, :, 0])[1]
                    mask_out.writestr(f"{img_id}_{idx}.png", mask)

    # image stats
    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("mean:", img_avr, ", std:", np.sqrt(img_std))


if __name__ == "__main__":
    main()
