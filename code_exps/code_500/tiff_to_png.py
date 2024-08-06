#!/usr/bin/env python3
from pathlib import Path
import multiprocessing
from PIL import Image
import numpy as np
import skimage.io
import tqdm

# Increase the decompression bomb warning limit
Image.MAX_IMAGE_PIXELS = None

def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:
    """Crop the white borders from the image."""
    assert image.shape[2] == 3
    assert image.dtype == np.uint8 or image.dtype == np.uint16
    ys, = (image.min((1, 2)) < value).nonzero()  # Find rows with non-white pixels
    xs, = (image.min(0).min(1) < value).nonzero()  # Find columns with non-white pixels
    if len(xs) == 0 or len(ys) == 0:
        return image  # Return the original image if no non-white pixels are found
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # Crop the image to the bounding box

def to_png(path: Path):
    """Convert a TIFF image to PNG format, processing different resolutions if available."""
    try:
        # print(f"Processing {path}")
        image = skimage.io.MultiImage(str(path))  # Load the multi-resolution TIFF image
        # print(f"Loaded {len(image)} resolutions for {path}")
        # Always process the first resolution
        image_to_png(path, '_0', image[0])
        # Process additional resolutions if present
        if len(image) > 1:
            image_to_png(path, '_1', image[1])
        if len(image) > 2:
            image_to_png(path, '_2', image[2])
    except Exception as e:
        print(f"Failed to process {path}: {e}")  # Print the error message

def image_to_png(path: Path, suffix: str, image: np.ndarray):
    """Save a numpy image array as a PNG file, cropping white borders and ensuring unique file names."""
    try:
        # print(f"Saving {path.stem}{suffix}.png")
        output_dir = Path('C:/University of Limerick/AI_ML/code_exps/code_500/input/prostate-cancer-grade-assessment/train_images_png')  # Define the output directory for PNG images
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        png_path = output_dir / f'{path.stem}{suffix}.png'  # Define the path for the PNG file
        if png_path.exists():
            try:
                Image.open(png_path).verify()  # Verify if the existing PNG file is valid
                # print(f"{png_path} already exists and is valid.")
            except Exception as e:
                print(f"Existing PNG file is invalid: {e}")
            else:
                return  # If the file is valid, skip saving
        image = crop_white(image)  # Crop white borders from the image
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)  # Convert uint16 to uint8 if needed
        pil_image = Image.fromarray(image)  # Convert the numpy array to a PIL Image
        pil_image.save(png_path, format='PNG')  # Save the image as a PNG file
        # print(f"Saved {png_path}")
    except Exception as e:
        print(f"Failed to save {path}: {e}")  # Print the error message and image details
        print(f"Image details: dtype={image.dtype}, shape={image.shape}, path={png_path}")

def main():
    """Main function to process all TIFF images in the specified directory and convert them to PNG format."""
    paths = list(Path('C:/University of Limerick/AI_ML/code_exps/code_500/input/prostate-cancer-grade-assessment/train_images').glob('*.tiff'))  # Get all TIFF files in the directory
    with multiprocessing.Pool(processes=4) as pool:  # Limit number of processes to reduce memory usage
        # Use multiprocessing to convert images in parallel and display progress with tqdm
        for _ in tqdm.tqdm(pool.imap_unordered(to_png, paths), total=len(paths)):
            pass

if __name__ == '__main__':
    main()  # Run the main function if the script is executed directly
