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
    """
    Crop the white borders from the image.

    Parameters:
    image (np.ndarray): The input image as a numpy array.
    value (int): The pixel value considered as white (default is 255).

    Returns:
    np.ndarray: The cropped image.
    """
    assert image.shape[2] == 3
    assert image.dtype == np.uint8 or image.dtype == np.uint16
    ys, = (image.min((1, 2)) < value).nonzero()  # Find rows with non-white pixels
    xs, = (image.min(0).min(1) < value).nonzero()  # Find columns with non-white pixels
    if len(xs) == 0 or len(ys) == 0:
        return image  # Return the original image if no non-white pixels are found
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # Crop the image to the bounding box

def to_jpeg(path: Path):
    """
    Convert a TIFF image to JPEG format, processing different resolutions if available.

    Parameters:
    path (Path): The file path of the TIFF image.
    """
    try:
        # print(f"Processing {path}")  # Debugging statement
        image = skimage.io.MultiImage(str(path))  # Load the multi-resolution TIFF image
        # print(f"Loaded {len(image)} resolutions for {path}")  # Debugging statement
        # Always process the first resolution
        image_to_jpeg(path, '_0', image[0])
        # Process additional resolutions if present
        if len(image) > 1:
            image_to_jpeg(path, '_1', image[1])
        if len(image) > 2:
            image_to_jpeg(path, '_2', image[2])
    except Exception as e:
        print(f"Failed to process {path}: {e}")  # Print the error message

def image_to_jpeg(path: Path, suffix: str, image: np.ndarray):
    """
    Save a numpy image array as a JPEG file, cropping white borders and ensuring unique file names.

    Parameters:
    path (Path): The file path of the original TIFF image.
    suffix (str): A suffix to add to the JPEG file name.
    image (np.ndarray): The image data as a numpy array.
    """
    try:
        # print(f"Saving {path.stem}{suffix}.jpeg")  # Debugging statement
        output_dir = Path('C:/University of Limerick/AI_ML/code_exps/code_50/input/prostate-cancer-grade-assessment/train_images_jpeg')  # Define the output directory for JPEG images
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        jpeg_path = output_dir / f'{path.stem}{suffix}.jpeg'  # Define the path for the JPEG file
        if jpeg_path.exists():
            try:
                Image.open(jpeg_path).verify()  # Verify if the existing JPEG file is valid
                # print(f"{jpeg_path} already exists and is valid.")  # Debugging statement
            except Exception as e:
                print(f"Existing JPEG file is invalid: {e}")  # Debugging statement
            else:
                return  # If the file is valid, skip saving
        image = crop_white(image)  # Crop white borders from the image
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)  # Convert uint16 to uint8 if needed
        pil_image = Image.fromarray(image)  # Convert the numpy array to a PIL Image
        pil_image.save(jpeg_path, format='JPEG', quality=90)  # Save the image as a JPEG file
        # print(f"Saved {jpeg_path}")  # Debugging statement
    except Exception as e:
        print(f"Failed to save {path}: {e}")  # Print the error message and image details
        print(f"Image details: dtype={image.dtype}, shape={image.shape}, path={jpeg_path}")

def main():
    """
    Main function to process all TIFF images in the specified directory and convert them to JPEG format.
    """
    paths = list(Path('C:/University of Limerick/AI_ML/code_exps/code_50/input/prostate-cancer-grade-assessment/train_images').glob('*.tiff'))  # Get all TIFF files in the directory
    # print(f"Found {len(paths)} TIFF files.")  # Debugging statement
    with multiprocessing.Pool(processes=4) as pool:  # Limit number of processes to reduce memory usage
        # Use multiprocessing to convert images in parallel and display progress with tqdm
        for _ in tqdm.tqdm(pool.imap_unordered(to_jpeg, paths), total=len(paths)):
            pass

if __name__ == '__main__':
    main()  # Run the main function if the script is executed directly
