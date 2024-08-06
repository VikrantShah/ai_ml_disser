import pandas as pd
import shutil
import os

# Function to extract a specified number of rows and save to a new CSV file
def extract_rows(input_file_path, num_rows=100):
    data = pd.read_csv(input_file_path)
    extracted_data = data.head(num_rows)
    
    # Create a custom output file name based on the number of rows extracted
    output_dir = f'C:/University of Limerick/AI_ML/code_exps/code_{num_rows}/input/prostate-cancer-grade-assessment'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file_path = os.path.join(output_dir, 'train.csv')
    
    extracted_data.to_csv(output_file_path, index=False)
    return extracted_data, output_file_path

# Function to copy image files to a new directory
def copy_images(image_names, source_dir, dest_dir, file_extension='tiff'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for image_name in image_names:
        source_path = os.path.join(source_dir, f"{image_name}.{file_extension}")
        dest_path = os.path.join(dest_dir, f"{image_name}.{file_extension}")
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)

# Function to copy image mask files to a new directory
def copy_images_mask(image_names, source_dir, dest_dir, file_extension='tiff'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for image_name in image_names:
        source_path = os.path.join(source_dir, f"{image_name}_mask.{file_extension}")
        dest_path = os.path.join(dest_dir, f"{image_name}_mask.{file_extension}")
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)

input_file_path = 'D:/input/prostate-cancer-grade-assessment/train.csv'
source_image_dir = 'D:/input/prostate-cancer-grade-assessment/train_images'
source_image_dir_mask = 'D:/input/prostate-cancer-grade-assessment/train_label_masks'
num_rows = 250  # Specify the number of rows wanted to extract

# Extract rows and save to new CSV file with a custom name
extracted_data, output_file_path = extract_rows(input_file_path, num_rows)
print(f"Extracted {num_rows} rows and saved to {output_file_path}")

# Define the destination directory for the images
dest_image_dir = f'C:/University of Limerick/AI_ML/code_exps/code_{num_rows}/input/prostate-cancer-grade-assessment/train_images'
dest_image_dir_mask = f'C:/University of Limerick/AI_ML/code_exps/code_{num_rows}/input/prostate-cancer-grade-assessment/train_label_masks'
image_names = extracted_data['image_id'].tolist()

# Copy the images
copy_images(image_names, source_image_dir, dest_image_dir)
print(f"Copied {len(image_names)} images to {dest_image_dir}")

# Copy the image masks
copy_images_mask(image_names, source_image_dir_mask, dest_image_dir_mask)
print(f"Copied {len(image_names)} image masks to {dest_image_dir_mask}")
