import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from shutil import copyfile

# --- CONFIGURATION ---
# IMPORTANT: Update this path to where you have downloaded the Nutrition5k dataset
NUTRITION5K_PATH = Path("path/to/your/nutrition5k_dataset") 

# How many images to process for a test run. 
# Set to None to process the entire dataset.
LIMIT = 200 

# Path to the metadata folder within the dataset
METADATA_PATH = NUTRITION5K_PATH / "metadata"

# Where to save the converted YOLO dataset
OUTPUT_PATH = Path("./dataset_yolo")

def get_bounding_box_from_mask(mask_path: Path) -> list:
    """Calculates the bounding box from a segmentation mask image."""
    if not mask_path.exists():
        return None
    
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    
    # Find indices of non-zero pixels
    rows, cols = np.where(mask_np > 0)
    
    if len(rows) == 0:
        return None
        
    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def convert_to_yolo_format(box: list, img_width: int, img_height: int) -> tuple:
    """Converts [xmin, ymin, xmax, ymax] to YOLO's [x_center, y_center, width, height] normalized format."""
    x_min, y_min, x_max, y_max = box
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return (x_center * dw, y_center * dh, width * dw, height * dh)

def process_dataset():
    """
    Main function to convert the Nutrition5k dataset to YOLO format.
    """
    if not NUTRITION5K_PATH.exists() or NUTRITION5K_PATH == Path("path/to/your/nutrition5k_dataset"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: Dataset path is not configured correctly.")
        print(f"Please open this script ('convert_nutrition5k.py') and update the")
        print(f"'NUTRITION5K_PATH' variable to the correct location.")
        print(f"Current incorrect path is: {NUTRITION5K_PATH}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print("--- Starting Nutrition5k to YOLO conversion ---")
    if LIMIT is not None:
        print(f"---!!! WARNING: Running in TEST MODE. Will process only {LIMIT} images. !!!---")
        print(f"---!!! To process the full dataset, set LIMIT = None in the script. !!!---")


    # 1. Load dish metadata and create class mapping
    dishes_df = pd.read_csv(METADATA_PATH / "dishes.csv")
    class_names = dishes_df['name'].tolist()
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes.")

    # 2. Create output directories
    print(f"Will create YOLO dataset at: {OUTPUT_PATH.resolve()}")
    yolo_images_train = OUTPUT_PATH / "images" / "train"
    yolo_labels_train = OUTPUT_PATH / "labels" / "train"
    yolo_images_valid = OUTPUT_PATH / "images" / "valid"
    yolo_labels_valid = OUTPUT_PATH / "labels" / "valid"
    
    os.makedirs(yolo_images_train, exist_ok=True)
    os.makedirs(yolo_labels_train, exist_ok=True)
    os.makedirs(yolo_images_valid, exist_ok=True)
    os.makedirs(yolo_labels_valid, exist_ok=True)

    # 3. Process each dish
    dish_metadata_path = METADATA_PATH / "dish_metadata.json"
    with open(dish_metadata_path, 'r') as f:
        all_dishes_meta = json.load(f)

    # Apply the limit if it's set
    if LIMIT is not None:
        all_dishes_meta = all_dishes_meta[:LIMIT]

    print(f"Processing {len(all_dishes_meta)} total dish entries...")
    
    for dish_meta in tqdm(all_dishes_meta, desc="Converting dishes"):
        dish_id = dish_meta['dish_id']
        dish_name = dish_meta['dish_name']
        
        if dish_name not in class_to_id:
            continue 
        class_id = class_to_id[dish_name]

        original_img_path = NUTRITION5K_PATH / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
        mask_path = NUTRITION5K_PATH / "imagery" / "realsense_overhead" / dish_id / "segmentation.png"

        if not original_img_path.exists():
            continue

        box = get_bounding_box_from_mask(mask_path)
        if not box:
            continue
            
        with Image.open(original_img_path) as img:
            img_width, img_height = img.size

        yolo_box = convert_to_yolo_format(box, img_width, img_height)

        is_train = (hash(dish_id) % 10) < 8 
        
        yolo_img_dir = yolo_images_train if is_train else yolo_images_valid
        yolo_label_dir = yolo_labels_train if is_train else yolo_labels_valid

        copyfile(original_img_path, yolo_img_dir / f"{dish_id}.png")

        label_path = yolo_label_dir / f"{dish_id}.txt"
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")
    
    # 4. Create data.yaml file
    print("Creating data.yaml file...")
    data_yaml = {
        'path': str(OUTPUT_PATH.resolve()),
        'train': str((OUTPUT_PATH / "images" / "train").resolve()),
        'val': str((OUTPUT_PATH / "images" / "valid").resolve()),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = OUTPUT_PATH / "nutrition5k_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("\n--- Conversion Complete! ---")
    print(f"YOLO dataset created at: {OUTPUT_PATH.resolve()}")
    print(f"YAML configuration file created at: {yaml_path.resolve()}")
    print("\nNext steps:")
    print(f"1. Check that the paths in '{yaml_path.resolve()}' are correct.")
    print("2. Run training: python train_yolo.py")

if __name__ == "__main__":
    process_dataset()
