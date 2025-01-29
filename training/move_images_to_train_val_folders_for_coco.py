import json
import shutil
import os

# Paths
TRAIN_JSON_PATH = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\coco_data_train.json"  # Path to the train JSON
VAL_JSON_PATH = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\coco_data_val.json"  # Path to the validation JSON
SOURCE_IMAGE_FOLDER = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\images"  # Path to source images folder
DESTINATION_TRAIN_FOLDER = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\train_images"  # Destination folder for train images
DESTINATION_VAL_FOLDER = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\val_images"  # Destination folder for validation images

# Ensure destination folders exist
os.makedirs(DESTINATION_TRAIN_FOLDER, exist_ok=True)
os.makedirs(DESTINATION_VAL_FOLDER, exist_ok=True)

# Function to copy images
def copy_images(json_path, destination_folder):
    # Load JSON annotations
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Loop through the annotations and copy images
    for annotation in data['images']:
        image_name = annotation['file_name']
        source_image_path = os.path.join(SOURCE_IMAGE_FOLDER, image_name)
        destination_image_path = os.path.join(destination_folder, image_name)

        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, destination_image_path)
            print(f"Copied: {image_name}")
        else:
            print(f"Image not found: {image_name}")

# Copy train images
print("Copying train images...")
copy_images(TRAIN_JSON_PATH, DESTINATION_TRAIN_FOLDER)

# Copy validation images
print("Copying validation images...")
copy_images(VAL_JSON_PATH, DESTINATION_VAL_FOLDER)