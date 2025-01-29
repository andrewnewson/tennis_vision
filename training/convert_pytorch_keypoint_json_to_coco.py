import json
import os
from PIL import Image

# Paths
INPUT_JSON = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\data_train.json"
OUTPUT_JSON = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\coco_data_train.json"
IMAGE_FOLDER = r"C:\Users\AndrewNewson\Downloads\tennis_court_det_dataset\data\images"

# Define your 14 keypoints (Modify names as needed)
keypoint_names = [
    "top_left", "top_right", "bottom_left", "bottom_right",
    "singles_top_left", "singles_bottom_left", "singles_top_right", "singles_bottom_right",
    "service_top_left", "service_top_right", "service_bottom_left", "service_bottom_right",
    "service_top_middle", "service_bottom_middle"
]

# Function to convert to COCO keypoint format
def convert_to_coco():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    coco_data = {"images": [], "annotations": [], "categories": []}
    annotation_id = 1
    image_id = 1

    for item in data:
        image_name = f"{item['id']}.png"
        keypoints = item["kps"]  # Expected format: [[x1, y1], [x2, y2], ...]

        if len(keypoints) != 14:
            print(f"Skipping {image_name}: Expected 14 keypoints, got {len(keypoints)}")
            continue

        # Load image dimensions
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found, skipping...")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        # Compute bounding box
        x_coords, y_coords = zip(*keypoints)
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        # Convert keypoints to COCO format
        coco_keypoints = [coord for point in keypoints for coord in (*point, 2)]

        # Add image metadata
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Add annotation
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "keypoints": coco_keypoints,
            "num_keypoints": 14
        })

        annotation_id += 1
        image_id += 1

    # Add category metadata
    coco_data["categories"].append({
        "id": 1,
        "name": "tennis_court",
        "keypoints": keypoint_names,
        "skeleton": []  # No need for skeleton connections
    })

    # Save to COCO JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO keypoint dataset saved as {OUTPUT_JSON}")

# Run conversion
convert_to_coco()