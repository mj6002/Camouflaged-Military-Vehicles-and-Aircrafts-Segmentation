import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
from tqdm import tqdm

# Define paths
ROOT_DIR = r'd:\Object Segmentation'
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
LABELS_DIR = os.path.join(ROOT_DIR, 'labels')
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, 'images', 'train')
VAL_IMAGES_DIR = os.path.join(ROOT_DIR, 'images', 'val')
TRAIN_LABELS_DIR = os.path.join(ROOT_DIR, 'labels', 'train')
VAL_LABELS_DIR = os.path.join(ROOT_DIR, 'labels', 'val')

# Create necessary directories
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)

# Load a pre-trained YOLOv11 segmentation model
print("Loading pre-trained YOLOv11 segmentation model...")
model = YOLO('yolo11n-seg.pt')  # Use the nano model for faster processing

# Class mapping from COCO to our custom classes
# COCO classes: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 4: airplane
class_mapping = {
    0: 0,  # person -> person
    0: 1,  # person -> soldier (will be filtered by confidence)
    0: 2,  # person -> sniper (will be filtered by confidence)
    2: 5,  # car -> military_vehicle
    3: 5,  # motorcycle -> military_vehicle
    5: 5,  # bus -> military_vehicle
    7: 5,  # truck -> military_vehicle
    4: 4,  # airplane -> aircraft
    8: 3,  # boat -> tank (will be filtered and refined)
}

# Function to split dataset into train and validation sets
def split_dataset(images_dir, train_ratio=0.8):
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    np.random.shuffle(all_images)
    
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Total images: {len(all_images)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Move images to train and val directories
    for img in train_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join(TRAIN_IMAGES_DIR, img))
    
    for img in val_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join(VAL_IMAGES_DIR, img))
    
    return train_images, val_images

# Function to generate segmentation labels using the pre-trained model
def generate_labels(image_paths, source_dir, labels_dir):
    for img_name in tqdm(image_paths, desc="Generating labels"):
        img_path = os.path.join(source_dir, img_name)
        
        # Run inference with the model
        results = model(img_path, verbose=False)
        
        # Create a label file with the same name but .txt extension
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        
        with open(label_path, 'w') as f:
            for result in results:
                if result.masks is not None:
                    # Get image dimensions
                    img = cv2.imread(img_path)
                    img_height, img_width = img.shape[:2]
                    
                    # Process each detected object
                    for i, (cls, mask) in enumerate(zip(result.boxes.cls.cpu().numpy(), result.masks.xy)):
                        cls_id = int(cls)
                        conf = float(result.boxes.conf[i].cpu().numpy())
                        
                        # Skip if confidence is too low or class is not in our mapping
                        if conf < 0.5 or cls_id not in class_mapping:
                            continue
                        
                        # Map COCO class to our custom class
                        mapped_cls = class_mapping[cls_id]
                        
                        # Special handling for soldier vs sniper vs person
                        if cls_id == 0:  # person
                            # Simple heuristic: if person is small, likely a soldier in distance
                            box = result.boxes.xyxy[i].cpu().numpy()
                            box_width = box[2] - box[0]
                            box_height = box[3] - box[1]
                            box_area = box_width * box_height
                            img_area = img_width * img_height
                            
                            if box_area / img_area < 0.05:
                                mapped_cls = 1  # soldier
                            elif "sniper" in img_name.lower() or "ghillie" in img_name.lower():
                                mapped_cls = 2  # sniper
                            else:
                                mapped_cls = 0  # person
                        
                        # Special handling for tanks
                        if "tank" in img_name.lower() and cls_id in [2, 7]:  # car or truck
                            mapped_cls = 3  # tank
                        
                        # Normalize coordinates
                        normalized_mask = []
                        for x, y in mask:
                            normalized_mask.append(x / img_width)
                            normalized_mask.append(y / img_height)
                        
                        # Write to file: class_id x1 y1 x2 y2 ...
                        line = f"{mapped_cls} " + " ".join([f"{coord:.6f}" for coord in normalized_mask])
                        f.write(line + '\n')

# Main execution
def main():
    print("Preparing YOLOv11 segmentation dataset...")
    
    # Split dataset
    print("\nSplitting dataset into train and validation sets...")
    train_images, val_images = split_dataset(IMAGES_DIR)
    
    # Generate labels
    print("\nGenerating segmentation labels for training images...")
    generate_labels(train_images, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    
    print("\nGenerating segmentation labels for validation images...")
    generate_labels(val_images, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    
    print("\nDataset preparation complete!")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Labels directory: {LABELS_DIR}")

if __name__ == "__main__":
    main()