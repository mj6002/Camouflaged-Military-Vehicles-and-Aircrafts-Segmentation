import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml

# Define paths
ROOT_DIR = r"D:/Projects_folder/Object_Segmentation"
DATA_YAML = os.path.join(ROOT_DIR, 'data.yaml')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'runs')

# Training configuration
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
WORKERS = 4

def train_model():
    print("Starting YOLOv11 segmentation model training...")
    
    # Load a pre-trained YOLOv11 segmentation model
    model = YOLO('yolo11n-seg.pt')  # You can also use 's', 'm', 'l', or 'x' variants for better accuracy
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        patience=10,  # Early stopping patience
        device='0',   # Use GPU if available
        project=os.path.join(OUTPUT_DIR, 'train'),
        name='yolov11_military_seg',
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=True,
        save=True,
        save_period=10,
        verbose=True
    )
    
    print("Training completed!")
    return results

def validate_model():
    print("\nValidating the trained model...")
    
    # Load the best trained model
    best_model_path = os.path.join(OUTPUT_DIR, 'train', 'yolov11_military_seg', 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        return
    
    model = YOLO(best_model_path)
    
    # Validate the model
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device='0',
        project=os.path.join(OUTPUT_DIR, 'val'),
        name='yolov11_military_seg',
        plots=True,
        verbose=True
    )
    
    print("\nValidation Results:")
    print(f"Segmentation mAP50-95: {metrics.seg.map:.4f}")
    print(f"Segmentation mAP50: {metrics.seg.map50:.4f}")
    print(f"Segmentation mAP75: {metrics.seg.map75:.4f}")
    
    return metrics

def predict_samples():
    print("\nRunning inference on sample images...")
    
    # Load the best trained model
    best_model_path = os.path.join(OUTPUT_DIR, 'train', 'yolov11_military_seg', 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        return
    
    model = YOLO(best_model_path)
    
    # Get some sample images from validation set
    val_images_dir = os.path.join(ROOT_DIR, 'images', 'val')
    sample_images = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir)[:5] 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not sample_images:
        print("No sample images found for prediction.")
        return
    
    # Run prediction
    results = model.predict(
        source=sample_images,
        imgsz=IMG_SIZE,
        conf=0.25,
        iou=0.45,
        max_det=300,
        device='0',
        project=os.path.join(OUTPUT_DIR, 'predict'),
        name='yolov11_military_seg',
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        visualize=True,
        verbose=True
    )
    
    print(f"Prediction completed. Results saved to {os.path.join(OUTPUT_DIR, 'predict', 'yolov11_military_seg')}")
    return results

def main():
    # Check if dataset is prepared
    train_labels_dir = os.path.join(ROOT_DIR, 'labels', 'train')
    val_labels_dir = os.path.join(ROOT_DIR, 'labels', 'val')
    
    if not os.path.exists(train_labels_dir) or not os.path.exists(val_labels_dir):
        print("Dataset not prepared. Please run prepare_dataset.py first.")
        return
    
    # Update data.yaml with correct paths
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_config['train'] = 'images/train'
    data_config['val'] = 'images/val'
    
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Train the model
    train_results = train_model()
    
    # Validate the model
    val_metrics = validate_model()
    
    # Run prediction on sample images
    predict_results = predict_samples()
    
    print("\nYOLOv11 segmentation model training and evaluation completed!")

if __name__ == "__main__":
    main()