# Camouflaged Military Vehicles and Aircrafts Segmentation using YOLOv11

This project implements object detection and segmentation for camouflaged military vehicles and aircrafts using YOLOv11. The implementation includes dataset preparation, model training, and inference scripts.

## Project Structure

```
c:\Object Segmentation/
├── images/                  # Original dataset images
├── data.yaml               # Dataset configuration file
├── prepare_dataset.py      # Script to prepare the dataset
├── train_model.py          # Script to train the YOLOv11 model
├── inference.py            # Script to run inference on new images
├── README.md               # This file
└── runs/                   # Training outputs (created during training)
    └── train/
        └── yolov11_military_seg/
            └── weights/
                └── best.pt  # Best trained model
```

## Setup

1. Install the required dependencies:

```bash
pip install ultralytics opencv-python numpy matplotlib tqdm pyyaml
```

2. Prepare the dataset:

```bash
python prepare_dataset.py
```

This script will:
- Split the images into training and validation sets
- Generate segmentation labels using a pre-trained YOLOv11 model
- Create the necessary directory structure

3. Train the YOLOv11 segmentation model:

```bash
python train_model.py
```

This script will:
- Train a YOLOv11 segmentation model on the prepared dataset
- Validate the trained model
- Run inference on sample images

4. Run inference on new images:

```bash
python inference.py --source path/to/images --conf 0.25
```

## Dataset

The dataset consists of images of camouflaged military vehicles and personnel. The classes include:

- Person
- Soldier
- Sniper
- Tank
- Aircraft
- Military Vehicle

## Model

The implementation uses YOLOv11 segmentation models from the Ultralytics framework. By default, it uses the `yolo11n-seg.pt` model (nano version) for faster processing, but you can switch to larger models (`s`, `m`, `l`, or `x` variants) for better accuracy.

## Inference

The inference script supports various options:

```bash
python inference.py --help
```

Common options include:

- `--model`: Path to the trained model (default: best.pt)
- `--source`: Path to input images or directory (default: images)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--img-size`: Inference size (default: 640)
- `--device`: Device to use (empty for auto)
- `--save-txt`: Save results to *.txt
- `--save-conf`: Save confidences in --save-txt labels
- `--hide-labels`: Hide labels
- `--hide-conf`: Hide confidences
- `--alpha`: Mask transparency (0-1, default: 0.5)

## Results

The inference results will be saved to the `results` directory, including:

- Visualized images with segmentation masks and bounding boxes
- Text files with detection results (if `--save-txt` is specified)

## Customization

You can customize the model by modifying the following files:

- `data.yaml`: Update the class names and paths
- `prepare_dataset.py`: Adjust the class mapping and dataset preparation logic
- `train_model.py`: Modify training parameters such as epochs, batch size, etc.
- `inference.py`: Change visualization settings and inference parameters

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLOv11 Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [Instance Segmentation Datasets](https://docs.ultralytics.com/datasets/segment/)
