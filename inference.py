import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from tqdm import tqdm

# Define paths
ROOT_DIR = r'D:\Projects_folder\Object_Segmentation'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'runs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define class colors (BGR format)
COLORS = {
    0: (0, 255, 0),     # person
    1: (0, 165, 255),   # soldier
    2: (0, 0, 255),     # sniper
    3: (255, 0, 0),     # tank
    4: (255, 255, 0),   # aircraft
    5: (128, 0, 128),   # military_vehicle
}

# Class labels
CLASS_NAMES = {
    0: 'person',
    1: 'soldier',
    2: 'sniper',
    3: 'tank',
    4: 'aircraft',
    5: 'military_vehicle',
}

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 Segmentation Inference')
    parser.add_argument('--model', type=str, default=r'D:\Projects_folder\Object_Segmentation\runs\train\yolov11_military_seg2\weights\best.pt',
                        help='Path to trained model (best.pt)')
    parser.add_argument('--source', type=str, default='test', help='Image or directory path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--device', type=str, default='', help='Device to run inference on')
    parser.add_argument('--save-txt', action='store_true', help='Save labels to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='Save confidence scores to *.txt')
    parser.add_argument('--hide-labels', action='store_true', help='Hide labels on image')
    parser.add_argument('--hide-conf', action='store_true', help='Hide confidence scores on image')
    parser.add_argument('--alpha', type=float, default=0.5, help='Mask transparency (0 to 1)')
    return parser.parse_args()

def process_image(model, image_path, args):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None, None
    
    results = model(image_path, conf=args.conf, iou=args.iou, imgsz=args.img_size)

    vis_img = img.copy()

    for result in results:
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                cls_id = int(box[5])
                conf = float(box[4])
                
                if cls_id not in CLASS_NAMES:
                    continue
                
                color = COLORS.get(cls_id, (255, 255, 255))
                class_name = CLASS_NAMES[cls_id]
                
                # Resize mask to image size
                mask_binary = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]))
                colored_mask = np.zeros_like(img)
                colored_mask[mask_binary > 0] = color
                
                vis_img = cv2.addWeighted(vis_img, 1, colored_mask, args.alpha, 0)

                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

                if not args.hide_labels:
                    label = f"{class_name}"
                    if not args.hide_conf:
                        label += f" {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(vis_img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                    cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return vis_img, results

def run_inference(args):
    model_path = args.model
    if not os.path.isabs(model_path):
        alt_model = os.path.join(OUTPUT_DIR, 'train', 'yolov11_military_seg', 'weights', 'best.pt')
        if os.path.exists(alt_model):
            model_path = alt_model

    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)

    if os.path.isdir(args.source):
        image_paths = [os.path.join(args.source, f) for f in os.listdir(args.source)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    elif os.path.isfile(args.source):
        image_paths = [args.source]
    else:
        print(f"❌ Invalid input source: {args.source}")
        return

    print(f"🔍 Found {len(image_paths)} image(s). Starting inference...")

    for image_path in tqdm(image_paths):
        filename = os.path.splitext(os.path.basename(image_path))[0]
        result_img, results = process_image(model, image_path, args)

        if result_img is not None:
            output_path = os.path.join(RESULTS_DIR, f"{filename}_result.jpg")
            if cv2.imwrite(output_path, result_img):
                print(f"✅ Saved: {output_path}")
            else:
                print(f"❌ Failed to save: {output_path}")

            if args.save_txt:
                txt_path = os.path.join(RESULTS_DIR, f"{filename}.txt")
                with open(txt_path, 'w') as f:
                    for result in results:
                        if result.boxes is not None:
                            boxes = result.boxes.data.cpu().numpy()
                            for box in boxes:
                                cls_id = int(box[5])
                                conf = float(box[4])
                                x1, y1, x2, y2 = box[:4]

                                img_h, img_w = result_img.shape[:2]
                                cx = (x1 + x2) / 2 / img_w
                                cy = (y1 + y2) / 2 / img_h
                                w = (x2 - x1) / img_w
                                h = (y2 - y1) / img_h

                                line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                                if args.save_conf:
                                    line += f" {conf:.4f}"
                                f.write(line + "\n")

    print(f"\n🏁 Inference complete. Results are saved in: {RESULTS_DIR}")

def main():
    args = parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()
