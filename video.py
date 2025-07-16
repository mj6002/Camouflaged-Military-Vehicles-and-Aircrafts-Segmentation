import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"D:\Projects_folder\Object_Segmentation\runs\train\yolov11_military_seg2\weights\best.pt")  # change to your model path

# Open video file (or use 0 for webcam)
video_path = "sample1.mp4"  # replace with your video file
cap = cv2.VideoCapture(video_path)

# Get video details
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500, 500))

    # Run YOLO inference on the frame
    results = model(frame)

    # The results[0].plot() returns frame with detections drawn
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Aircraft Detection", annotated_frame)

    # Write to output file
    out.write(annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
