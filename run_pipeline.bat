@echo off
echo YOLOv11 Military Vehicle and Personnel Segmentation Pipeline
echo ========================================================
echo.

echo Step 1: Installing required packages...
echo ----------------------------------------
pip install ultralytics opencv-python numpy matplotlib tqdm pyyaml
echo.

echo Step 2: Preparing the dataset...
echo ------------------------------
python prepare_dataset.py
echo.

echo Step 3: Training the YOLOv11 segmentation model...
echo -----------------------------------------------
python train_model.py
echo.

echo Step 4: Running inference on test images...
echo ---------------------------------------
python inference.py --source images --conf 0.25
echo.

echo Pipeline completed successfully!
echo Results are available in the 'results' directory.
echo.

pause