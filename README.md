# TennisCourtDetectorwithYolov8

The goal of this project is to see the performance of YOLOv8 at detecting the 14 keypoints of a tennis court to map the lines and perspective.

![](images/output3.png)

## Environment Setup
The project uses Python 3.12 and PyTorch. Setup the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Preparation
The dataset consists of tennis court images with 14 annotated keypoints per image.

1. Ensure the dataset archive `tennis_court_det_dataset.zip` is located in the project root.
2. Run the preparation script to extract the dataset and convert it into the YOLO format required by ultralytics:

```bash
python prepare_dataset.py
```

This will extract the zip file and create a `datasets/tennis_data` folder inside your project directory containing the YOLO images and labels formatted according to `datasets_config/tennis.yml`.

## Training the Model
To train the YOLOv8-pose model on the tennis court dataset:

```bash
python train.py --epochs 10 --batch_size 8
```

*Hardware: Trained on a NVIDIA GeForce RTX 2070 SUPER, Python 3.12, torch 2.6.0+cu124*

*(Adjust `--epochs` and `--batch_size` according to your hardware capabilities. The model uses `yolov8s-pose.pt` as the base model and saves results to `runs/pose/train/` or `runs/pose/train2/`.)*

## Evaluation
To evaluate the trained model:
- You can run the `evaluate.ipynb` Jupyter Notebook.
- This will output the mAP50 and mAP50-95 scores, as well as visualize the predicted keypoints vs ground truth on a test set. It also demonstrates mapping the court perspective to a 2D top-down reference court.

### Training Results (Epoch 10, Batch Size 8, Imgsz 640)


After 10 epochs of training the `yolov8s-pose.pt` base model, the following validation metrics were achieved:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **mAP50(B)** | 99.50% | Bounding box detection precision (court presence) |
| **mAP50(P)** | 99.49% | Keypoint prediction accuracy (corners & intersections) |
| **mAP50-95(P)** | 99.31% | Extremely strict keypoint mapping precision |
| **Recall(B)** | 100.0% | Zero missed courts in the validation dataset |

## Video Inference
To test the line and point detection on a live tennis video:
1. Obtain a `.mp4` video of a tennis match.
2. Run the inference script, specifying your input video and the path to your best trained weights:
   ```bash
   python inference_video.py --video_path your_video.mp4 --model_path runs/pose/train/weights/best.pt --output_path output.mp4
   ```
   This will output a video showing the tennis court lines and keypoints overlaid directly on the original footage.

![](images/infer_video_tennis.mp4)

## Extra

Homography View of Court
![](images/output1.png)

## Acknowledgments
Thanks to and inspired by https://github.com/yastrebksv/TennisCourtDetector. The dataset can also be found at this GitHub repository.