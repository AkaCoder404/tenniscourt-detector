from ultralytics import YOLO
import os
import torch
import argparse
import time

# Get arguments
parser = argparse.ArgumentParser('training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
parser.add_argument('--epochs', default=10, type=int, help='number of epoch in training')
parser.add_argument('--save_model', default=True, type=bool, help='save the model')
parser.add_argument('--imgsz', default=640, type=int, help='image size')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

model = YOLO('yolov8s-pose.pt')

start_time = time.time()
model.train(data="datasets_config/tennis.yml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            plots=True,
            save=args.save_model)


end_time = time.time()
print("-" * 50)
print(f"Total training time {end_time - start_time}")
print("-" * 50)
