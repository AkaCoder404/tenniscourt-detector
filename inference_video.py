import cv2
import numpy as np
import torch
import argparse
from ultralytics import YOLO
from utils import plot_keypoints, plot_lines

def main():
    parser = argparse.ArgumentParser("Video Inference")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_path", type=str, default="runs/pose/train/weights/best.pt", help="Path to YOLO model weights")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path for output video")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load the trained model
    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    print("Starting video inference...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run inference
        results = model.predict(frame, conf=0.5, device=device, verbose=False)

        # Draw on main frame
        annotated_frame = frame.copy()

        for r in results:
            if r.keypoints is not None and r.keypoints.data is not None and len(r.keypoints.data) > 0 and r.keypoints.data.shape[1] > 0:
                # Assuming 1 detection per frame (the tennis court)
                kps = r.keypoints.data[0].cpu().numpy() # Shape [14, 3]

                # Convert to normalized format required by utils.py
                norm_kps = []
                for kp in kps:
                    x, y, conf = kp
                    norm_kps.append((x / frame_width, y / frame_height, conf))
                
                # Draw keypoints and lines on original frame
                annotated_frame = plot_keypoints(annotated_frame, norm_kps, color=(0, 0, 255))
                annotated_frame = plot_lines(annotated_frame, norm_kps, grid="tennis")

        out.write(annotated_frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Inference complete! Output saved to {args.output_path}")

if __name__ == "__main__":
    main()
