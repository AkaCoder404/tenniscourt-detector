import os
import json
import shutil
import zipfile

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, "datasets")
    orig_path = os.path.join(datasets_dir, "data")
    yolo_data_path = os.path.join(datasets_dir, "tennis_data")
    
    os.makedirs(datasets_dir, exist_ok=True)

    # Use the locally provided zip file
    zip_path = os.path.join(base_dir, "tennis_court_det_dataset.zip")
    
    if not os.path.exists(orig_path):
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
        print("Extraction complete.")
    else:
        print("Original dataset already exists, skipping extraction.")

    print("Converting to YOLO format...")
    
    image_height = 720
    image_width = 1280
    center_x = image_width / 2
    center_y = image_height / 2

    # Process Train Data
    train_images_path = os.path.join(yolo_data_path, "images", "train")
    train_labels_path = os.path.join(yolo_data_path, "labels", "train")
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)

    data_train = os.path.join(orig_path, "data_train.json")
    if os.path.exists(data_train):
        with open(data_train, "r") as f:
            data = json.load(f)

        for item in data:
            image_name = item["id"]
            kps = item["kps"]

            original_image_path = os.path.join(orig_path, "images", image_name + ".png")
            new_image_path = os.path.join(train_images_path, image_name + ".png")
            if os.path.exists(original_image_path):
                shutil.copyfile(original_image_path, new_image_path)

            label_file_path = os.path.join(train_labels_path, image_name + ".txt")
            with open(label_file_path, "w") as label_file:
                label_file.write("0 ")
                label_file.write(f"{center_x / image_width:.6f} {center_y / image_height:.6f} 1.000000 1.000000 ")
                for kp in kps:
                    x, y = kp
                    x = x / image_width
                    y = y / image_height
                    v = 2.0
                    if x < 0 or x > 1 or y < 0 or y > 1:
                        x = 0; y = 0; v = 0
                    label_file.write(f"{x:.6f} {y:.6f} {v:.6f} ")

    # Process Val Data
    valid_images_path = os.path.join(yolo_data_path, "images", "val")
    valid_labels_path = os.path.join(yolo_data_path, "labels", "val")
    os.makedirs(valid_images_path, exist_ok=True)
    os.makedirs(valid_labels_path, exist_ok=True)

    data_valid = os.path.join(orig_path, "data_val.json")
    if os.path.exists(data_valid):
        with open(data_valid, "r") as f:
            data = json.load(f)

        for item in data:
            image_name = item["id"]
            kps = item["kps"]

            original_image_path = os.path.join(orig_path, "images", image_name + ".png")
            new_image_path = os.path.join(valid_images_path, image_name + ".png")
            if os.path.exists(original_image_path):
                shutil.copyfile(original_image_path, new_image_path)

            label_file_path = os.path.join(valid_labels_path, image_name + ".txt")
            with open(label_file_path, "w") as label_file:
                label_file.write("0 ")
                label_file.write(f"{center_x / image_width:.6f} {center_y / image_height:.6f} 1.000000 1.000000 ")
                for kp in kps:
                    x, y = kp  # Fixed bug here where x, y were not unpacked
                    x = x / image_width
                    y = y / image_height
                    v = 2.0
                    if x < 0 or x > 1 or y < 0 or y > 1:
                        x = 0; y = 0; v = 0
                    label_file.write(f"{x:.6f} {y:.6f} {v:.6f} ")

    print("YOLO dataset successfully formatted at: ", yolo_data_path)

if __name__ == "__main__":
    main()
