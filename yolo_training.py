# YOLO Model Training
from yolov5 import train as yolo_train

# Define YOLO configuration
def prepare_yolo_config():
    with open("yolo_dataset.yaml", "w") as f:
        f.write(
            """\npath: ./data\ntrain: ./train/images\nval: ./val/images\nnc: 3\nnames: ['without_mask', 'mask', 'incorrect_mask']\n"""
        )
    print("YOLO configuration file created.")

prepare_yolo_config()

# YOLO Training Configuration
yolo_config = {
    "data": "yolo_dataset.yaml",  # Path to the dataset configuration file
    "img_size": 416,               # Input image size
    "batch_size": 16,
    "epochs": 80,
    "weights": "yolov5s.pt",     # Pretrained YOLOv5 weights
    "project": "YOLOv5_Face_Mask_Detection",
    "name": "exp",
    "cache": True
}

# Train YOLO Model
yolo_train.train(**yolo_config)

print("YOLO model training completed.")
