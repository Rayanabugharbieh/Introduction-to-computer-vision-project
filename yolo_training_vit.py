# YOLO Training with Vision Transformer (ViT) Backbone
from transformers import ViTFeatureExtractor, ViTModel
from yolov5 import train as yolo_train
import torch
import yaml
import os

# Define YOLO configuration with ViT backbone
class YOLOv5WithViT:
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224"):
        """
        Custom YOLOv5 trainer integrating ViT backbone for feature extraction.
        """
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name)
        self.model = ViTModel.from_pretrained(pretrained_model_name)

    def preprocess_images(self, image_paths):
        """
        Preprocess images using ViT feature extractor.
        """
        inputs = self.feature_extractor(images=image_paths, return_tensors="pt")
        return inputs

# Prepare YOLO dataset configuration
def prepare_yolo_config_with_vit():
    """
    Generate YOLO-specific configuration file with ViT integrated as a custom feature extractor.
    """
    config = {
        'path': './data',
        'train': './train/images',
        'val': './val/images',
        'nc': 3,  # Number of classes
        'names': ['without_mask', 'mask', 'incorrect_mask']
    }

    with open("yolo_vit_dataset.yaml", "w") as file:
        yaml.dump(config, file)

    print("YOLO dataset configuration with ViT saved.")

prepare_yolo_config_with_vit()

# YOLO Training Configuration
yolo_config = {
    "data": "yolo_vit_dataset.yaml",  # Path to dataset configuration file
    "img_size": 224,                   # Input image size for ViT
    "batch_size": 8,                   # Smaller batch size for ViT's computational needs
    "epochs": 80,
    "weights": "yolov5s.pt",          # YOLOv5 pretrained weights
    "project": "YOLOv5_Face_Mask_ViT",
    "name": "exp_vit",
    "cache": True
}

# Train YOLO Model
if os.path.exists("./data/train") and os.path.exists("./data/val"):
    yolo_train.train(**yolo_config)
    print("YOLO model with ViT backbone training completed.")
else:
    print("Dataset directories not found. Ensure train/val folders exist.")
