# Introduction-to-computer-vision-project
Real-Time Object Detection and Classification of Face Masks

Project Overview

This project focuses on developing a real-time object detection system to identify and classify face masks. The system detects three classes: correctly-worn masks, incorrectly-worn masks, and no masks. It leverages two deep learning models:

SSD (Single Shot Multibox Detector)

YOLO (You Only Look Once)

Key Features

Dataset preprocessing and augmentation.

Training two object detection models (SSD and YOLO).

Real-time inference using a webcam or video input.

File Structure

Scripts

dataset_preparation.py

Prepares the dataset for training.

Includes data splitting, preprocessing, and saving.

ssd_training.py

Implements the SSD model using MobileNetV2 as the backbone.

Trains the model and saves the best-performing weights.

yolo_training.py

Configures and trains the YOLO model.

Uses YOLOv5 pretrained weights for fine-tuning.

full_workflow.py (Optional)

Combines all steps for an end-to-end execution of the project.

Other Files

yolo_dataset.yaml: Configuration file for YOLO training.

Saved Models:

ssd_model.h5: Trained SSD model weights.

YOLO weights are saved within the YOLOv5 project folder.

Requirements

Dependencies

Ensure the following Python libraries are installed:

TensorFlow

OpenCV

Numpy

Pandas

Scikit-learn

PyTorch (for YOLOv5)

tqdm

Dataset

Download the "Face Mask Detection" dataset from Kaggle. Extract and organize it in the following structure:

data/
    train/
        images/
        labels/
    val/
        images/
        labels/

How to Run

Dataset Preparation:
Run the dataset_preparation.py script to preprocess and split the dataset:

python dataset_preparation.py

Train SSD Model:
Train the SSD model with the following command:

python ssd_training.py

Train YOLO Model:
Train the YOLO model using YOLOv5:

python yolo_training.py

Full Workflow (Optional):
Execute the entire workflow in one step:

python full_workflow.py

Evaluation

The models are evaluated using:

Precision and Recall

IoU (Intersection over Union)

mAP (Mean Average Precision)

The evaluation metrics are logged during training and can be visualized for both SSD and YOLO models.

Real-Time Inference

To test the models in real-time:

Use the SSD or YOLO model weights.

Implement a real-time detection script (not included here but can be added).

Use a webcam or video file as input.

Notes

Ensure that you have a GPU available for training.

YOLOv5 requires specific setup instructions available in the YOLOv5 GitHub Repository.

Credits

Dataset: Kaggle - Face Mask Detection

Developed as part of a deep learning project on object detection.
