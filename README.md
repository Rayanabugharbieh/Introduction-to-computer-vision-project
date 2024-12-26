# Real-Time Object Detection and Classification of Face Masks

## Project Overview
This project focuses on developing a real-time system for detecting and classifying face masks. The system detects three classes: correctly-worn masks, incorrectly-worn masks, and no masks. It leverages two state-of-the-art object detection models: 

1. **SSD (Single Shot Multibox Detector)**
2. **YOLO (You Only Look Once)**
3. **Bonus: SSD and YOLO with Vision Transformer (ViT) Backbone**

### Key Features
- Dataset preprocessing and augmentation.
- Training two object detection models (SSD and YOLO).
- Enhanced versions of SSD and YOLO using Vision Transformers (ViT) for feature extraction.
- Real-time inference using a webcam or video input.

---

## File Structure

### Scripts
1. **`dataset_preparation.py`**
   - Prepares the dataset for CNN-based training.
   - Includes data splitting, preprocessing, and saving.

2. **`dataset_preparation_vit.py`**
   - Prepares the dataset for ViT-based training.
   - Ensures compatibility with Vision Transformer input requirements.

3. **`ssd_training.py`**
   - Implements the SSD model using MobileNetV2 as the backbone.
   - Trains the model and saves the best-performing weights.

4. **`ssd_training_vit.py`**
   - Implements the SSD model using a Vision Transformer backbone.
   - Trains the model and saves the best-performing weights.

5. **`yolo_training.py`**
   - Configures and trains the YOLO model using YOLOv5.
   - Uses CNN-based feature extraction.

6. **`yolo_training_vit.py`**
   - Configures and trains the YOLO model using Vision Transformer as the backbone.

7. **`full_workflow.py`** (Optional)
   - Combines all steps for an end-to-end execution of the project.

### Other Files
- **`yolo_dataset.yaml`**: Configuration file for YOLO training (CNN-based).
- **`yolo_vit_dataset.yaml`**: Configuration file for YOLO training (ViT-based).
- **Saved Models**:
  - `ssd_model.h5`: Trained SSD model weights with MobileNetV2 backbone.
  - `ssd_vit_model.h5`: Trained SSD model weights with ViT backbone.
  - YOLO weights are saved within the YOLOv5 project folder.

---

## Requirements

### Dependencies
Ensure the following Python libraries are installed:
- TensorFlow
- OpenCV
- Numpy
- Pandas
- Scikit-learn
- PyTorch (for YOLOv5)
- Transformers (for ViT)
- tqdm

### Dataset
Download the "Face Mask Detection" dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection). Extract and organize it in the following structure:
```
data/
    train/
        images/
        labels/
    val/
        images/
        labels/
```

---

## How to Run

1. **Dataset Preparation**:
   - For CNN-based models:
     ```bash
     python dataset_preparation.py
     ```
   - For ViT-based models:
     ```bash
     python dataset_preparation_vit.py
     ```

2. **Train SSD Model**:
   - CNN Backbone:
     ```bash
     python ssd_training.py
     ```
   - ViT Backbone:
     ```bash
     python ssd_training_vit.py
     ```

3. **Train YOLO Model**:
   - CNN Backbone:
     ```bash
     python yolo_training.py
     ```
   - ViT Backbone:
     ```bash
     python yolo_training_vit.py
     ```

4. **Full Workflow** (Optional):
   Execute the entire workflow in one step:
   ```bash
   python full_workflow.py
   ```

---

## Evaluation
The models are evaluated using:
- **Precision and Recall**
- **IoU (Intersection over Union)**
- **mAP (Mean Average Precision)**

The evaluation metrics are logged during training and can be visualized for both CNN and ViT-based models.

---

## Real-Time Inference
To test the models in real-time:
1. Use the SSD or YOLO model weights (CNN or ViT-based).
2. Implement a real-time detection script (not included here but can be added).
3. Use a webcam or video file as input.

---

## Notes
- Ensure that you have a GPU available for training.
- YOLOv5 requires specific setup instructions available in the [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5).
- Vision Transformers require the `transformers` library from Hugging Face.

---

## Credits
Dataset: [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

Developed as part of a deep learning project on object detection.
