# Dataset Preparation for Vision Transformers
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define dataset paths
data_dir = "path_to_dataset"
annotations_file = os.path.join(data_dir, "annotations.csv")
images_dir = os.path.join(data_dir, "images")

# Load annotations
def load_annotations(annotations_file):
    df = pd.read_csv(annotations_file)
    print(f"Loaded {len(df)} annotations.")
    return df

annotations = load_annotations(annotations_file)

# Split dataset into training and validation
train_df, val_df = train_test_split(annotations, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# Function for preprocessing images
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess images to fit input requirements of Vision Transformers.
    - Resize to target_size (default is 224x224 for ViT).
    - Normalize pixel values to range [0, 1].
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    return image

# Prepare datasets
def prepare_dataset(df, images_dir, target_size=(224, 224)):
    """
    Prepare image datasets for training and validation.
    - Process each image and corresponding label.
    - Returns numpy arrays for images and labels.
    """
    images = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(images_dir, row['filename'])
        image = preprocess_image(image_path, target_size)
        label = [row['x_min'], row['y_min'], row['x_max'], row['y_max'], row['class']]
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Process datasets for ViT input dimensions
train_images, train_labels = prepare_dataset(train_df, images_dir)
val_images, val_labels = prepare_dataset(val_df, images_dir)

# Save preprocessed data for Vision Transformer models
np.save("train_images_vit.npy", train_images)
np.save("train_labels_vit.npy", train_labels)
np.save("val_images_vit.npy", val_images)
np.save("val_labels_vit.npy", val_labels)

print("Dataset preparation for Vision Transformer models completed and saved.")
