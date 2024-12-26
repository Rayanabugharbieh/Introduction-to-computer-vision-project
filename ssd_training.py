# Step 1: Dataset Preparation
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
def preprocess_image(image_path, target_size=(300, 300)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    return image

# Prepare datasets
def prepare_dataset(df, images_dir, target_size=(300, 300)):
    images = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(images_dir, row['filename'])
        image = preprocess_image(image_path, target_size)
        label = [row['x_min'], row['y_min'], row['x_max'], row['y_max'], row['class']]
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = prepare_dataset(train_df, images_dir)
val_images, val_labels = prepare_dataset(val_df, images_dir)

# Save preprocessed data
np.save("train_images.npy", train_images)
np.save("train_labels.npy", train_labels)
np.save("val_images.npy", val_images)
np.save("val_labels.npy", val_labels)

print("Dataset preparation completed and saved.")

# Step 2: SSD Model Training
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load preprocessed data
train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
val_images = np.load("val_images.npy")
val_labels = np.load("val_labels.npy")

# Define SSD Model
def create_ssd_model(input_shape=(300, 300, 3), num_classes=3):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(num_classes + 4, activation="sigmoid")(x)  # Classes + 4 for bounding box coordinates
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Compile the model
ssd_model = create_ssd_model()
ssd_model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="mse",  # Use mean squared error for bounding box regression
                  metrics=["accuracy"])

# Define callbacks
checkpoint = ModelCheckpoint("ssd_model.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model
history = ssd_model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=80,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# Save the training history
with open("ssd_training_history.npy", "wb") as f:
    np.save(f, history.history)

print("SSD model training completed.")
