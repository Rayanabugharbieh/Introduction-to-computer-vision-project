# SSD Training with Vision Transformer (ViT) Backbone
import tensorflow as tf
from transformers import TFViTModel
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# Load preprocessed data
train_images = np.load("train_images_vit.npy")
train_labels = np.load("train_labels_vit.npy")
val_images = np.load("val_images_vit.npy")
val_labels = np.load("val_labels_vit.npy")

# Define SSD Model with Vision Transformer (ViT) Backbone
def create_ssd_vit_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Creates an SSD model using Vision Transformer (ViT) as the backbone.
    """
    vit_base = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Freeze ViT layers to use pre-trained features
    for layer in vit_base.layers:
        layer.trainable = False

    inputs = Input(shape=input_shape)
    vit_features = vit_base(inputs)[0]  # Extract features from ViT

    x = GlobalAveragePooling2D()(vit_features)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    output = Dense(num_classes + 4, activation="sigmoid")(x)  # Classes + bounding box coords

    model = Model(inputs, output)
    return model

# Compile the model
ssd_vit_model = create_ssd_vit_model()
ssd_vit_model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss="mse",  # Mean Squared Error for bounding box regression
                      metrics=["accuracy"])

# Define callbacks
checkpoint = ModelCheckpoint("ssd_vit_model.h5", save_best_only=True, monitor="val_loss", mode="min")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model
history = ssd_vit_model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=80,
    batch_size=16,  # Reduced batch size for ViT's higher computational needs
    callbacks=[checkpoint, early_stop]
)

# Save the training history
with open("ssd_vit_training_history.npy", "wb") as f:
    np.save(f, history.history)

print("SSD with Vision Transformer backbone training completed.")
