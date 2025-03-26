import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# ✅ Step 1: Define dataset path
data_path = r"C:\Users\yedur\OneDrive\Desktop\glucomadetection\dataset"
# ✅ Step 2: Verify dataset path exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Error: Dataset path not found - {data_path}")
else:
    print(f"✅ Dataset found at: {data_path}")

# ✅ Step 3: Define categories
categories = {"glaucoma": 1, "normal": 0}  # Adjust if your dataset has different folder names

# ✅ Step 4: Check if subfolders exist
for category in categories.keys():
    folder_path = os.path.join(data_path, category)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Error: Folder '{folder_path}' does not exist!")
    else:
        print(f"✅ Found category folder: {folder_path}")

# ✅ Step 5: Load and preprocess images
img_size = (224, 224)

def load_images(data_path):
    images = []
    labels = []

    for category, label in categories.items():
        folder_path = os.path.join(data_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize image
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

print("📥 Loading dataset...")
X, y = load_images(data_path)
print(f"✅ Dataset Loaded: {X.shape}, Labels: {y.shape}")

# ✅ Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# ✅ Step 7: Build the MobileNetV2 Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

# Add classifier layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)  # Binary classification

# Define final model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ✅ Step 8: Train the Model
print("🚀 Training model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# ✅ Step 9: Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")

# ✅ Step 10: Save Model
model.save("glaucoma_model.h5")
print("✅ Model saved as 'glaucoma_model.h5'")

