"""Train a multi-class model: crop + each weed species.

Dataset layout (folder name = class name):
    dataset/
        crop/
        Parthenium/
        Nutsedge/
        Bermuda Grass/
        ...
Weed folder names should match the keys in weed_info.py (WEEDS) so that
the app can show the details for each species.

Run:  python day6_species_train.py
Creates: model.keras (new multi-class model) and class_names.json
"""
import json
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64          # app.py also resizes to 64x64 - keep them equal
DATASET_PATH = "dataset"

# 1. Classes come from folder names
class_names = sorted(
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
)
num_classes = len(class_names)
print("Classes:", class_names)

# 2. Load images
data, labels = [], []
for label, category in enumerate(class_names):
    folder = os.path.join(DATASET_PATH, category)
    count = 0
    for name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            continue
        data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        labels.append(label)
        count += 1
    print(f"  {category}: {count} images")

data = np.array(data) / 255.0
labels = np.array(labels)
labels_cat = to_categorical(labels, num_classes)

# 3. Split (stratified so every class appears in train and test)
X_train, X_test, y_train, y_test, lab_train, _ = train_test_split(
    data, labels_cat, labels, test_size=0.2, stratify=labels, random_state=42
)

# 4. Class weights help when some species have fewer images
weights = compute_class_weight("balanced", classes=np.unique(lab_train), y=lab_train)
class_weight = {int(c): float(w) for c, w in zip(np.unique(lab_train), weights)}

# 5. Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# 6. CNN (same as before, only the output layer changes)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=40,
    validation_data=(X_test, y_test),
    class_weight=class_weight,
    callbacks=[early_stop],
)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# 7. Save model + class names (order must match training labels)
model.save("model.keras")
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False)

print("Saved model.keras and class_names.json")
