import tensorflow as tf
import json
from pathlib import Path

# Dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25

# Check dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError("Dataset folder not found!")

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Save class names
class_names = train_data.class_names

print("\nPlant Species:")
for name in class_names:
    print(name)

with open(BASE_DIR / "species_class_names.json", "w") as f:
    json.dump(class_names, f)

# Improve data loading speed
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model separately
model.save(BASE_DIR / "species_model.keras")

print("\nTraining completed!")
print("Species model saved successfully.")