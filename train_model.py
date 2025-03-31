import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from load_data import load_data
from tensorflow.keras.optimizers import Adam

train_path = "dataset/train_set"
train_gen, val_gen = load_data(train_path)

# Use MobileNetV2 as feature extractor
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze layers to retain pre-trained features

# Unfreeze last 4 layers for fine-tuning
for layer in base_model.layers[-4:]:  
    layer.trainable = True

# Model Define
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),  # Extra dense layer
    Dropout(0.5),
    Dense(len(train_gen.class_indices), activation='softmax')  # Dynamic output
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_gen, validation_data=val_gen, epochs=40, steps_per_epoch=len(train_gen), validation_steps=len(train_gen))

# Save Model
model.save("model/skin_disease_model.h5")
