import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Vary brightness
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Load the trained model
    model = tf.keras.models.load_model("model/skin_disease_model.h5")
    
    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(val_generator)

    # Print accuracy
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    return train_generator, val_generator

if __name__ == "__main__":
    train_path = "dataset/train_set"
    train_gen, val_gen = load_data(train_path)
    print(f"Classes Detected: {train_gen.class_indices}")
