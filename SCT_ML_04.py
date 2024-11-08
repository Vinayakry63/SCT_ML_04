import os
import logging
import warnings
import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disable oneDNN optimizations for consistent results
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(filename='training_steps.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the gesture recognition model training process.")

# Define categories and image size
CATEGORIES = ["01_palm", '02_l', '03_fist', '04_fist_moved', '05_thumb',
              '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']
IMG_SIZE = 50
logging.info("Categories and image size defined.")

# Load image data
data_path = r"D:\Task4\leapGestRecog\leapGestRecog"  # Update this path based on your directory structure
image_data = []
logging.info("Starting to load image data.")

for dr in os.listdir(data_path):
    for category in CATEGORIES:
        class_index = CATEGORIES.index(category)
        path = os.path.join(data_path, dr, category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is not None:  # Check if image is loaded
                    image_data.append([cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)), class_index])
            except Exception as e:
                logging.error(f"Error loading image {img} in category {category}: {e}")

logging.info("Image data loading complete. Total images loaded: %d", len(image_data))

# Shuffle the data
import random
random.shuffle(image_data)
logging.info("Image data shuffled.")

# Prepare input and labels
input_data = []
label = []
for X, y in image_data:
    input_data.append(X)
    label.append(y)

# Convert to arrays
input_data = np.array(input_data) / 255.0  # Normalize
label = to_categorical(np.array(label))  # One-hot encode labels
input_data.shape = (-1, IMG_SIZE, IMG_SIZE, 1)
logging.info("Data prepared and normalized.")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.2, random_state=42, shuffle=True)
logging.info("Data split into training and testing sets.")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
logging.info("Data augmentation configuration completed.")

# Build the CNN model
model = keras.models.Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(CATEGORIES), activation='softmax'))
logging.info("CNN model structure built.")

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
logging.info("Model compiled.")

# Model Checkpointing
checkpoint = ModelCheckpoint('best_gesture_recognition_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
logging.info("Checkpointing and early stopping configured.")

# Train the model
logging.info("Starting model training.")
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=20, callbacks=[early_stopping, checkpoint])

# Plot training history
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='red')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
logging.info("Model training complete and history plotted.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
logging.info('Test accuracy: {:.2f}%'.format(test_accuracy * 100))

# Save the final model
model.save('gesture_recognition_final_model.keras')
logging.info("Final model saved.")
