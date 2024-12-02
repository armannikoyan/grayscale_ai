import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Paths to dataset
train_black_path = './dataset/data/train_black'
train_color_path = './dataset/data/train_color'
IMG_SIZE = 128  # Reduced image size to avoid memory issues
BATCH_SIZE = 8  # Smaller batch size to fit in memory

# Data generator class
class DataGenerator(Sequence):
    def init(self, black_dir, color_dir, batch_size=BATCH_SIZE):
        self.black_paths = [os.path.join(black_dir, f) for f in os.listdir(black_dir)]
        self.color_paths = [os.path.join(color_dir, f) for f in os.listdir(color_dir)]
        self.batch_size = batch_size

    def len(self):
        return int(np.ceil(len(self.black_paths) / self.batch_size))

    def getitem(self, idx):
        batch_black_paths = self.black_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_color_paths = self.color_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        grayscale_images, color_images = [], []
        for black_path, color_path in zip(batch_black_paths, batch_color_paths):
            # Load grayscale image
            gray_img = cv2.imread(black_path, cv2.IMREAD_GRAYSCALE)
            gray_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE)) / 255.0
            grayscale_images.append(gray_img[..., np.newaxis])

            # Load color image
            color_img = cv2.imread(color_path)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_img = cv2.resize(color_img, (IMG_SIZE, IMG_SIZE)) / 255.0
            color_images.append(color_img)

        return np.array(grayscale_images), np.array(color_images)

# Create training data generator
train_gen = DataGenerator(train_black_path, train_color_path, batch_size=BATCH_SIZE)

# Define the colorization model
def build_colorization_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))  # Grayscale input

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # Reduced filters
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    # Decoder
    up1 = UpSampling2D((2, 2))(conv3)
    upconv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)

    up2 = UpSampling2D((2, 2))(upconv1)
    upconv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(upconv2)  # RGB output

    return Model(inputs, outputs)

# Build and compile the model
model = build_colorization_model()
model.compile(optimizer='adam', loss='mse')
model.summary()

# Save the best model during training
checkpoint = ModelCheckpoint('colorization_model.keras', save_best_only=True, monitor='loss', mode='min')

# Train the model with data generator
history = model.fit(
    train_gen,
    epochs=20,
    callbacks=[checkpoint]
)

print("Training complete. Model saved as 'colorization_model.keras'")
