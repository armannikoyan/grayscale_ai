import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

test_black_path = './dataset/data/test_black'
test_color_path = './dataset/data/test_color'
model_path = 'colorization_model.keras'
IMG_SIZE = 128

def load_image_pairs(black_dir, color_dir):
    grayscale_images = []
    color_images = []
    for file_name in os.listdir(black_dir):
        black_path = os.path.join(black_dir, file_name)
        color_path = os.path.join(color_dir, file_name)
        if os.path.exists(black_path) and os.path.exists(color_path):
            gray_img = cv2.imread(black_path, cv2.IMREAD_GRAYSCALE)
            gray_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE)) / 255.0
            grayscale_images.append(gray_img[..., np.newaxis])

            color_img = cv2.imread(color_path)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_img = cv2.resize(color_img, (IMG_SIZE, IMG_SIZE)) / 255.0
            color_images.append(color_img)

    return np.array(grayscale_images), np.array(color_images)

X_test, y_test = load_image_pairs(test_black_path, test_color_path)
print(f"Testing data loaded: {X_test.shape}, {y_test.shape}")

model = load_model(model_path)
print("Model loaded!")

predicted_colors = model.predict(X_test)

mse = mean_squared_error(y_test.reshape(-1, 3), predicted_colors.reshape(-1, 3))
print(f"Mean Squared Error (MSE) on test set: {mse}")

def display_images(gray, color, predicted, n=10):
    plt.figure(figsize=(15, 10))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(gray[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title('Grayscale')

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(color[i])
        plt.axis('off')
        plt.title('Ground Truth')

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(predicted[i])
        plt.axis('off')
        plt.title('Predicted')
    plt.show()

display_images(X_test, y_test, predicted_colors, n=10)
