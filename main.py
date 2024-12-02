import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Paths to dataset and model
test_black_path = './dataset/data/test_black'
test_color_path = './dataset/data/test_color'
model_path = 'colorization_model.keras'  # Corrected model path with .keras extension
IMG_SIZE = 128  # Match the training image size

# Function to load grayscale and color images
def load_image_pairs(black_dir, color_dir):
    grayscale_images = []
    color_images = []
    for file_name in os.listdir(black_dir):
        black_path = os.path.join(black_dir, file_name)
        color_path = os.path.join(color_dir, file_name)
        if os.path.exists(black_path) and os.path.exists(color_path):
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

# Load testing data
X_test, y_test = load_image_pairs(test_black_path, test_color_path)
print(f"Testing data loaded: {X_test.shape}, {y_test.shape}")

# Load the trained model
model = load_model(model_path)
print("Model loaded!")

# Predict on test data
predicted_colors = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) between ground truth and predicted colors
mse = mean_squared_error(y_test.reshape(-1, 3), predicted_colors.reshape(-1, 3))
print(f"Mean Squared Error (MSE) on test set: {mse}")

# Visualize results
def display_images(gray, color, predicted, n=10):
    plt.figure(figsize=(15, 10))  # Increased figure size to display more images
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

# Display results for more images
display_images(X_test, y_test, predicted_colors, n=10)  # Show 10 images
