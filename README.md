# Grayscale to Color Image Colorization Using Deep Learning

## Overview
This project implements a deep learning-based approach to colorize grayscale images. It uses a convolutional neural network (CNN) with an encoder-decoder architecture to predict RGB colors for grayscale inputs. The model is trained on paired grayscale and color images and evaluates its performance using Mean Squared Error (MSE).

---

## Features
- Automatic colorization of grayscale images into RGB.
- Efficient training with a custom `DataGenerator` to handle large datasets.
- Visual comparisons of input, ground truth, and predictions.
- Save and load trained models for future use.

---

## Project Structure
📂 project/<br>
├── 📂 dataset/<br>
&nbsp;│&nbsp;├── 📂 train_black/ # Grayscale training images<br>
&nbsp;│&nbsp;├── 📂 train_color/ # Color training images<br>
&nbsp;│&nbsp;├── 📂 test_black/ # Grayscale testing images<br>
&nbsp;│&nbsp;├── 📂 test_color/ # Color testing images<br>
├── 📄 colorization_train.py # Training script<br>
├── 📄 colorization_test.py # Testing and visualization script<br>
├── 📄 README.md # Project documentation<br>
└── 📄 colorization_model.keras # Saved trained model<br>
