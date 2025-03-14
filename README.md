# MultimediaWork
# Fashion-MNIST Classification with CNN

## Overview
This project involves training a convolutional neural network (CNN) to classify images from the Fashion-MNIST dataset. The dataset contains 70,000 grayscale images of 10 different fashion categories. The model is built using TensorFlow and Keras.

## Dataset
The Fashion-MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- Image size: 28x28 pixels
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install tensorflow pandas numpy seaborn matplotlib scikit-learn
```

## Project Structure
- Import Libraries**: Load required Python packages.
- Load Dataset**: Download and prepare the Fashion-MNIST dataset.
- Data Preprocessing**: Normalize and reshape images, apply one-hot encoding.
- Model Building**: Define and compile a CNN model.
- **Training the Model**: Train the CNN with early stopping and data augmentation.
- **Evaluation**: Assess performance using confusion matrix and classification reports.

## Usage
Run the Jupyter Notebook to execute the model training and evaluation process:
```bash
jupyter notebook fashionmnist.ipynb
```

## Results
The trained CNN achieves high accuracy in classifying Fashion-MNIST images. Evaluation metrics include:
- Accuracy
- Loss Curve
- Confusion Matrix





