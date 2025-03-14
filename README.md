# MultimediaWork
# README.md

# Fashion MNIST Classification

## Overview
This project builds a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The dataset contains grayscale images of 10 clothing categories.

## Installation
To set up the environment, install dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Model
1. Ensure all dependencies are installed.
2. Run the Python script:
   ```bash
   python fashionmnist.py
   ```
3. The model will train and display accuracy metrics.
4. Sample predictions will be visualized using Matplotlib.

## Files
- `fashionmnist.py`: The main script for training and evaluating the model.
- `requirements.txt`: List of dependencies.
- `README.md`: Documentation and setup instructions.

## Notes
- Ensure you have Python 3.8 or later installed.
- If running in PyCharm and facing display issues, set the Matplotlib backend using:
  ```python
  import matplotlib
  matplotlib.use('TkAgg')
  ```

## Author
Created by our group
