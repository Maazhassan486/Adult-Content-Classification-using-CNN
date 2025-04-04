
# Adult Content Classification using CNN

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images as either "Adult" or "Non-Adult" content. The code is designed to:

- Train a model on a dataset organized by class.
- Save and load the model to avoid retraining.
- Predict the class of new images using the trained model.

## Dataset Structure

Your dataset should be organized in the following directory structure:

```
P2datasetFull/
├── train/
│   ├── 1/    # Contains non-adult images
│   └── 2/    # Contains adult images
├── test1/
│   ├── 1/    # Contains non-adult images
│   └── 2/    # Contains adult images
└── val/      # (Optional) Validation data with the same subdirectory structure as train/
    ├── 1/    # Contains non-adult images
    └── 2/    # Contains adult images
```

> **Note:** If you plan to fine-tune or validate your model further, ensure the validation set is properly set up as shown above.

## Requirements

- **Python:** 3.6 or higher
- **TensorFlow:** 2.x
- **Pillow:** For image processing
- **Matplotlib:** For plotting and visualization
- **NumPy:** For numerical operations

Install the required packages using pip:

```bash
pip install tensorflow pillow matplotlib numpy
```

## Usage

### 1. Training and Saving the Model

- The script first checks if a saved model exists at the specified path.  
- If the model exists, it loads the model; otherwise, it proceeds with training.
- The model is then saved in Keras' native format with the `.keras` extension.

To run the training process, execute:

```bash
python cnn1.py
```

### 2. Loading an Existing Model and Making Predictions

- If a saved model exists, the script loads it to avoid retraining.
- The provided `predict_image` function allows you to classify a new image.
- Update the `img_path` variable in the script with the path to the image you wish to classify, then run the script.

Example:

```python
img_path = r"path_to_your_image.jpg"
predict_image(img_path, model)
```

This will display the image along with the predicted class ("Adult" or "Non-Adult").

## Code Overview

- **Data Augmentation and Preprocessing:**  
  The code uses `ImageDataGenerator` to apply real-time data augmentation (rotation, width/height shifts, shearing, zooming, and horizontal flipping) to the training images and rescale pixel values to the range [0, 1].

- **Model Architecture:**  
  The CNN model consists of:
  - Three convolutional layers with increasing filter sizes.
  - Max pooling layers to reduce spatial dimensions.
  - A flatten layer followed by dense layers.
  - A final dense layer with a sigmoid activation function for binary classification.

- **Training and Evaluation:**  
  The model is trained using the training data and validated using the test data. After training, the model's accuracy is evaluated on the test set.

- **Saving/Loading Model:**  
  The model is saved to a specified path after training. If the model file already exists, it is loaded instead of retraining.

- **Prediction Function:**  
  The `predict_image` function preprocesses a given image and uses the loaded model to predict whether it contains adult content.

## Customization

- **Hyperparameters:**  
  You can adjust the `epochs`, `batch_size`, and augmentation parameters within the `ImageDataGenerator`.

- **File Paths:**  
  Modify the file paths for the training, testing, validation directories, and the model save path to match your environment.

- **Validation Set Usage:**  
  The code currently uses the test set for validation during training. If you wish to use the validation set (`val` folder) for monitoring training performance or fine-tuning, add or modify the corresponding `ImageDataGenerator` and flow settings.

## License

This project is provided for educational purposes. You are free to modify and use it according to your needs.

