from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Define the path to save or load the model
save_path = r"C:\Users\maazg\PycharmProjects\PythonProject\.venv\cnn1model.keras"

# Check if the model exists
if os.path.exists(save_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(save_path)
else:
    print("No existing model found. Proceeding with training...")
    # Define paths to your training and testing directories.
    train_dir = r"C:\Users\maazg\Downloads\Adult content dataset\P2datasetFull\train"
    test_dir = r"C:\Users\maazg\Downloads\Adult content dataset\P2datasetFull\test1"
    # Define image parameters
    target_size = (150, 150)  # Resize images to 150x150 pixels
    batch_size = 32

    # Create an ImageDataGenerator for data augmentation on the training set and rescaling on both sets.
    train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
     )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load training data from directory. The subdirectory names (1 and 2) will be used as class labels.
    train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'  # Because this is a binary classification problem
     )

# Load testing data from directory.
    test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
     )

# Build a convolutional neural network model.
    model = tf.keras.models.Sequential([
    # Convolutional layer with 32 filters and a kernel size of 3x3.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Additional convolutional layers.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the output and add dense layers.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),

    # Final output layer with 1 neuron and sigmoid activation for binary classification.
    tf.keras.layers.Dense(1, activation='sigmoid')
     ])

 # Compile the model with the binary crossentropy loss and the adam optimizer.
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model. Adjust the number of epochs as needed.
    epochs = 10
    history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
     )


# Evaluate the model on the test dataset.
    test_loss, test_accuracy = model.evaluate(test_generator)
    print("Test accuracy:", test_accuracy)

# Optionally, display some predictions.
# Get one batch of test images and labels.
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)
# Convert probabilities to binary predictions (0: non-adult, 1: adult)
    predicted_classes = (predictions > 0.5).astype("int32")

    plt.figure(figsize=(12, 12))
    for i in range(9):
      plt.subplot(3, 3, i + 1)
      plt.imshow(test_images[i])
      pred_label = "Adult" if predicted_classes[i] == 1 else "Non-Adult"
      true_label = "Adult" if test_labels[i] == 1 else "Non-Adult"
      plt.title(f"Predicted: {pred_label}\nActual: {true_label}")
      plt.axis('off')
      plt.show()
     # Save the model to a file.
    save_path = r"C:\Users\maazg\PycharmProjects\PythonProject\.venv\cnn1model.keras"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = "Adult" if prediction[0][0] > 0.5 else "Non-Adult"

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Example usage:
# Provide the path to the image you want to classify
img_path = r"C:\Users\maazg\Documents\university assignments\AI\assignment1\21L-5793\Screenshots_Assignment1\Screenshot 2025-04-03 165348.png"
predict_image(img_path, model)

