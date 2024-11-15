# Author: Ali Akbari, 30171539
# Course: CPSC 433 F24 T04

import tensorflow as tf
import numpy as np

tf.random.set_seed(1234)

print("--Get data--")
# Load the notMNIST dataset from a .npz file
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


print("--Process data--")
# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension to the data for compatibility with CNN layers
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
input_shape = (28, 28, 1)

print("--Make model--")
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape), # Convolutional layer
    tf.keras.layers.BatchNormalization(), # Normalize feature maps
    tf.keras.layers.MaxPooling2D((2, 2)), # Downsample feature maps
    tf.keras.layers.Dropout(0.25),  # Prevent overfitting
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), # Another convolutional layer
    tf.keras.layers.BatchNormalization(), # Normalize feature maps
    tf.keras.layers.MaxPooling2D((2, 2)), # Downsample feature maps
    tf.keras.layers.Dropout(0.25), # Prevent overfitting
    
    tf.keras.layers.Flatten(), # Flatten data for the fully connected layers
    tf.keras.layers.Dense(128, activation='relu'), # Fully connected layer
    tf.keras.layers.BatchNormalization(), # Normalize feature maps
    tf.keras.layers.Dropout(0.5), # Prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'), # Another dense layer
    tf.keras.layers.BatchNormalization(), # Normalize feature maps
    tf.keras.layers.Dropout(0.5), # Prevent overfitting
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes

])

# Compile the model with optimizer, loss function, and accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 20 epochs
print("--Fit model--")
model.fit(x_train, y_train, epochs=20, verbose=2)


# Evaluate the model's performance on training and testing data
print("--Evaluate model--")
model_loss1, model_acc1 = model.evaluate(x_train,  y_train, verbose=2)
model_loss2, model_acc2 = model.evaluate(x_test,  y_test, verbose=2)
print(f"Train / Test Accuracy: {model_acc1*100:.1f}% / {model_acc2*100:.1f}%")

# Save the trained model
model.save("notMNIST-Complete.keras")