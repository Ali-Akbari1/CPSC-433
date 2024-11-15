# Author: Ali Akbari, 30171539
# Course: CPSC 433 F24 T04

import tensorflow as tf
import numpy as np
import pandas as pd

tf.random.set_seed(1234)

print("--Load data--")
# Load the train and test CSV files
train_data = pd.read_csv("draft_train.csv")
test_data = pd.read_csv("draft_test.csv")

# Separate the features (X) and labels (y) for training and testing datasets
X_train = train_data.drop(columns=['drafted']).values
y_train = train_data['drafted'].values
X_test = test_data.drop(columns=['drafted']).values
y_test = test_data['drafted'].values

print("--Process data--")
# Get all unique positions and create a mapping of position names to numbers
unique_positions = train_data['position'].unique()
position_mapping = {position: idx for idx, position in enumerate(unique_positions)}

# Map position names to their corresponding numbers in the training and testing data
train_data['position'] = train_data['position'].map(position_mapping)
test_data['position'] = test_data['position'].map(position_mapping)

# Convert data back to NumPy arrays after mapping
X_train = train_data.drop(columns=['drafted']).values
y_train = train_data['drafted'].values
X_test = test_data.drop(columns=['drafted']).values
y_test = test_data['drafted'].values

# One-hot encode the "position" column to represent positions as binary vectors
position_column_index = 0  # "position" is the first column
num_positions = len(unique_positions)

# Initialize empty arrays for the one-hot encoded position data
X_train_position = np.zeros((X_train.shape[0], num_positions))
X_test_position = np.zeros((X_test.shape[0], num_positions))

# Fill in the one-hot encoded arrays for training and testing data
for i in range(X_train.shape[0]):
    X_train_position[i, int(X_train[i, position_column_index])] = 1
for i in range(X_test.shape[0]):
    X_test_position[i, int(X_test[i, position_column_index])] = 1

# Replace the original "position" column with the one-hot encoded columns
X_train = np.concatenate([X_train_position, X_train[:, 1:]], axis=1)
X_test = np.concatenate([X_test_position, X_test[:, 1:]], axis=1)


# Normalize the numerical features (excluding the one-hot encoded columns)
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

print("--Make model--")
# Neural Network Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.BatchNormalization(), # Normalize the layer outputs
    tf.keras.layers.Dropout(0.5), # Dropout for regularization
    tf.keras.layers.Dense(256, activation='relu'),  # Hidden layer
    tf.keras.layers.BatchNormalization(), # Normalize the layer outputs
    tf.keras.layers.Dropout(0.5), # Dropout for regularization
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the training data and validate on the test data
print("--Fit model--")
history = model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=2, validation_data=(X_test, y_test))

# Evaluate the model's performance on both training and testing data
print("--Evaluate model--")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Train / Test Accuracy: {train_acc*100:.1f}% / {test_acc*100:.1f}%")

# Save the trained model
model.save("CFLModel-Complete.keras")
