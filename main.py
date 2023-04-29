import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy
import collections
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = datasets.load_digits()

# Get the input images from the dataset
data = digits.images

# Calculate the moments of the input images
moments = []
for img in data:
    # Calculate the moments
    M = cv2.moments(img)
    moments.append(M)

# Convert the moments to a pandas DataFrame
df_data = pd.DataFrame(moments)

# Convert the DataFrame to float32
df_data = df_data.astype('float32')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test =train_test_split(df_data, digits.target,
test_size=0.2, shuffle=False)

# Define the neural network model
model = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(df_data.shape[1],)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics =['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

# Print the test accuracy
print('\nTest accuracy:', test_acc)

model.save("handwritten.model")





