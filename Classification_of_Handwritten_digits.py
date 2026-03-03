# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models

#Load the MNIST dataset
mnist = tf.keras.dataset.mnist
(x_train, y_train) = (x_test, y_test) = mnist.load_data()

#   Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape = (28, 28)),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation = 'softmax')
])
# Compile the model
model.compile(Optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
model.fix(x_train, y_train, epochs = 5)

#Evaluate the model
model.evaluate(x_test, y_test)