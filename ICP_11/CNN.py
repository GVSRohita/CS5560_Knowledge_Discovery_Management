# Import library
import numpy as np
from keras.datasets import cifar10
from keras.datasets import mnist, fashion_mnist
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from matplotlib import pyplot as plt

# Loading input data
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Identify classes associated with the fashion mnist data set
fashionClass = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

# Identify images corresponding to their class names
print("Random 5 Images from the Training Data:")
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(fashionClass[train_labels[i]])
    plt.show()

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Define parameters for the model
num_filters = 8
filter_size = 3
pool_size = 2

# Define the model to be build
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Training the model using train data identified
fashionModel = model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)

# Identify the summary of the model built
print("Summary of the model built:")
print(model.summary())

# Predict on the first 5 test images
predictions = model.predict(test_images[:5])

# Print our model's predictions
print("These are the model predictions :")
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]
print("\n")

# Check our predictions against the ground truths
print("These are the corresponding labels :")
print(test_labels[:5]) # [7, 2, 1, 0, 4]

# Evaluate the performance of the model built
# plot the acc and val_acc
plt.plot(fashionModel.history['accuracy'])
plt.plot(fashionModel.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.show()

# plot the loss and val_loss
plt.plot(fashionModel.history['loss'])
plt.plot(fashionModel.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

