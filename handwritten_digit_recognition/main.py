import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# loading data

mnist = tf.keras.datasets.mnist

# splitting data

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# normalising data

x_train = x_train / 255.0
x_test = x_test / 255.0

# ḷoading model

model = tf.keras.models.Sequential()

# adding layer

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# adding dense layer

model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#  combining all

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

#  train model

model.fit(x_train, tf.keras.utils.to_categorical(y_train), epochs=10)



#  saving a model

model.save("handwritten.model")


loss, accuracy = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test))

print(loss)
print(accuracy)
