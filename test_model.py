import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# configurations
MODEL_PATH = 'saved_model/my_model'
WEIGHTS_PATH = 'saved_weight/my_checkpoint'
EPOCHS = 10
IMG_HEIGHT = 180
IMG_WIDTH = 180

# create model
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Restore the weights
model.load_weights(WEIGHTS_PATH)

# Restore entire model NOTE: you can remove model defination if if you want to use this option.
#model = keras.models.load_model(MODEL_PATH)

for img_path in sys.argv[1:]:
  img = keras.preprocessing.image.load_img(
      img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  class_names = ['correct_mask', 'no_mask']
  print(predictions[0])
  print(np.max(score), score)
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
