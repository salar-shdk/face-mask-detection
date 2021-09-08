import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# configurations
OPENCV_FACE_DETECTOR = 'haarcascade_frontalface_default.xml'
MODEL_WEIGHTS = 'saved_weight/my_checkpoint'
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['correct_mask', 'no_mask']

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
model.load_weights(MODEL_WEIGHTS)

# Load the cascade
face_cascade = cv2.CascadeClassifier(OPENCV_FACE_DETECTOR)

# capture video from webcam. 
cap = cv2.VideoCapture(0)
# use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (start_x, start_y, width, height) in faces:
        end_x, end_y = start_x + width, start_y + height
        face = frame[start_y:end_y, start_x:end_x]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (180, 180))
        img_array = keras.preprocessing.image.img_to_array(face)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
        )

        label = f'{CLASS_NAMES[np.argmax(score)]}, {100 * np.max(score)}'
        color = (0, 255, 0) if 'correct' in label  else (0, 0, 255)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    # Display
    cv2.imshow('img', frame)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
