import os
path = "/content/drive/MyDrive/photos"
classes= os.listdir(path)
print(classes)

from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.preprocessing import image_dataset_from_directory

base_dir = "/content/drive/MyDrive/photos"

# Create datasets
train_datagen = image_dataset_from_directory(base_dir,
image_size=(200,200),
subset='training',
seed = 1,
validation_split=0.1,
       batch_size= 32)

test_datagen = image_dataset_from_directory(base_dir,
image_size=(200,200),
subset='validation',
seed = 1,
validation_split=0.1,
batch_size= 32)

import tensorflow as tf

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D


model = tf.keras.models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
layers.MaxPooling2D(2, 2),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),

layers.Flatten(),
layers.Dense(512, activation='relu'),
layers.BatchNormalization(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.1),
layers.BatchNormalization(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.2),
layers.BatchNormalization(),
layers.Dense(1, activation='sigmoid') #output layer
])


import tensorflow as tf

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D


model = tf.keras.models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
layers.MaxPooling2D(2, 2),


layers.Flatten(),
layers.Dense(512, activation='relu'),

layers.Dense(1, activation='sigmoid') #output layer
])



keras.utils.plot_model(
model,
show_shapes=True,
show_dtype=True,
show_layer_activations=True
)

model.compile(
loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
history = model.fit(train_datagen,
epochs=15,
validation_data=test_datagen,
verbose=1)

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
test_image = image.load_img("/content/drive/MyDrive/photos/test/shoe/shoe1.jpg", target_size=(200, 200))
plt.imshow(test_image)
plt.axis('off')  # Optional: Hide the axis for a cleaner image display
plt.show()

# Convert the image to an array and expand dimensions
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.0  # Normalize to match model input expectations

# Predict using the model
result = model.predict(test_image)
print(result)

# Interpret the result
if result[0][0] > 0.5:  # Assuming the model outputs probabilities
    print("shoe")
else:
    print("slipper")

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2

# 1. Model Construction
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a Flatten layer before the Dense layers
model.add(Flatten())

# Alternatively, use GlobalAveragePooling2D to reduce dimensions
# model.add(GlobalAveragePooling2D())

# Add dense layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

# Add output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# 2. Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Training the Model
history = model.fit(train_datagen,
                    epochs=10,
                    validation_data=test_datagen,
                    verbose=1)


from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
test_image = image.load_img("/content/drive/MyDrive/photos/test/slipper/slipper1.jpg", target_size=(200, 200))
plt.imshow(test_image)
plt.axis('off')  # Optional: Hide the axis for a cleaner image display
plt.show()

# Convert the image to an array and expand dimensions
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.0  # Normalize to match model input expectations

# Predict using the model
result = model.predict(test_image)
print(result)

# Interpret the result
if result[0][0] > 0.5:  # Assuming the model outputs probabilities
    print("slipper")
else:
    print("shoe")
