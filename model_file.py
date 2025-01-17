import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

directory = 'input/leapGestRecog/leapGestRecog/00/'
folders = [f for f in os.listdir(directory) if not f.startswith('.')]
lookup = {folder: idx for idx, folder in enumerate(folders)}
reverselookup = {idx: folder for idx, folder in enumerate(folders)}
# print(lookup)

x_data = []
y_data = []
IMG_SIZE = 150
num_classes = 10
datacount = 0 
for i in range(0, num_classes):
    for j in os.listdir('input/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir('input/leapGestRecog/0' + str(i) + '/' + j + '/'):
                path = 'input/leapGestRecog/0' + str(i) + '/' + j + '/' + k
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
y_data=to_categorical(y_data)
x_data = x_data.reshape((datacount, IMG_SIZE, IMG_SIZE, 1))
x_data = x_data/255
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.30,random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Metric to be monitored
    factor=0.2,            # Factor by which the learning rate will be reduced
    patience=3,            # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=0.0001          # Lower bound on the learning rate
)

early_stopping = EarlyStopping(
    monitor='val_loss',    # Metric to be monitored
    patience=5,            # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train model 
batch_size = 8
epochs = 10

# Train model with callbacks
history = model.fit(
    x_train, y_train, 
    validation_data=(x_test, y_test), 
    epochs=epochs, 
    batch_size=batch_size,
    callbacks=[reduce_lr, early_stopping]  # Add the callbacks here
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')