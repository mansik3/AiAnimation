import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

# Data Preprocessing
# Parameters
video_folder_path = 'videos'
img_height, img_width = 64, 64
num_classes = 3 
max_frames = 100

def load_videos(video_folder_path, max_frames):
    X = []
    y = []
    for class_folder in os.listdir(video_folder_path):
        class_path = os.path.join(video_folder_path, class_folder)
        if not os.path.isdir(class_path) or not class_folder.isdigit():
            continue
        label = int(class_folder)  # Assuming folder names are numeric labels
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            # print(video_path)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (img_width, img_height))
                frames.append(frame)
            cap.release()
            # truncate the video
            # print(len(frames))
            if len(frames) > max_frames:
                frames = frames[:max_frames]
            else:
                for _ in range(max_frames - len(frames)):
                    frames.append(np.zeros((img_height, img_width, 3), dtype=np.uint8))
            if frames:
                X.append(frames)
                y.append(label)
    return np.array(X), np.array(y)

X, y = load_videos(video_folder_path, max_frames)
y = to_categorical(y, num_classes=num_classes)

# # Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Xtrain = ", X_train)

#Define the CNN Model
input_shape = (max_frames, img_height, img_width, 3)  # (frames, height, width, channels)

# # Define the model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# # Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Summary of the model
model.summary()

# Train model 
# Training parameters
batch_size = 8
epochs = 5

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

import matplotlib.pyplot as plt

# Visualize some sample frames
def visualize_samples(X, y, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(10, 2))
        for j in range(10):  # show first 10 frames
            plt.subplot(1, 10, j+1)
            plt.imshow(X[i][j])
            plt.axis('off')
        plt.title(f'Class: {np.argmax(y[i])}')
        plt.show()
print('visualizing')
visualize_samples(X_train, y_train)

# Modify model compilation with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


