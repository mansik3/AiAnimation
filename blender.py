## This is a non-function basic outline file. Still need to work on this 

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import socket
import bpy

model = load_model('gest_recog.keras')

IMG_SIZE = 150

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(resized).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    img_array = img_array / 255.0

    # Predict gesture
    prediction = model.predict(img_array)
    gesture = np.argmax(prediction)

    # Send gesture data to Blender
    send_gesture_to_blender(gesture)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def send_gesture_to_blender(gesture):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 12345)
    message = str(gesture).encode()
    sock.sendto(message, server_address)

# Create a socket to receive data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('localhost', 1118))

# Function to receive data and update character
def update_character():
    sock.setblocking(0)
    try:
        data, _ = sock.recvfrom(1024)
        gesture = int(data.decode())
        move_character_based_on_gesture(gesture)
    except BlockingIOError:
        pass

# Function to move character based on gesture
def move_character_based_on_gesture(gesture):
    # Example: Update the location of an object named 'Character'
    obj = bpy.data.objects['Character']
    if gesture == 0:
        obj.location.x += 1.0
    elif gesture == 1:
        obj.location.x -= 1.0
    # Add more gestures and their corresponding movements

# Add a handler to call update_character periodically
bpy.app.handlers.frame_change_pre.append(lambda scene: update_character())

def move_character_based_on_gesture(gesture):
    # Example: Update the location of an object named 'Character'
    obj = bpy.data.objects['Character']
    if gesture == 0:
        obj.location.x += 1.0
    elif gesture == 1:
        obj.location.x -= 1.0
    # Add more gestures and their corresponding movements

