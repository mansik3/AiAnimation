import tensorflow as tf
import numpy as np
import cv2
import torch

from tensorflow.keras.preprocessing.image import img_to_array

video_paths = ["move_vids/one_pers_dance_1.mp4", "move_vids/one_pers_dance_2.mp4"]

weights_path = "yolo_files/yolov7.onnx"
labels_path = "yolo_files/coco.names"

# Load YOLO model
net = cv2.dnn.readNetFromONNX(weights_path)
labels = open(labels_path).read().strip().split("\n")

video_captures = []
frames = []
masks = [] 

for video_path in video_paths:
  video_captures.append(cv2.VideoCapture(video_path))

print("success so far")
