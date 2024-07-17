import cv2
import numpy as np
from collections import deque
import os
import datetime

# Define paths to YOLO files
yolo_cfg = "yolov3.cfg"
yolo_weights = "yolov3.weights"
coco_names = "coco.names"

# Ensure files exist
assert os.path.isfile(yolo_cfg), "yolov3.cfg not found"
assert os.path.isfile(yolo_weights), "yolov3.weights not found"
assert os.path.isfile(coco_names), "coco.names not found"

# Load YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names for the YOLO model
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open a video capture (use 0 for webcam or provide the camera IP)
cap = cv2.VideoCapture("rtsp://admin:123456@192.168.1.2:554/stream1")  # Change 0 to your Xiaomi camera's IP stream URL

open("logfile.txt", "w").close()
file = open("logfile.txt", "a")

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    people_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            people_count += 1

    # Detect tailgating: more than one person detected
    if people_count > 1:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file.write(f"{current_time}"+str(4))
        cv2.putText(frame, "Tailgating Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()