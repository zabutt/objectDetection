# app.py

import streamlit as st
import cv2
import numpy as np
import urllib.request
import os

# Load YOLOv8 model (you can replace this with your own trained model)
# Download the weights and config files from the official YOLOv8 repository
weights_url = "https://github.com/WongKinYiu/yolov8/releases/download/1.0/yolov8.weights"
cfg_url = "https://github.com/WongKinYiu/yolov8/releases/download/1.0/yolov8.cfg"
names_url = "https://github.com/WongKinYiu/yolov8/releases/download/1.0/coco.names"

@st.cache
def load_model():
    # Download model files
    urllib.request.urlretrieve(weights_url, "yolov8.weights")
    urllib.request.urlretrieve(cfg_url, "yolov8.cfg")
    urllib.request.urlretrieve(names_url, "coco.names")

    # Load YOLOv8 model
    net = cv2.dnn.readNet("yolov8.weights", "yolov8.cfg")
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    return net, classes

def detect_objects(image, net, classes, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    objects = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                label = classes[class_id]
                objects.append(label)
    return objects

def main():
    st.title("Object Detection App")
    st.sidebar.header("Settings")
    confidence_level = st.sidebar.slider("Confidence Level", 0.0, 1.0, 0.5, 0.01)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = np.array(Image.open(uploaded_image))
        net, classes = load_model()
        detected_objects = detect_objects(image, net, classes, confidence_level)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detected Objects:")
        for obj in detected_objects:
            st.write(f"- {obj}")

if __name__ == "__main__":
    main()
