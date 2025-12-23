import streamlit as st
import cv2
import numpy as np

st.title('YOLO Person Detection Showcase')

model_choice = st.selectbox('Choose YOLO Model', ['YOLOv4', 'YOLOv4-tiny'])

if model_choice == 'YOLOv4':
    weights_path = 'yolov4.weights'
    config_path = 'yolov4.cfg'
    input_size = (416, 416)
else:
    weights_path = 'yolov4-tiny.weights'
    config_path = 'yolov4-tiny.cfg'
    input_size = (320, 320)

# FIX: Use allow_output_mutation=True to avoid unhashable Net error
@st.cache(allow_output_mutation=True)
def load_network(weights, config):
    net = cv2.dnn.readNet(weights, config)
    layer_names = net.getLayerNames()
    out_layer_idxs = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in out_layer_idxs]
    return net, output_layers

net, output_layers = load_network(weights_path, config_path)

uploaded_file = st.file_uploader('Upload a video file (.mp4, .avi, .mov)', type=['mp4', 'avi', 'mov'])
if uploaded_file is not None:
    with open('temp_video.mp4', 'wb') as f:
        f.write(uploaded_file.read())
    video_path = 'temp_video.mp4'
else:
    st.warning("Please upload a video file to start detection.")
    st.stop()

cap = cv2.VideoCapture(video_path)
frame_placeholder = st.empty()
scale_factor = 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # person class in COCO
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(display_frame, channels='RGB')

cap.release()
