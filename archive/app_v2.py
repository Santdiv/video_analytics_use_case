import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(layout="wide")
st.subheader("Video Analytics")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox('Choose YOLO Model', ['YOLOv4', 'YOLOv4-tiny'])
    uploaded_file = st.file_uploader('Upload a video file (.mp4, .avi, .mov)', type=['mp4', 'avi', 'mov'])
    analytics_option = st.radio(
        "Select Analytics Option:",
        ["Heatmap", "Dwell analysis", "Person detection"]
    )

if model_choice == 'YOLOv4':
    weights_path = 'models/yolov4.weights'
    config_path = 'models/yolov4.cfg'
    input_size = (416, 416)
else:
    weights_path = 'models/yolov4-tiny.weights'
    config_path = 'models/yolov4-tiny.cfg'
    input_size = (320, 320)


@st.cache_resource
def load_network(weights, config):
    net = cv2.dnn.readNet(weights, config)
    layer_names = net.getLayerNames()
    out_layer_idxs = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in out_layer_idxs]
    return net, output_layers


# Define dwell zone shifted to the right side of the frame (adjust x coords dynamically later)
ZONE_OFFSET_FROM_RIGHT = 100  # pixels from right edge
ZONE_WIDTH = 300
ZONE_HEIGHT = 400

dwell_times = dict()
tracker = DeepSort(max_age=30)


def in_zone(bbox, zone):
    x1, y1, x2, y2 = zone
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def process_frame(frame, detections, zone):
    global dwell_times
    
    detections_list = []
    for det in detections:
        bbox = det[:4].tolist() if isinstance(det, np.ndarray) else list(det[:4])
        conf = float(det[4])
        detections_list.append([bbox, conf])

    results = tracker.update_tracks(detections_list, frame=frame)
    
    cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 2)
    
    for track in results:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        if in_zone((l, t, r, b), zone):
            if track_id not in dwell_times:
                dwell_times[track_id] = 0
            dwell_times[track_id] += 1
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame


if analytics_option == "Heatmap":
    st.info("This page is under maintenance")

elif analytics_option == "Dwell analysis":
    if uploaded_file is None:
        st.warning("Please upload a video file from the sidebar to start dwell time analysis.")
    else:
        temp_path = 'temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        net, output_layers = load_network(weights_path, config_path)
        cap = cv2.VideoCapture(temp_path)
        frame_placeholder = st.empty()
        scale_factor = 0.7
        
        # Get video width and height to dynamically set zone on right side
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define zone shifted to right side within frame height limits
        zone_left = video_width - ZONE_OFFSET_FROM_RIGHT - ZONE_WIDTH
        zone_top = max(0, (video_height - ZONE_HEIGHT) // 2)  # center vertically
        zone_right = zone_left + ZONE_WIDTH
        zone_bottom = zone_top + ZONE_HEIGHT
        ZONE = (zone_left, zone_top, zone_right, zone_bottom)
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback to 30 if FPS unavailable
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            detected_boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > 0.5:
                        center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                        x1 = int(center_x - w / 2)
                        y1 = int(center_y - h / 2)
                        x2 = x1 + w
                        y2 = y1 + h
                        detected_boxes.append([x1, y1, x2, y2, confidence])
            
            if detected_boxes:
                dets_np = np.array(detected_boxes, dtype=float)
                if dets_np.ndim == 1 and dets_np.size == 5:
                    dets_np = dets_np.reshape((1, 5))
            else:
                dets_np = np.empty((0, 5))
            
            processed_frame = process_frame(frame, dets_np, ZONE)
            
            display_frame = cv2.resize(processed_frame, None, fx=scale_factor, fy=scale_factor)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels='RGB')

        cap.release()
        
        # After video processing complete - save dwell times and show download button
        df_dwell = pd.DataFrame(list(dwell_times.items()), columns=['Track ID', 'Frames_in_Zone'])
        df_dwell['Dwell_Time_Secs'] = df_dwell['Frames_in_Zone'] / fps
        
        st.write("Dwell Times (per Track):")
        st.dataframe(df_dwell)
        
        csv_bytes = df_dwell.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download dwell time CSV",
            data=csv_bytes,
            file_name='dwell_time_output.csv',
            mime='text/csv'
        )
    
else:
    st.subheader("Person detection using YOLO")
    if uploaded_file is None:
        st.warning("Please upload a video file from the sidebar to start detection.")
    else:
        temp_path = 'temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        net, output_layers = load_network(weights_path, config_path)
        cap = cv2.VideoCapture(temp_path)
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
                    if class_id == 0 and confidence > 0.5:
                        center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels='RGB')
        cap.release()
