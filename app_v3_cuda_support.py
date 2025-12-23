import os
import re
import uuid
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import math
from itertools import combinations
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import io
from streamlit_image_coordinates import streamlit_image_coordinates
from zone_processor import process_video


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(layout="wide")
st.title("🛒 Retail Video Analytics")

# st.set_page_config(layout="wide")
# st.subheader("Video Analytics")

# ---------------------------
# Sidebar controls / settings
# ---------------------------
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox('Choose YOLO Model', ['YOLOv4', 'YOLOv4-tiny'])
    uploaded_file = st.file_uploader('Upload a video file (.mp4, .avi, .mov)', type=['mp4', 'avi', 'mov'])
    analytics_option = st.radio(
        "Select Analytics Option:",
        ["Heatmap", "Dwell analysis", "Person detection", "Proximity Analysis", "Zone Engagement"]
    )

    st.header("Model params")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    nms_thresh = st.slider("NMS threshold", 0.1, 1.0, 0.45, 0.05)
    use_cuda_requested = st.checkbox("Use CUDA (if available)", value=False)

    st.header("Zone params (right-side)")
    ZONE_OFFSET_FROM_RIGHT = st.number_input("Offset from right (px)", min_value=0, max_value=2000, value=100, step=10)
    ZONE_WIDTH = st.number_input("Zone width (px)", min_value=50, max_value=2000, value=300, step=10)
    ZONE_HEIGHT = st.number_input("Zone height (px)", min_value=50, max_value=2000, value=400, step=10)

# ---------------------------
# Model selection
# ---------------------------
if model_choice == 'YOLOv4':
    weights_path = './models/yolov4.weights'
    config_path = './models/yolov4.cfg'
    input_size = (416, 416)
else:
    weights_path = './models/yolov4-tiny.weights'
    config_path = './models/yolov4-tiny.cfg'
    input_size = (320, 320)

# ---------------------------
# CUDA availability check
# ---------------------------
def dnn_cuda_available() -> bool:
    try:
        info = cv2.getBuildInformation()
        built_with_cuda = bool(re.search(r'(CUDA|cuDNN)', info, re.I))
    except Exception:
        built_with_cuda = False

    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        has_device = (device_count is not None) and (device_count > 0)
    except Exception:
        has_device = False

    return built_with_cuda and has_device

# ---------------------------
# Load network with cache
# ---------------------------
@st.cache_resource
def load_network(weights, config, use_cuda=False, prefer_fp16=True):
    net = cv2.dnn.readNet(weights, config)

    want_cuda = bool(use_cuda) and dnn_cuda_available()
    backend_msg = "CPU"

    if want_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            if prefer_fp16:
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            else:
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            backend_msg = "CUDA"
        except Exception:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            backend_msg = "CPU (fallback)"
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend_msg = "CPU"

    layer_names = net.getLayerNames()
    out_layer_idxs = net.getUnconnectedOutLayers().flatten()
    output_layers = [layer_names[i - 1] for i in out_layer_idxs]
    return net, output_layers, backend_msg

# ---------------------------
# Dwell times and tracker
# ---------------------------
dwell_times_ms = dict()
tracker = DeepSort(max_age=60)

# ---------------------------
# Utility functions
# ---------------------------
def to_excel(df1, df2):
    """
    Writes two pandas DataFrames to a single Excel file with two sheets.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        # writer.close() is automatically called when exiting the 'with' block
    
    processed_data = output.getvalue()
    return processed_data

def in_zone(bbox, zone):
    x1, y1, x2, y2 = zone
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    return x1 <= cx <= x2 and y1 <= cy <= y2

def process_frame(frame, detections_xyxy_conf, zone, dt_ms):
    global dwell_times_ms, tracker

    detections_list = []
    for det in detections_xyxy_conf:
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

        if in_zone((l, t, r, b), zone) and dt_ms > 0:
            dwell_times_ms[track_id] = dwell_times_ms.get(track_id, 0.0) + dt_ms

        in_zone_now = in_zone((l, t, r, b), zone)
        color = (0, 200, 0) if in_zone_now else (255, 0, 0)
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
        dwell_sec = dwell_times_ms.get(track_id, 0.0) / 1000.0
        cv2.putText(frame, f'ID {track_id} | {dwell_sec:.1f}s',
                    (int(l), max(15, int(t) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

def yolo_person_detections(frame, net, output_layers, input_size, conf_thresh, nms_thresh):
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes_xywh = []
    confidences = []
    class_ids = []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < conf_thresh:
                continue
            if cid != 0:  # Person class
                continue
            cx, cy, w, h = det[0:4]
            x = int((cx - w / 2) * W)
            y = int((cy - h / 2) * H)
            w_px = int(w * W)
            h_px = int(h * H)
            boxes_xywh.append([x, y, w_px, h_px])
            confidences.append(conf)
            class_ids.append(cid)

    detected_xyxy_conf = []
    if len(boxes_xywh) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_thresh, nms_thresh)
        if len(indices) > 0:
            max_box_width = 0.7 * W
            max_box_height = 0.9 * H

            for i in indices.flatten():
                x, y, w_px, h_px = boxes_xywh[i]
                if w_px > max_box_width or h_px > max_box_height:
                    continue
                x1, y1 = x, y
                x2, y2 = x + w_px, y + h_px
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W - 1))
                y2 = max(0, min(y2, H - 1))
                detected_xyxy_conf.append([x1, y1, x2, y2, float(confidences[i])])

    return detected_xyxy_conf

# ---------------------------
# Handle uploaded video saving safely using session state
# ---------------------------
def save_uploaded_video(uploaded_file):
    if 'temp_video_path' not in st.session_state:
        st.session_state.temp_video_path = None

    if uploaded_file is not None:
        # Delete previous temp file if exists
        if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
            try:
                os.remove(st.session_state.temp_video_path)
            except Exception:
                pass
        unique_filename = f"temp_video_{uuid.uuid4()}.mp4"
        with open(unique_filename, 'wb') as f:
            f.write(uploaded_file.read())
        st.session_state.temp_video_path = unique_filename
        return unique_filename
    return None

# ---------------------------
# Main app flow
# ---------------------------
if uploaded_file is not None:
    temp_path = save_uploaded_video(uploaded_file)
else:
    temp_path = None

# ---------------- HEATMAP OPTION (Hotspot Analysis) ----------------
if analytics_option == "Heatmap":
    st.subheader("Hotspot Heatmap (YOLOv4-Tiny)")
    if temp_path is None:
        st.warning("Please upload a video file from the sidebar to start heatmap generation.")
    else:
        # temp_path = 'temp_video.mp4'
        # with open(temp_path, 'wb') as f:
        #     f.write(uploaded_file.read())

        net, output_layers, backend_msg = load_network(weights_path, config_path)
        st.sidebar.write(f"Backend/Target in use: {backend_msg}")

        cap = cv2.VideoCapture(temp_path)

        st.info("Processing video for hotspot heatmap... This may take some time.")

        centroids = []
        frame_count = 0
        first_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if first_frame is None:
                first_frame = frame.copy()
            h, w = frame.shape[:2]

            # YOLO detections
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > 0.5:  # class_id=0 → person
                        center_x, center_y, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype(int)
                        cx, cy = int(center_x), int(center_y)
                        centroids.append((cx, cy))

        cap.release()

        if len(centroids) == 0:
            st.error("No persons detected in the video.")
        else:
            st.success(f"✅ Processed {frame_count} frames. Generating hotspot heatmap...")

            # Create a blank density map
            density_map = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.float32)

            # Add 1 at each centroid location
            for (x, y) in centroids:
                if 0 <= y < density_map.shape[0] and 0 <= x < density_map.shape[1]:
                    density_map[y, x] += 1

            # Apply Gaussian smoothing → makes hotspots continuous
            density_map = cv2.GaussianBlur(density_map, (0, 0), sigmaX=50, sigmaY=50)

            # Normalize and colorize
            heatmap_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

            # Overlay heatmap on original frame
            overlay = cv2.addWeighted(first_frame, 0.6, heatmap_color, 0.4, 0)

            # Show in Streamlit
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                     caption="Hotspot Heatmap (Most Active Areas)", use_column_width=True)

            # Save
            cv2.imwrite("hotspot_heatmap.png", overlay)

            # Download
            success, buffer = cv2.imencode(".png", overlay)
            if success:
                st.download_button(
                    label="Download Heatmap Image",
                    data=buffer.tobytes(),
                    file_name="hotspot_heatmap.png",
                    mime="image/png"
                )

elif analytics_option == "Dwell analysis":
    if temp_path is None:
        st.warning("Please upload a video file from the sidebar to start dwell time analysis.")
    else:
        net, output_layers, backend_msg = load_network(
            weights_path, config_path, use_cuda=use_cuda_requested, prefer_fp16=True
        )
        st.sidebar.write(f"Backend/Target in use: {backend_msg}")

        cap = cv2.VideoCapture(temp_path)
        frame_placeholder = st.empty()
        scale_factor = 0.7

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        zone_left = max(0, video_width - ZONE_OFFSET_FROM_RIGHT - ZONE_WIDTH)
        zone_top = max(0, (video_height - ZONE_HEIGHT) // 2)
        zone_right = min(video_width - 1, zone_left + ZONE_WIDTH)
        zone_bottom = min(video_height - 1, zone_top + ZONE_HEIGHT)
        ZONE = (zone_left, zone_top, zone_right, zone_bottom)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        last_pos_ms = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if (pos_ms is None) or (pos_ms <= 0):
                if last_pos_ms is None:
                    pos_ms = 0.0
                else:
                    pos_ms = last_pos_ms + (1000.0 / fps)
            dt_ms = 0.0
            if last_pos_ms is not None:
                dt_ms = max(0.0, pos_ms - last_pos_ms)
                if dt_ms == 0.0:
                    dt_ms = (1000.0 / fps)
            last_pos_ms = pos_ms

            detected_boxes = yolo_person_detections(
                frame, net, output_layers, input_size, conf_thresh, nms_thresh
            )

            processed_frame = process_frame(frame, detected_boxes, ZONE, dt_ms)

            display_frame = cv2.resize(processed_frame, None, fx=scale_factor, fy=scale_factor)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels='RGB')

        cap.release()

        rows = []
        for tid, ms in dwell_times_ms.items():
            rows.append((tid, ms / 1000.0))
        df_dwell = pd.DataFrame(rows, columns=['Track ID', 'Dwell_Time_Secs'])

        st.write("Dwell Times (per Track):")
        st.dataframe(df_dwell)

        csv_bytes = df_dwell.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download dwell time CSV",
            data=csv_bytes,
            file_name='dwell_time_output.csv',
            mime='text/csv'
        )

elif analytics_option=="Person detection":
    st.subheader("Person detection using YOLO")
    if temp_path is None:
        st.warning("Please upload a video file from the sidebar to start detection.")
    else:
        net, output_layers, backend_msg = load_network(
            weights_path, config_path, use_cuda=use_cuda_requested, prefer_fp16=True
        )
        st.sidebar.write(f"Backend/Target in use: {backend_msg}")

        cap = cv2.VideoCapture(temp_path)
        frame_placeholder = st.empty()
        scale_factor = 0.7

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = yolo_person_detections(
                frame, net, output_layers, input_size, conf_thresh, nms_thresh
            )

            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels='RGB')

        cap.release()

elif analytics_option=="Proximity Analysis":
    pixels_per_meter = st.number_input("Pixels per meter (0 = pixel units)", min_value=0.0, value=0.0)
    people_threshold = st.number_input("Person-to-person threshold", value=150.0)
    zone_threshold = st.number_input("Person-to-zone threshold", value=120.0)
    zone_input = st.text_input("Zone (x1,y1,x2,y2)", "100,200,400,600")

    st.subheader("Person detection using YOLO")
    if temp_path is None:
        st.warning("Please upload a video file from the sidebar to start detection.")
    else:
        net, output_layers, backend_msg = load_network(
            weights_path, config_path, use_cuda=use_cuda_requested, prefer_fp16=True
        )
        st.sidebar.write(f"Backend/Target in use: {backend_msg}")

        cap = cv2.VideoCapture(temp_path)

        ZONE = tuple(map(int, zone_input.split(",")))

        if model_choice == "YOLOv4":
            weights_path = "models/yolov4.weights"
            config_path = "models/yolov4.cfg"
            input_size = (416, 416)
        else:
            weights_path = "models/yolov4-tiny.weights"
            config_path = "models/yolov4-tiny.cfg"
            input_size = (320, 320)

        # tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        # tfile.write(uploaded.read())
        video_path = temp_path

        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        def center(box):
            x1, y1, x2, y2 = box
            return int((x1 + x2) / 2), int((y1 + y2) / 2)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            persons = []
            for out in outs:
                for det in out:
                    scores = det[5:]
                    cls = np.argmax(scores)
                    conf = scores[cls]
                    if cls == 0 and conf > 0.5:
                        cx, cy, bw, bh = det[0]*w, det[1]*h, det[2]*w, det[3]*h
                        x1, y1 = int(cx - bw/2), int(cy - bh/2)
                        x2, y2 = x1 + int(bw), y1 + int(bh)
                        persons.append((x1, y1, x2, y2))

            annotated = frame.copy()

            # person to person
            for p1, p2 in combinations(persons, 2):
                c1, c2 = center(p1), center(p2)
                dpx = math.dist(c1, c2)
                dist = dpx / pixels_per_meter if pixels_per_meter > 0 else dpx

                if dist <= people_threshold:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.line(annotated, c1, c2, color, 2)
                mid = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
                cv2.putText(annotated, f"{dist:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # person to zone
            zx1, zy1, zx2, zy2 = ZONE
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)
            for p in persons:
                cx, cy = center(p)
                dx = max(zx1 - cx, 0, cx - zx2)
                dy = max(zy1 - cy, 0, cy - zy2)
                dpx = math.hypot(dx, dy)
                dist = dpx / pixels_per_meter if pixels_per_meter > 0 else dpx

                if dist <= zone_threshold:
                    cv2.rectangle(annotated, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
                else:
                    cv2.rectangle(annotated, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)

            show = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(show)

        cap.release()

elif analytics_option=="Zone Engagement":
    if temp_path is None:
        st.warning("Please upload a video file from the sidebar to start detection.")
    else:
        net, output_layers, backend_msg = load_network(
            weights_path, config_path, use_cuda=use_cuda_requested, prefer_fp16=True
        )
        st.sidebar.write(f"Backend/Target in use: {backend_msg}")

        video_path = temp_path
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Unable to read video")
            st.stop()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        st.subheader("🎯 Draw Zones (Click two corners per zone)")
        st.info("Click TOP-LEFT → then BOTTOM-RIGHT. Repeat for multiple zones.")

        # -----------------------------
        # Initialize session state
        # -----------------------------
        if "clicks" not in st.session_state:
            st.session_state.clicks = []
        if "zones" not in st.session_state:
            st.session_state.zones = []

        # -----------------------------
        # Capture click coordinates
        # -----------------------------
        coords = streamlit_image_coordinates(frame_rgb, key="zone_image")

        if coords:
            st.session_state.clicks.append((coords["x"], coords["y"]))

            # When 2 clicks exist → create zone
            if len(st.session_state.clicks) == 2:
                (x1, y1), (x2, y2) = st.session_state.clicks
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                st.session_state.zones.append(((x1, y1), (x2, y2)))
                st.session_state.clicks = []

                st.success(f"Zone added: ({x1},{y1}) → ({x2},{y2})")

        # -----------------------------
        # Draw zones for preview
        # -----------------------------
        preview = frame_rgb.copy()
        for idx, ((x1,y1),(x2,y2)) in enumerate(st.session_state.zones):
            cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(preview, f"Zone {idx}", (x1+5,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        st.image(preview, caption="Zone Preview", use_column_width=True)

        # -----------------------------
        # Controls
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧹 Clear Zones"):
                st.session_state.zones = []
                st.session_state.clicks = []
                st.warning("Zones cleared")

        with col2:
            if st.button("💾 Save Zones"):
                with open("zones.json", "w") as f:
                    json.dump({"zones": st.session_state.zones}, f)
                st.success("Zones saved")

        # -----------------------------
        # Run analytics
        # -----------------------------
        if st.button("▶️ Run Zone Engagement Analytics"):
            if not st.session_state.zones:
                st.warning("Please draw and save zones first")
            else:
                with st.spinner("Processing video..."):
                    df_customer_zone, df_zone_metrics = process_video(
                        video_path,
                        {
                            "cfg": "./models/yolov4-tiny.cfg",
                            "weights": "./models/yolov4-tiny.weights",
                            "names": "./models/coco.names",
                            "zones": "zones.json"
                        }
                    )

                # -----------------------------
                # Results
                # -----------------------------
                st.subheader("📊 Per-Customer Per-Zone Dwell Time")
                st.dataframe(df_customer_zone, use_container_width=True)

                st.subheader("📊 Zone Metrics")
                st.dataframe(df_zone_metrics, use_container_width=True)

                # -----------------------------
                # Downloads
                # -----------------------------
                st.download_button(
                    label="📥 Download Excel Workbook",
                    data=to_excel(df_customer_zone, df_zone_metrics),
                    file_name='customer_zone_metrics.xlsx',
                    mime='application/vnd.ms-excel'
                )