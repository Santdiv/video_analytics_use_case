import streamlit as st
import cv2
import numpy as np
import tempfile
import math
from itertools import combinations

st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["Upload Video", "Proximity Analysis"])

with tab1:
    uploaded = st.file_uploader("Upload CCTV video", type=["mp4","avi","mov"])
    model_choice = st.selectbox("Select YOLO Model", ["YOLOv4", "YOLOv4-tiny"])
    pixels_per_meter = st.number_input("Pixels per meter (0 = pixel units)", min_value=0.0, value=0.0)
    people_threshold = st.number_input("Person-to-person threshold", value=150.0)
    zone_threshold = st.number_input("Person-to-zone threshold", value=120.0)
    zone_input = st.text_input("Zone (x1,y1,x2,y2)", "100,200,400,600")

with tab2:
    if uploaded is None:
        st.warning("Upload a video in the first tab")
        st.stop()

    ZONE = tuple(map(int, zone_input.split(",")))

    if model_choice == "YOLOv4":
        weights_path = "models/yolov4.weights"
        cfg_path = "models/yolov4.cfg"
        input_size = (416, 416)
    else:
        weights_path = "models/yolov4-tiny.weights"
        cfg_path = "models/yolov4-tiny.cfg"
        input_size = (320, 320)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    video_path = tfile.name

    net = cv2.dnn.readNet(weights_path, cfg_path)
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