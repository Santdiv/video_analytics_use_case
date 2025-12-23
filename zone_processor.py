"""
zone_processor.py

This module contains all heavy video analytics logic:
- YOLOv4-tiny person detection
- Centroid tracking
- Zone visit counting
- Dwell time calculation
- Pickup detection
- CSV-ready outputs

This file is UI-agnostic and can be reused in CLI, Streamlit, Flask, etc.
"""

# =============================
# IMPORTS
# =============================
import cv2
import numpy as np
import time
import json
import math
import pandas as pd
from collections import defaultdict, deque

# =============================
# LOAD ZONES
# =============================
def load_zones(zone_file="zones.json"):
    """
    Loads pre-defined zones from JSON.
    Zones must be created once using cv2.selectROIs.
    """
    with open(zone_file, "r") as f:
        data = json.load(f)
    return data["zones"]

# =============================
# YOLO INITIALIZATION
# =============================
def load_yolo(cfg, weights, names):
    """
    Loads YOLOv4-tiny network into OpenCV DNN (CPU).
    """
    with open(names, "r") as f:
        classes = [c.strip() for c in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, out_layers, classes

# =============================
# PERSON DETECTION
# =============================
def detect_people(frame, net, out_layers, person_id, conf=0.45, nms=0.4):
    """
    Runs YOLOv4-tiny on a frame and returns person bounding boxes.
    """
    H, W = frame.shape[:2]

    # Convert image to YOLO blob
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416,416), swapRB=True, crop=False
    )
    net.setInput(blob)

    outputs = net.forward(out_layers)

    boxes, scores = [], []
    for out in outputs:
        for det in out:
            class_scores = det[5:]
            cid = np.argmax(class_scores)
            score = class_scores[cid]

            # Filter only persons
            if cid == person_id and score > conf:
                cx, cy, w, h = (det[:4] * np.array([W, H, W, H])).astype(int)
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                boxes.append([x1, y1, w, h])
                scores.append(float(score))

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf, nms)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            results.append((x, y, x+w, y+h))

    return results

# =============================
# CENTROID TRACKER
# =============================
class CentroidTracker:
    """
    Tracks object centroids across frames using distance matching.
    """
    def __init__(self, max_lost=30, dist_thresh=80):
        self.next_id = 0
        self.objects = {}
        self.lost = {}
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.lost[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.lost[oid]

    def update(self, boxes):
        """
        Updates tracker with new bounding boxes.
        """
        if len(boxes) == 0:
            for oid in list(self.lost.keys()):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array(
            [((x1+x2)//2, (y1+y2)//2) for x1,y1,x2,y2 in boxes]
        )

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(
            np.array(object_centroids)[:,None] - input_centroids[None,:], axis=2
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r,c] > self.dist_thresh:
                continue

            oid = object_ids[r]
            self.objects[oid] = tuple(input_centroids[c])
            self.lost[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(D.shape[0])) - used_rows:
            oid = object_ids[r]
            self.lost[oid] += 1
            if self.lost[oid] > self.max_lost:
                self.deregister(oid)

        for c in set(range(input_centroids.shape[0])) - used_cols:
            self.register(tuple(input_centroids[c]))

        return self.objects

# =============================
# MAIN VIDEO PROCESSOR
# =============================
def process_video(video_path, model_cfg):
    """
    Core analytics pipeline:
    - Reads video
    - Detects persons
    - Tracks them
    - Computes zone visits & dwell
    Returns DataFrames.
    """

    # Load zones and model
    zones = load_zones(model_cfg["zones"])
    net, out_layers, classes = load_yolo(
        model_cfg["cfg"], model_cfg["weights"], model_cfg["names"]
    )
    person_id = classes.index("person")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps

    tracker = CentroidTracker()

    # Analytics containers
    zone_visitors = defaultdict(set)
    dwell_time = defaultdict(lambda: defaultdict(float))
    centroid_history = defaultdict(lambda: deque(maxlen=10))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_people(frame, net, out_layers, person_id)
        objects = tracker.update(boxes)

        for oid, (cx, cy) in objects.items():
            centroid_history[oid].append((cx, cy))

            for zid, ((x1,y1),(x2,y2)) in enumerate(zones):
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    zone_visitors[zid].add(oid)
                    dwell_time[oid][zid] += frame_time

    cap.release()

    # Convert analytics to DataFrames
    rows = []
    for cid, zones_data in dwell_time.items():
        for zid, sec in zones_data.items():
            rows.append({
                "customer_id": cid,
                "zone": zid,
                "dwell_time_sec": round(sec,2)
            })

    df_customer_zone = pd.DataFrame(rows)

    zone_rows = []
    for zid, visitors in zone_visitors.items():
        zone_rows.append({
            "zone": zid,
            "unique_visitors": len(visitors),
            "total_dwell_sec": df_customer_zone[df_customer_zone.zone==zid].dwell_time_sec.sum()
        })

    df_zone_metrics = pd.DataFrame(zone_rows)

    return df_customer_zone, df_zone_metrics