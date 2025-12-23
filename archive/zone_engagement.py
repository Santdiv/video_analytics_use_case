import cv2
import numpy as np
import time
import json
import pandas as pd
import math
from collections import defaultdict, deque

# -----------------------------
# CONFIG - update paths & tuning here
# -----------------------------
VIDEO_PATH = "./temp_video.mp4"
YOLO_CFG = "./models/yolov4-tiny.cfg"
YOLO_WEIGHTS = "./models/yolov4-tiny.weights"
COCO_NAMES = "./models/coco.names"

# Processing resolution: set to None to keep original video size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Detection / tracker params
DETECT_EVERY_N_FRAMES = 3     # run full detection every N frames (CPU optimization)
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 80       # for centroid matching (pixels)

# Output
OUTPUT_VIDEO = None  # "annotated_output.avi" to save annotated video

# Pickup heuristic params (tune for your camera)
CENTROID_HISTORY_SIZE = 16
PICKUP_MAX_WINDOW_SECS = 3.0
PICKUP_MIN_APPROACH_DIST = 8.0   # px/frame (negative delta)
PICKUP_PAUSE_FRAMES = 3
PICKUP_RETREAT_DIST = 6.0        # px/frame (positive delta)
PICKUP_MIN_DWELL = 0.5           # seconds inside zone to consider pickup
PICKUP_COOLDOWN_SEC = 4.0        # avoid duplicate logs within cooldown

# Appearance matching for canonical IDs (lightweight)
HIST_COMPARE_METHOD = cv2.HISTCMP_CORREL
HIST_MATCH_THRESHOLD = 0.55  # tune 0.5-0.8 depending on video

# -----------------------------
# Load YOLOv4-tiny (OpenCV DNN)
# -----------------------------
with open(COCO_NAMES, "r") as f:
    CLASS_NAMES = [c.strip() for c in f.readlines()]
PERSON_CLASS_ID = CLASS_NAMES.index("person")

net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
try:
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except Exception:
    out_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# -----------------------------
# Helper: Interactive ROI selection
# -----------------------------
def select_zones(video_path, resize_to=None):
    """
    Shows a frame and lets user draw multiple rectangular ROIs for product/display zones.
    Returns zones list as [((x1,y1),(x2,y2)), ...] in the same coordinates as processing frames.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read video for ROI selection.")
    if resize_to is not None:
        frame = cv2.resize(frame, resize_to)
    print("[INFO] Draw display zone ROIs. Press ENTER/SPACE after each, ESC when done.")
    rois = cv2.selectROIs("Select Zones", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Zones")
    zones = []
    for (x, y, w, h) in rois:
        zones.append(((int(x), int(y)), (int(x + w), int(y + h))))
    return zones, frame.shape[:2]

# -----------------------------
# Helper: YOLO detection (people only)
# -----------------------------
def detect_people(frame, input_size=(416,416), conf_th=CONF_THRESHOLD, nms_th=NMS_THRESHOLD):
    """
    Run YOLOv4-tiny on the frame and return person bounding boxes as (x1,y1,x2,y2).
    """
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(out_layers)
    boxes = []
    confidences = []
    for out in layer_outputs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if class_id == PERSON_CLASS_ID and conf > conf_th:
                cx, cy, w, h = (detection[0:4] * np.array([W, H, W, H])).astype("int")
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(conf)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W - 1, x + w), min(H - 1, y + h)
            results.append((x1, y1, x2, y2))
    return results

# -----------------------------
# Centroid tracker (robust)
# -----------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=50, distance_threshold=DISTANCE_THRESHOLD):
        self.next_id = 0
        self.objects = {}         # id -> (cx,cy)
        self.disappeared = {}     # id -> frames missing
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        if oid in self.objects: del self.objects[oid]
        if oid in self.disappeared: del self.disappeared[oid]

    def update(self, rects):
        """
        rects: list of (x1,y1,x2,y2)
        Returns: dict {id: (cx,cy)}
        """
        if len(rects) == 0:
            # mark disappeared
            remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    remove.append(oid)
            for oid in remove:
                self.deregister(oid)
            return self.objects

        # compute centroids for detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1,y1,x2,y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        # register if no existing objects
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
            return self.objects

        # compute distance matrix between existing and new centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.distance_threshold:
                continue
            oid = object_ids[row]
            self.objects[oid] = tuple(input_centroids[col])
            self.disappeared[oid] = 0
            used_rows.add(row); used_cols.add(col)

        # mark unmatched existing as disappeared
        unmatched_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unmatched_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # register unmatched new centroids
        unmatched_cols = set(range(0, input_centroids.shape[0])).difference(used_cols)
        for col in unmatched_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects

# -----------------------------
# Appearance histogram helpers for canonical ID mapping
# -----------------------------
def get_hsv_hist(frame, bbox, size=(30,32)):
    x1,y1,x2,y2 = bbox
    if x2<=x1 or y2<=y1:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, size, [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(h1, h2):
    if h1 is None or h2 is None: return -1.0
    return cv2.compareHist(h1, h2, HIST_COMPARE_METHOD)

# -----------------------------
# Pickup detection (centroid-history heuristic)
# -----------------------------
def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def detect_pickup(cid, zone_idx, zones, centroid_history, frame_time, pickups_by_cid):
    """
    Decide if canonical customer cid likely performed a pickup in zone_idx using:
      - centroid approach -> pause -> retreat pattern relative to zone center
      - minimal dwell inside zone
      - cooldown to avoid duplicates
    Returns True/False.
    """
    hist = centroid_history.get(cid)
    if not hist or len(hist) < 6:
        return False
    hist_list = list(hist)  # list of (t, (x,y))
    times = [h[0] for h in hist_list]
    coords = [h[1] for h in hist_list]
    # trim to recent window
    t_end = times[-1]
    i = 0
    while i < len(times) and (t_end - times[i] > PICKUP_MAX_WINDOW_SECS):
        i += 1
    coords = coords[i:]
    times = times[i:]
    if len(coords) < 6: return False
    # zone center
    (zx1, zy1), (zx2, zy2) = zones[zone_idx]
    zcenter = ((zx1+zx2)//2, (zy1+zy2)//2)
    dists = [euclid(c, zcenter) for c in coords]
    deltas = [d2-d1 for d1,d2 in zip(dists[:-1], dists[1:])]
    # find approach segment (two consecutive negative deltas)
    approach_end = None
    for idx in range(len(deltas)-2):
        if deltas[idx] < -PICKUP_MIN_APPROACH_DIST and deltas[idx+1] < -PICKUP_MIN_APPROACH_DIST:
            j = idx+2
            while j < len(deltas) and deltas[j] < -PICKUP_MIN_APPROACH_DIST:
                j += 1
            approach_end = j-1
            break
    if approach_end is None: return False
    # pause: small movement near zero
    pause_count = 0; k = approach_end + 1
    while k < len(deltas) and abs(deltas[k]) < 3.0:
        pause_count += 1; k += 1
    if pause_count < PICKUP_PAUSE_FRAMES: return False
    # retreat: positive delta beyond threshold
    retreat_ok = False
    r = k
    while r < len(deltas):
        if deltas[r] > PICKUP_RETREAT_DIST:
            retreat_ok = True; break
        r += 1
    if not retreat_ok: return False
    # ensure minimal dwell inside zone
    inside_count = sum(1 for c in coords if zx1 <= c[0] <= zx2 and zy1 <= c[1] <= zy2)
    if inside_count * frame_time < PICKUP_MIN_DWELL: return False
    # cooldown: recent pickup for same cid-zone?
    last = [t for (z,t) in pickups_by_cid.get(cid, []) if z==zone_idx]
    if last and (time.time() - last[-1] < PICKUP_COOLDOWN_SEC): return False
    return True

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # 1) Select zones (interactive)
    zones, frame_shape = select_zones(VIDEO_PATH, resize_to=(FRAME_WIDTH, FRAME_HEIGHT) if FRAME_WIDTH else None)
    with open("zones.json", "w") as f: json.dump({"zones": zones}, f)
    print(f"[INFO] Selected {len(zones)} zones and saved to zones.json")

    # 2) Open video and prepare output writer if requested
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise RuntimeError("Could not open video")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps
    out_writer = None
    if OUTPUT_VIDEO:
        tw = FRAME_WIDTH if FRAME_WIDTH else orig_w
        th = FRAME_HEIGHT if FRAME_HEIGHT else orig_h
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (tw, th))
        print(f"[INFO] Writing annotated video to {OUTPUT_VIDEO}")

    # 3) Data structures
    tracker = CentroidTracker(max_disappeared=40, distance_threshold=DISTANCE_THRESHOLD)
    centroid_history = defaultdict(lambda: deque(maxlen=CENTROID_HISTORY_SIZE))  # canonical id -> deque of (t,(x,y))
    canonical_hist_db = {}  # canonical_id -> hsv hist
    alias_map = {}          # tracker_id -> canonical_id
    next_canonical_id = 0

    id_zone_accum = defaultdict(lambda: defaultdict(float))  # canonical_id -> {zone: seconds}
    zone_unique_visitors = defaultdict(set)                 # zone -> set(canonical_id)
    pickups_by_cid = defaultdict(list)                      # canonical_id -> list of (zone, timestamp)

    frame_idx = 0
    last_boxes = []

    print("[INFO] Processing video. Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if FRAME_WIDTH and FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # 4) Detection cadence
        if frame_idx % DETECT_EVERY_N_FRAMES == 1:
            boxes = detect_people(frame)
            last_boxes = boxes.copy()
        else:
            boxes = last_boxes

        # 5) Update tracker (returns tracker_id -> centroid)
        objects = tracker.update(boxes)

        # 6) Map tracker ids to bounding boxes (nearest centroid)
        box_centroids = []
        for b in boxes:
            x1,y1,x2,y2 = b
            box_centroids.append(((x1+x2)//2, (y1+y2)//2, b))
        obj_to_bbox = {}
        for tid, centroid in objects.items():
            if box_centroids:
                dists = [math.hypot(centroid[0]-bc[0], centroid[1]-bc[1]) for bc in box_centroids]
                idx = int(np.argmin(dists))
                obj_to_bbox[tid] = box_centroids[idx][2]
            else:
                obj_to_bbox[tid] = None

        now = time.time()

        # 7) For each tracker id, map to canonical id (appearance match or new)
        for tid, cent in list(objects.items()):
            bbox = obj_to_bbox.get(tid)
            # Attempt to map tracker id -> canonical id using alias_map
            cid = alias_map.get(tid, None)
            if cid is None:
                mapped = False
                if bbox is not None:
                    hist = get_hsv_hist(frame, bbox)
                    if hist is not None and canonical_hist_db:
                        # compare against existing canonical histograms
                        best_score = -1.0; best_cid = None
                        for k,h in canonical_hist_db.items():
                            s = compare_hist(hist, h)
                            if s > best_score:
                                best_score = s; best_cid = k
                        if best_score >= HIST_MATCH_THRESHOLD:
                            cid = best_cid
                            alias_map[tid] = cid
                            mapped = True
                if not mapped:
                    cid = next_canonical_id
                    next_canonical_id += 1
                    alias_map[tid] = cid
                    # store histogram if bbox available
                    if bbox is not None:
                        ch = get_hsv_hist(frame, bbox)
                        if ch is not None: canonical_hist_db[cid] = ch

            # update centroid history under canonical id
            centroid_xy = (int(cent[0]), int(cent[1]))
            centroid_history[cid].append((now, centroid_xy))

            # visualize canonical id
            cv2.putText(frame, f"CID {cid}", (centroid_xy[0]-10, centroid_xy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.circle(frame, centroid_xy, 4, (0,255,255), -1)

            # 8) Determine current zone (if any) using bbox centroid (more robust than tracker centroid sometimes)
            current_zone = None
            if bbox is not None:
                bx1,by1,bx2,by2 = bbox
                cx,cy = (int((bx1+bx2)/2), int((by1+by2)/2))
                for zidx, ((zx1,zy1),(zx2,zy2)) in enumerate(zones):
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        current_zone = zidx
                        break

            # 9) Zone visit and dwell accumulation
            prev_zone = None
            # find previous zone from id_zone_accum keys (we store only accum, so maintain an id_current_zone local mapping)
            # We'll store id_current_zone in canonical_hist_db as an attribute? keep a local dict:
            if 'id_current_zone' not in globals():
                globals()['id_current_zone'] = {}
            prev_zone = globals()['id_current_zone'].get(cid, None)
            if current_zone is not None and prev_zone != current_zone:
                globals()['id_current_zone'][cid] = current_zone
                zone_unique_visitors[current_zone].add(cid)
            elif current_zone is not None and prev_zone == current_zone:
                id_zone_accum[cid][current_zone] += frame_time
            elif current_zone is None and prev_zone is not None:
                globals()['id_current_zone'][cid] = None

            # 10) pickup detection when inside zone
            if current_zone is not None:
                picked = detect_pickup(cid, current_zone, zones, centroid_history, frame_time, pickups_by_cid)
                if picked:
                    pickups_by_cid[cid].append((current_zone, now))
                    # annotate pickup
                    cv2.putText(frame, f"PICKUP CID{cid}->Z{current_zone}", (cx, by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # 11) update canonical histogram (running average) to keep appearance DB fresh
            if bbox is not None:
                ch = get_hsv_hist(frame, bbox)
                if ch is not None:
                    if cid not in canonical_hist_db:
                        canonical_hist_db[cid] = ch
                    else:
                        # running average: simple smoother
                        canonical_hist_db[cid] = 0.6*canonical_hist_db[cid] + 0.4*ch

        # 12) Draw zones and stats
        for zidx, ((zx1,zy1),(zx2,zy2)) in enumerate(zones):
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0,200,0), 2)
            cv2.putText(frame, f"Zone {zidx}", (zx1+4, zy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
            cv2.putText(frame, f"Visits: {len(zone_unique_visitors[zidx])}", (zx1+4, zy2+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

        # draw boxes for visualization
        for tid, bbox in obj_to_bbox.items():
            if bbox is None: continue
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,128,0), 2)

        # show / write
        cv2.imshow("Zone Engagement (YOLOv4-tiny)", frame)
        if out_writer: out_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Release resources
    cap.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()

    # -----------------------------
    # Export CSVs
    # -----------------------------
    # customer_zone_log.csv: per-customer per-zone dwell
    rows = []
    for cid, zone_map in id_zone_accum.items():
        for zid, seconds in zone_map.items():
            rows.append({"customer_id": cid, "zone": zid, "dwell_time_sec": round(seconds, 2)})
    df = pd.DataFrame(rows)
    df.to_csv("customer_zone_log.csv", index=False)
    print("[INFO] Saved customer_zone_log.csv")

    # customer_pickups.csv: per-customer pickups
    pickups_rows = []
    for cid, events in pickups_by_cid.items():
        for (z, ts) in events:
            pickups_rows.append({"customer_id": cid, "zone": z, "ts": ts,
                                 "time_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))})
    dfp = pd.DataFrame(pickups_rows)
    dfp.to_csv("customer_pickups.csv", index=False)
    print("[INFO] Saved customer_pickups.csv")

    # zone_metrics.csv: aggregated per-zone metrics
    zones_stats = []
    for zidx in range(len(zones)):
        unique_vis = len(zone_unique_visitors[zidx])
        total_dwell = df[df.zone==zidx].dwell_time_sec.sum() if not df.empty else 0.0
        avg_dwell = df[df.zone==zidx].dwell_time_sec.mean() if not df.empty and (df[df.zone==zidx].shape[0]>0) else 0.0
        items_picked = dfp[dfp.zone==zidx].shape[0] if not dfp.empty else 0
        zones_stats.append({"zone": zidx, "unique_visitors": unique_vis,
                            "total_dwell_sec": round(total_dwell,2),
                            "avg_dwell_sec": round(avg_dwell,2),
                            "items_picked": items_picked})
    dfz = pd.DataFrame(zones_stats)
    dfz.to_csv("zone_metrics.csv", index=False)
    print("[INFO] Saved zone_metrics.csv")
    print(dfz.to_string(index=False))

if __name__ == "__main__":
    main()
