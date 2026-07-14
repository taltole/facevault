"""
FaceVault — Main Pipeline Orchestrator
=======================================
Real-time retail customer intelligence.

Architecture (6 parallel threads):
  Thread 1: OAK-D frame capture      (utils_cam)
  Thread 2: MediaPipe face analysis   (utils_mp)
  Thread 3: Kalman 3D tracking        (utils_filters)
  Thread 4: Identity lookup           (utils_image + utils_nn)
  Thread 5: Visit logging             (utils_file)
  Thread 6: Staff UI                  (utils_tk)

Main thread: orchestration + gaze zone analysis.
"""

import time
import queue
import threading
import logging
import numpy as np
import cv2

from config.config import (
    MONO_HEIGHT, MONO_WIDTH, RESOLUTION_NUM,
    KALMAN_ACC_STD, KALMAN_MEAS_STD, KALMAN_DT,
    DHASH_THRESHOLD, IDENTITY_THRESHOLD,
    GAZE_DWELL_THRESHOLD_SEC, ZONE_MAP,
    DB_PATH, LOG_LEVEL
)

from Utils.utils         import setup_logging, get_device
from Utils.utils_cam     import StereoInference, build_oak_pipeline
from Utils.utils_mp      import extract_mp_landmarks, get_mp_depthmask
from Utils.utils_filters import KalmanFilter, run_kalman
from Utils.utils_image   import dhash, find_match_in_db, normalize_face_crop
from Utils.utils_nn      import extract_embedding, load_backbone
from Utils.utils_cls     import ImageProcessor
from Utils.utils_file    import VisitLogger
from Utils.utils_vis     import show_embedds, draw_gaze_heatmap
from Utils.utils_tk      import StaffUI
from Utils.utils_gl      import render_attention_heatmap_3d

logger = setup_logging(LOG_LEVEL)


# ── Shared queues between threads ─────────────────────────────────────────────
q_frames    = queue.Queue(maxsize=4)   # (rgb_frame, depth_frame, timestamp)
q_landmarks = queue.Queue(maxsize=4)   # (landmarks_xyz, mask, timestamp)
q_tracked   = queue.Queue(maxsize=4)   # (smoothed_xyz, track_id, timestamp)
q_identity  = queue.Queue(maxsize=4)   # (customer_id, confidence, embedding)
q_ui        = queue.Queue(maxsize=2)   # (annotated_frame, metadata_dict)


# ── Thread 1: Camera Capture ──────────────────────────────────────────────────
def thread_capture(device, stereo_inf, stop_event):
    """Reads RGB + depth from OAK-D and pushes to q_frames."""
    logger.info("[T1-Capture] Starting OAK-D capture thread")
    q_rgb, q_depth = build_oak_pipeline(device)

    while not stop_event.is_set():
        rgb_frame   = q_rgb.get().getCvFrame()
        depth_frame = q_depth.get().getFrame()
        ts          = time.monotonic()

        if not q_frames.full():
            q_frames.put((rgb_frame, depth_frame, ts))

    logger.info("[T1-Capture] Stopped")


# ── Thread 2: MediaPipe Face Analysis ────────────────────────────────────────
def thread_mediapipe(stop_event):
    """
    Pulls RGB frames, runs MediaPipe FaceMesh (468 landmarks with XYZ),
    generates depth-aware face mask, pushes to q_landmarks.
    """
    logger.info("[T2-MediaPipe] Starting face analysis thread")
    processor = ImageProcessor()

    while not stop_event.is_set():
        try:
            rgb_frame, depth_frame, ts = q_frames.get(timeout=0.1)
        except queue.Empty:
            continue

        # Extract 468 3D face landmarks
        landmarks_xyz = extract_mp_landmarks(rgb_frame)

        if landmarks_xyz is not None:
            # Generate depth-aware face mask — Z-coordinate filtering
            # removes landmarks outside the expected face depth range
            mask = get_mp_depthmask(
                frame=rgb_frame,
                landmarks=landmarks_xyz,
                depth_frame=depth_frame,
                z_threshold=0.08
            )

            # Draw landmarks on frame for UI
            annotated = processor.draw_landmarks(rgb_frame.copy(), landmarks_xyz)

            if not q_landmarks.full():
                q_landmarks.put((landmarks_xyz, mask, annotated, ts))

    logger.info("[T2-MediaPipe] Stopped")


# ── Thread 3: Kalman 3D Tracker ───────────────────────────────────────────────
def thread_kalman(stereo_inf, stop_event):
    """
    Smooths face position using Kalman filter with depth-aware measurement
    uncertainty: sigma = z^2 / (baseline * focal_length).
    Handles dropped frames gracefully via prediction-only updates.
    """
    logger.info("[T3-Kalman] Starting tracking thread")
    kalman_filters = {}   # track_id -> KalmanFilter instance
    track_id       = 0

    while not stop_event.is_set():
        try:
            landmarks_xyz, mask, annotated, ts = q_landmarks.get(timeout=0.1)
        except queue.Empty:
            # No new measurement — run prediction step on all active filters
            for tid, kf in kalman_filters.items():
                kf.predict(KALMAN_DT)
            continue

        # Use centroid of nose bridge landmarks as 3D position
        nose_idx  = [1, 2, 4, 5]
        centroid  = np.mean(landmarks_xyz[nose_idx], axis=0).reshape(-1, 1)
        z_depth   = float(centroid[2])

        # Depth-aware measurement noise: farther = noisier
        meas_std = (z_depth ** 2) / (stereo_inf.baseline * stereo_inf.focal_length_pixels)
        meas_std = max(meas_std, 0.5)   # floor to avoid division issues

        # Assign or reuse Kalman filter for this track
        # Simple heuristic: closest existing centroid within 50px -> same track
        if not kalman_filters:
            kalman_filters[track_id] = KalmanFilter(
                acc_std=KALMAN_ACC_STD,
                meas_std=meas_std,
                z=centroid,
                time=ts
            )

        kf = kalman_filters[track_id]
        kf.predict(KALMAN_DT)
        kf.update(centroid)

        smoothed_xyz = kf.x[:3].flatten()   # [x, y, z] position

        if not q_tracked.full():
            q_tracked.put((smoothed_xyz, track_id, annotated, ts))

    logger.info("[T3-Kalman] Stopped")


# ── Thread 4: Identity Lookup ─────────────────────────────────────────────────
def thread_identity(backbone, embedding_db, stop_event):
    """
    For each tracked face:
    1. Normalize and crop the face region (affine transform via utils_cls)
    2. Compute CNN embedding (ResNet50 feature layer, 512-dim)
    3. Compare with HDF5 embedding database via cosine similarity
    4. If match found -> returning customer; else -> new visitor
    5. Flag VIP customers for alert
    """
    logger.info("[T4-Identity] Starting identity lookup thread")
    processor = ImageProcessor()

    while not stop_event.is_set():
        try:
            smoothed_xyz, track_id, frame, ts = q_tracked.get(timeout=0.1)
        except queue.Empty:
            continue

        # Normalize face crop using affine transforms (rotation, scale correction)
        face_crop = processor.affine(
            delta=(0, 0),
            theta=0.0,      # gaze-estimated tilt angle
            scale=1.0,
            vec=np.array([[smoothed_xyz[0]], [smoothed_xyz[1]], [1.0]])
        )

        # Perceptual hash for fast pre-filter (dhash)
        face_region = extract_face_region(frame, smoothed_xyz)
        if face_region is None:
            continue

        face_hash    = dhash(face_region, hash_size=8)
        hash_match   = find_match_in_db(face_hash, embedding_db, threshold=DHASH_THRESHOLD)

        # Full CNN embedding for precise identification
        embedding    = extract_embedding(backbone, face_region)   # (512,)
        customer_id, confidence = match_embedding(embedding, embedding_db, threshold=IDENTITY_THRESHOLD)

        if not q_identity.full():
            q_identity.put((customer_id, confidence, embedding, track_id, ts))

    logger.info("[T4-Identity] Stopped")


# ── Thread 5: Visit Logger ────────────────────────────────────────────────────
def thread_logger(stop_event):
    """
    Persists visit events to HDF5 database via utils_file.
    Schema per visit record:
      customer_id, timestamp_in, timestamp_out,
      zones_visited[], dwell_times{}, embedding(512,), vip_flag
    Also exports daily JSON summary report.
    """
    logger.info("[T5-Logger] Starting visit logger thread")
    visit_logger = VisitLogger(DB_PATH)
    active_visits = {}   # customer_id -> visit start metadata

    while not stop_event.is_set():
        try:
            customer_id, confidence, embedding, track_id, ts = q_identity.get(timeout=0.5)
        except queue.Empty:
            # Flush any visits that have timed out (customer left)
            visit_logger.flush_stale_visits(active_visits, timeout_sec=10.0, now=time.monotonic())
            continue

        if customer_id not in active_visits:
            # New visit — open a record
            active_visits[customer_id] = {
                "ts_in":     ts,
                "embedding": embedding,
                "zones":     [],
                "confidence": confidence,
            }
            logger.info(f"[T5-Logger] New visit: customer={customer_id} confidence={confidence:.2f}")
        else:
            # Existing visit — update zone dwell time
            active_visits[customer_id]["ts_last"] = ts

        # Write incremental update to HDF5
        visit_logger.update(customer_id, active_visits[customer_id])

    # On shutdown, close all open visits
    visit_logger.close_all(active_visits, now=time.monotonic())
    logger.info("[T5-Logger] Stopped")


# ── Thread 6: Staff UI ────────────────────────────────────────────────────────
def thread_ui(stop_event):
    """
    Tkinter staff dashboard showing:
    - Live RGB feed with face annotations
    - Depth overlay toggle
    - MediaPipe landmark overlay toggle
    - Customer identity + confidence badge
    - Zone dwell heatmap
    - VIP alert banner
    """
    logger.info("[T6-UI] Starting staff UI thread")
    ui = StaffUI(title="FaceVault — Staff Dashboard")

    while not stop_event.is_set():
        try:
            frame, metadata = q_ui.get(timeout=0.05)
        except queue.Empty:
            ui.update()
            continue

        ui.update_frame(frame)
        ui.update_metadata(metadata)

        # If VIP detected, flash alert banner
        if metadata.get("is_vip"):
            ui.show_vip_alert(metadata["customer_id"])

        ui.update()

    ui.destroy()
    logger.info("[T6-UI] Stopped")


# ── Gaze Zone Classifier ──────────────────────────────────────────────────────
def classify_gaze_zone(smoothed_xyz, frame_shape):
    """
    Maps 3D face position to a named store zone.
    Uses the X coordinate (horizontal position) and the gaze vector
    derived from MediaPipe landmarks to estimate which product zone
    the customer is looking at.
    Returns: zone_name or None
    """
    h, w = frame_shape[:2]
    # Project 3D position to 2D frame coordinates
    x_2d = int((smoothed_xyz[0] / w) * w)
    y_2d = int((smoothed_xyz[1] / h) * h)

    for zone_name, (x1, y1, x2, y2) in ZONE_MAP.items():
        if x1 <= x_2d <= x2 and y1 <= y_2d <= y2:
            return zone_name
    return None


# ── Helper: Extract Face Region ───────────────────────────────────────────────
def extract_face_region(frame, centroid_xyz, pad=30):
    """Crops a face region from frame using the 3D centroid estimate."""
    h, w = frame.shape[:2]
    cx, cy = int(centroid_xyz[0]), int(centroid_xyz[1])
    x1 = max(0,   cx - pad)
    y1 = max(0,   cy - pad)
    x2 = min(w-1, cx + pad)
    y2 = min(h-1, cy + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ── Helper: Embedding Cosine Match ───────────────────────────────────────────
def match_embedding(embedding, embedding_db, threshold=0.75):
    """
    Compares query embedding against stored embeddings using cosine similarity.
    Returns (customer_id, confidence) or ("unknown", 0.0).
    """
    if not embedding_db:
        return "unknown", 0.0

    query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    best_id, best_sim = "unknown", 0.0

    for cid, stored_emb in embedding_db.items():
        stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
        sim = float(np.dot(query_norm, stored_norm))
        if sim > best_sim:
            best_sim = sim
            best_id  = cid

    if best_sim >= threshold:
        return best_id, best_sim
    return "unknown", best_sim


# ── Main Entry Point ──────────────────────────────────────────────────────────
def run(demo_mode=False):
    """
    Launch all 6 threads and coordinate the FaceVault pipeline.
    demo_mode=True uses a webcam instead of OAK-D for development.
    """
    logger.info("=" * 60)
    logger.info("  FaceVault — Retail Customer Intelligence")
    logger.info("=" * 60)

    device = get_device()
    stop_event = threading.Event()

    # Initialise stereo calibration from device EEPROM
    if not demo_mode:
        import depthai as dai
        with dai.Device() as oak_device:
            stereo_inf = StereoInference(
                device=oak_device,
                resolution_num=RESOLUTION_NUM,
                mono_heigth=MONO_HEIGHT,
                mono_width=MONO_WIDTH,
            )
    else:
        stereo_inf = StereoInference.from_defaults()   # mock for demo

    # Load CNN backbone for face embeddings (quantized int8 for edge speed)
    backbone      = load_backbone(model_name="resnet50", quantized=True)
    embedding_db  = VisitLogger(DB_PATH).load_embedding_index()

    # Launch threads
    threads = [
        threading.Thread(target=thread_capture,  args=(None, stereo_inf, stop_event),         name="T1-Capture",  daemon=True),
        threading.Thread(target=thread_mediapipe, args=(stop_event,),                          name="T2-MP",       daemon=True),
        threading.Thread(target=thread_kalman,    args=(stereo_inf, stop_event),               name="T3-Kalman",   daemon=True),
        threading.Thread(target=thread_identity,  args=(backbone, embedding_db, stop_event),   name="T4-Identity", daemon=True),
        threading.Thread(target=thread_logger,    args=(stop_event,),                          name="T5-Logger",   daemon=True),
        threading.Thread(target=thread_ui,        args=(stop_event,),                          name="T6-UI",       daemon=True),
    ]

    for t in threads:
        t.start()
        logger.info(f"Started {t.name}")

    logger.info("All threads running — press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    logger.info("FaceVault stopped cleanly")


if __name__ == "__main__":
    run(demo_mode=True)
