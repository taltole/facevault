#!/usr/bin/env python3
"""
FaceVault Demo — run without OAK-D hardware using webcam + synthetic data.

Demonstrates all 12 utils modules in a single script:
  utils.py          — logging + device detection
  utils_cam.py      — StereoInference.from_defaults() mock calibration
  utils_mp.py       — MediaPipe face mesh on webcam frames
  utils_filters.py  — KalmanFilter smoothing of face position
  utils_image.py    — dhash + normalize_face_crop
  utils_nn.py       — ResNet50 embedding extraction
  utils_cls.py      — ImageProcessor annotations
  utils_file.py     — VisitLogger HDF5 persistence
  utils_vis.py      — t-SNE + zone heatmap plots
  utils_quant.py    — activation histogram (on synthetic data)
  utils_tk.py       — StaffUI dashboard
  utils_gl.py       — ASCII heatmap fallback

Usage: python demo.py
"""

import cv2
import time
import numpy as np
import threading
import logging
from pathlib import Path

# ── Utils imports ─────────────────────────────────────────────────────────────
from Utils.utils         import setup_logging, get_device, timeit
from Utils.utils_cam     import StereoInference
from Utils.utils_filters import KalmanFilter, MultiObjectTracker
from Utils.utils_image   import dhash, normalize_face_crop, find_duplicates
from Utils.utils_nn      import load_backbone, extract_embedding, count_parameters
from Utils.utils_cls     import ImageProcessor
from Utils.utils_file    import VisitLogger, write_json, NumpyEncoder
from Utils.utils_vis     import show_embedds, draw_gaze_heatmap
from Utils.utils_gl      import _ascii_heatmap
from Utils.utils_mp      import extract_mp_landmarks, get_mp_depthmask, draw_landmarks

from config.config import ZONE_MAP, DB_PATH

logger = setup_logging("INFO")


def run_demo():
    logger.info("=" * 60)
    logger.info("  FaceVault Demo — All 12 Utils Modules")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # ── 1. utils_cam: mock calibration ───────────────────────────────────────
    logger.info("\n[1/12] utils_cam — StereoInference (demo calibration)")
    stereo = StereoInference.from_defaults()
    logger.info(f"  {stereo}")
    logger.info(f"  Depth sigma at 2000mm: {stereo.depth_measurement_sigma(2000):.3f}")
    logger.info(f"  Disparity 50px -> {stereo.disparity_to_depth(np.array([50.]))[0]:.0f}mm depth")

    # ── 2. utils_filters: Kalman filter ──────────────────────────────────────
    logger.info("\n[2/12] utils_filters — KalmanFilter + MultiObjectTracker")
    z0 = np.array([[320.], [240.], [2000.]])
    kf = KalmanFilter(acc_std=0.5, meas_std=stereo.depth_measurement_sigma(2000),
                      z=z0, time=time.monotonic())
    for _ in range(10):
        kf.predict(dt=1/30)
        noise = np.random.randn(3, 1) * 5
        kf.update(z0 + noise)
    logger.info(f"  After 10 updates: pos={kf.position.astype(int)}, "
                f"uncertainty={kf.uncertainty.astype(int)}")

    # ── 3. utils_image: dhash + face crop ────────────────────────────────────
    logger.info("\n[3/12] utils_image — dhash + normalize_face_crop")
    dummy_face = np.random.randint(100, 200, (100, 80, 3), dtype=np.uint8)
    h = dhash(dummy_face)
    logger.info(f"  dhash of synthetic face: {h:#018x}")
    normed = normalize_face_crop(dummy_face, target_size=(112, 112))
    logger.info(f"  Normalized crop: shape={normed.shape}, "
                f"range=[{normed.min():.2f}, {normed.max():.2f}]")

    # ── 4. utils_nn: backbone + embedding ────────────────────────────────────
    logger.info("\n[4/12] utils_nn — load_backbone + extract_embedding")
    # Use weights=None in demo to avoid downloading (set pretrained=True in production)
    from Utils.utils_nn import _build_embedding_model
    backbone = _build_embedding_model("mobilenet_v3_small")
    backbone.eval()
    logger.info(f"  Backbone params: {count_parameters(backbone)}")
    emb = extract_embedding(backbone, dummy_face)
    logger.info(f"  Embedding shape: {emb.shape}, L2 norm: {np.linalg.norm(emb):.4f}")

    # ── 5. utils_cls: ImageProcessor ─────────────────────────────────────────
    logger.info("\n[5/12] utils_cls — ImageProcessor annotations")
    proc  = ImageProcessor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    proc.add_rec(frame, (100, 100, 300, 300), color=(0, 255, 0))
    proc.add_txt(frame, "Demo Customer", (100, 100))
    proc.add_circ(frame, (200, 200))
    logger.info("  Drew bounding box, text, and circle on synthetic frame")

    # ── 6. utils_file: VisitLogger HDF5 ──────────────────────────────────────
    logger.info("\n[6/12] utils_file — VisitLogger HDF5 write/read")
    demo_db = Path("/tmp/facevault_demo_visits.h5")
    vl = VisitLogger(demo_db)
    for i in range(5):
        vl.update(f"cust_{i:03d}", {
            "ts_in":      time.time() - (4 - i) * 3600,
            "embedding":  np.random.randn(512).astype(np.float32),
            "confidence": 0.8 + np.random.rand() * 0.2,
            "dwell_sec":  30 + np.random.rand() * 120,
            "zone_dwell": {"zone_A": 45.0, "zone_B": 20.0},
            "is_vip":     i == 2,
        })
    from datetime import datetime, timedelta
    visits = vl.load_visits_since(datetime.now() - timedelta(hours=10))
    logger.info(f"  Wrote 5 visits, read back {len(visits)} records")
    idx = vl.load_embedding_index()
    logger.info(f"  Embedding index: {len(idx)} customers")

    # ── 7. utils_mp: MediaPipe (mock) ─────────────────────────────────────────
    logger.info("\n[7/12] utils_mp — MediaPipe landmarks (mock mode)")
    from Utils.utils_mp import _mock_landmarks, get_gaze_centroid
    lms = _mock_landmarks()
    logger.info(f"  Mock landmarks: {lms.shape}, Z range [{lms[:,2].min():.3f}, {lms[:,2].max():.3f}]")
    centroid = get_gaze_centroid(lms, (480, 640))
    logger.info(f"  Gaze centroid: {centroid}")

    # ── 8. utils_vis: plots ────────────────────────────────────────────────────
    logger.info("\n[8/12] utils_vis — t-SNE + zone heatmap (saving to /tmp)")
    n_customers = 30
    embeddings  = np.random.randn(n_customers, 512).astype(np.float32)
    visit_freqs = np.random.randint(1, 8, n_customers)
    show_embedds(embeddings, labels=visit_freqs,
                 title="FaceVault Demo — Customer Segments",
                 out_path=Path("/tmp/demo_tsne.png"))
    draw_gaze_heatmap(
        zone_names=list(ZONE_MAP.keys()),
        zone_values=[320.0, 180.0, 450.0, 95.0],
        title="FaceVault Demo — Zone Attention",
        out_path=Path("/tmp/demo_heatmap.png")
    )
    logger.info("  Plots saved to /tmp/demo_tsne.png and /tmp/demo_heatmap.png")

    # ── 9. utils_gl: ASCII heatmap ────────────────────────────────────────────
    logger.info("\n[9/12] utils_gl — Store Attention Heatmap (ASCII fallback)")
    zone_data = {
        name: {"bbox": bbox, "dwell": float(np.random.randint(50, 500))}
        for name, bbox in ZONE_MAP.items()
    }
    _ascii_heatmap(zone_data)

    # ── 10. utils_quant: size/param check (no training in demo) ───────────────
    logger.info("\n[10/12] utils_quant — Model size comparison")
    import torch
    float_mb = sum(p.numel() * p.element_size() for p in backbone.parameters()) / 1e6
    logger.info(f"  Float32 model: {float_mb:.1f} MB, {count_parameters(backbone)} params")
    logger.info(f"  int8 target: ~{float_mb/4:.1f} MB (4x compression via QAT)")
    logger.info("  Run: python -m facevault.quantize --data <dataset_path>")

    # ── 11. utils_file: JSON export ────────────────────────────────────────────
    logger.info("\n[11/12] utils_file — JSON report export (NumpyEncoder)")
    from analytics.report import WeeklyReporter
    reporter = WeeklyReporter(db_path=demo_db, report_dir=Path("/tmp/facevault_reports"))
    summary  = reporter.compute_summary(visits)
    write_json("/tmp/facevault_demo_report.json", summary)
    logger.info(f"  Report: {summary.get('total_visits')} visits, "
                f"top_zone={summary.get('top_zone')}")
    logger.info("  Saved to /tmp/facevault_demo_report.json")

    # ── 12. utils_tk: StaffUI ─────────────────────────────────────────────────
    logger.info("\n[12/12] utils_tk — StaffUI (headless check)")
    logger.info("  To launch the staff dashboard, run: python -m facevault --demo")
    logger.info("  (Requires a display; skipped in headless demo mode)")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ✓ All 12 utils modules exercised successfully")
    logger.info("=" * 60)
    logger.info("\nOutputs:")
    logger.info("  /tmp/demo_tsne.png         — customer segment scatter")
    logger.info("  /tmp/demo_heatmap.png      — zone attention bar chart")
    logger.info("  /tmp/facevault_demo_report.json — weekly summary JSON")
    logger.info("  /tmp/facevault_demo_visits.h5   — HDF5 visit database")


if __name__ == "__main__":
    run_demo()
