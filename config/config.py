"""
FaceVault — Configuration
Central config for all pipeline parameters.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
ASSETS_DIR   = ROOT_DIR / "assets"
DB_PATH      = ROOT_DIR / "data" / "visits.h5"
LOG_PATH     = ROOT_DIR / "data" / "logs"
MODEL_PATH   = ROOT_DIR / "models"
REPORT_DIR   = ROOT_DIR / "reports"

for d in [DB_PATH.parent, LOG_PATH, MODEL_PATH, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Camera (OAK-D) ───────────────────────────────────────────────────────────
MONO_HEIGHT      = 400          # px — stereo camera resolution height
MONO_WIDTH       = 640          # px — stereo camera resolution width
COLOR_HEIGHT     = 1080
COLOR_WIDTH      = 1920
RESOLUTION_NUM   = 400          # OAK-D resolution enum
FPS              = 30

# ── Face Detection & Recognition ─────────────────────────────────────────────
MIN_DETECTION_CONF   = 0.6      # MediaPipe face mesh minimum detection confidence
MIN_TRACKING_CONF    = 0.5      # MediaPipe tracking confidence
DHASH_SIZE           = 8        # perceptual hash size for returning-customer detection
DHASH_THRESHOLD      = 10       # hamming distance threshold — lower = stricter match
EMBEDDING_DIM        = 512      # CNN face embedding dimension (ResNet50 feature layer)
IDENTITY_THRESHOLD   = 0.75     # cosine similarity threshold for identity match

# ── Kalman Filter ─────────────────────────────────────────────────────────────
KALMAN_ACC_STD   = 0.5          # process noise — acceleration standard deviation
KALMAN_MEAS_STD  = 1.2          # measurement noise — depth-dependent (z^2 / baseline*focal)
KALMAN_DT        = 1.0 / FPS    # time step per frame

# ── Gaze & Attention ─────────────────────────────────────────────────────────
GAZE_DWELL_THRESHOLD_SEC = 2.0  # seconds of sustained gaze = "attention event"
ZONE_MAP = {                     # store product zones: name -> (x1,y1,x2,y2) in frame coords
    "entrance":    (0,    0,    320, 540),
    "zone_A":      (320,  0,    640, 540),
    "zone_B":      (640,  0,    960, 540),
    "checkout":    (960,  0,    1280,540),
}

# ── Quantisation ──────────────────────────────────────────────────────────────
QAT_EPOCHS       = 15           # quantisation-aware training epochs
QAT_LR           = 0.001
QAT_OBSERVER_OFF = 3            # epoch to disable observers
QAT_BN_FREEZE    = 5            # epoch to freeze BatchNorm stats
QUANT_BACKEND    = "fbgemm"     # quantisation backend (x86 = fbgemm, ARM = qnnpack)

# ── VIP Alert ────────────────────────────────────────────────────────────────
VIP_CUSTOMER_IDS = []           # list of known VIP embedding hashes — loaded at startup
VIP_ALERT_EMAIL  = os.getenv("FACEVAULT_ALERT_EMAIL", "manager@retailco.com")

# ── UI ────────────────────────────────────────────────────────────────────────
WINDOW_TITLE     = "FaceVault — Staff Dashboard"
UI_REFRESH_MS    = 33           # ~30 fps UI refresh

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL        = "INFO"
VERBOSITY        = 1
