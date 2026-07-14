"""
FaceVault Utils — Core constants, logging, device detection, threading helpers.
Used across all 6 pipeline threads.
"""

import os
import cv2
import logging
import threading
import numpy as np
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from screeninfo import get_monitors
    _monitors = get_monitors()
    monitor_w = _monitors[0].width  if _monitors else 1920
    monitor_h = _monitors[0].height if _monitors else 1080
    monitor_name = "primary"
except Exception:
    monitor_w, monitor_h, monitor_name = 1920, 1080, "default"

# ── Visual separators (used in logging) ───────────────────────────────────────
breaker  = "-" * 60
sbreaker = "=" * 60

# ── Colour constants (BGR) ────────────────────────────────────────────────────
B, G, R = (255, 0, 0), (0, 255, 0), (0, 0, 255)
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
CYAN    = (255, 255, 0)
YELLOW  = (0, 255, 255)

# ── Global config (overridden by config.py) ───────────────────────────────────
VERBOSITY = 1
M_PATH    = Path(__file__).resolve().parent.parent / "models"
M_PATH.mkdir(parents=True, exist_ok=True)

TARGET_SHAPE = (256, 256)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with timestamp + level prefix."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("facevault")
    logger.info(f"Logging initialised at level {level}")
    logger.info(f"Monitor: {monitor_name} ({monitor_w}x{monitor_h})")
    return logger


def get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def timeit(fn):
    """Decorator: logs function execution time in ms."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        logging.getLogger("facevault").debug(f"{fn.__name__} took {ms:.1f}ms")
        return result
    return wrapper


def run_in_pool(fn, items, max_workers=4):
    """Run fn(item) for each item in parallel using a thread pool."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(fn, items))
    return results
