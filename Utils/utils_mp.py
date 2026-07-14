"""
FaceVault Utils — MediaPipe Face Mesh Analysis.

Extracts 468 3D face landmarks from each frame, then:
  - Generates a depth-aware convex-hull face mask
  - Exports landmark XYZ coordinates for gaze estimation
  - Draws annotated overlays for the staff UI

Based on actual utils_mp.py implementation by Tal Toledano.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger("facevault.mp")

TARGET_SHAPE = (256, 256)

# Nose bridge landmark indices (used for gaze centroid)
NOSE_BRIDGE_IDX = [1, 2, 4, 5, 6, 168, 195, 197]

# Outer face oval indices (used for mask boundary)
FACE_OVAL_IDX   = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                   361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                   176, 149, 150, 136, 172, 58,  132, 93,  234, 127,
                   162, 21,  54,  103, 67,  109]


def extract_mp_landmarks(image: np.ndarray,
                          min_detection_confidence: float = 0.5,
                          min_tracking_confidence: float  = 0.5
                          ) -> Optional[np.ndarray]:
    """
    Run MediaPipe FaceMesh on a BGR frame.

    Returns:
        np.ndarray shape (468, 3) — normalised (x, y, z) per landmark,
        where x, y are in [0,1] relative to frame dims.
        Returns None if no face detected.
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh

        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as face_mesh:
            results = face_mesh.process(rgb)

        image.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        face_lms = results.multi_face_landmarks[0]
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_lms.landmark],
            dtype=np.float32
        )  # shape (468, 3)

        return landmarks

    except ImportError:
        logger.warning("MediaPipe not installed — returning mock landmarks")
        return _mock_landmarks()
    except Exception as e:
        logger.error(f"MediaPipe error: {e}")
        return None


def get_mp_depthmask(frame: np.ndarray,
                      landmarks: np.ndarray,
                      depth_frame: Optional[np.ndarray] = None,
                      z_threshold: float = 0.08) -> np.ndarray:
    """
    Generate a binary face mask using MediaPipe face oval landmarks,
    filtered by depth (Z coordinate) to exclude background clutter.

    The mask is a convex hull of the face oval landmarks whose Z value
    is within z_threshold of the median — filtering out ear/hair landmarks
    that MediaPipe sometimes incorrectly assigns at wrong depths.

    Args:
        frame:       BGR frame (used for output shape)
        landmarks:   (468, 3) array from extract_mp_landmarks
        depth_frame: Optional depth map (mm) for additional spatial filtering
        z_threshold: Max Z deviation to include landmark in mask

    Returns:
        Binary mask np.ndarray (H, W, 1) — uint8, 255 inside face, 0 outside.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w, 1), dtype=np.uint8)

    if landmarks is None:
        return mask

    # Filter face oval landmarks by Z depth
    oval_lms = landmarks[FACE_OVAL_IDX]
    z_median = np.median(oval_lms[:, 2])
    keep     = np.abs(oval_lms[:, 2] - z_median) < z_threshold

    filtered = oval_lms[keep]
    if len(filtered) < 4:
        return mask

    # Convert normalised coords -> pixel coords
    pts = np.array(
        [[int(lm[0] * w), int(lm[1] * h)] for lm in filtered],
        dtype=np.int32
    )

    # Convex hull mask
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    # Optional: AND with depth-based foreground (pixels within 500mm of face centroid)
    if depth_frame is not None:
        nose_lms  = landmarks[NOSE_BRIDGE_IDX]
        cx        = int(np.mean(nose_lms[:, 0]) * w)
        cy        = int(np.mean(nose_lms[:, 1]) * h)
        cx        = np.clip(cx, 0, w-1)
        cy        = np.clip(cy, 0, h-1)

        face_depth = float(depth_frame[cy, cx])
        if face_depth > 0:
            depth_near = (depth_frame > face_depth - 500).astype(np.uint8) * 255
            depth_near = depth_near[:, :, np.newaxis]
            mask       = cv2.bitwise_and(mask, depth_near)

    return mask


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray,
                    color: tuple = (0, 200, 100), radius: int = 1) -> np.ndarray:
    """
    Draw all 468 face landmarks as coloured dots on the frame.
    Used by Thread 2 to produce the annotated frame for the staff UI.
    """
    if landmarks is None:
        return image

    h, w = image.shape[:2]
    annotated = image.copy()

    for lm in landmarks:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        cv2.circle(annotated, (x, y), radius, color, -1)

    # Also draw nose bridge in a distinct colour
    for idx in NOSE_BRIDGE_IDX:
        lm = landmarks[idx]
        x  = int(lm[0] * w)
        y  = int(lm[1] * h)
        cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

    return annotated


def get_gaze_centroid(landmarks: np.ndarray,
                       frame_shape: tuple) -> Optional[Tuple[int, int, float]]:
    """
    Estimate gaze centroid (x, y) in pixel coords and depth z from nose bridge.

    Returns:
        (cx_px, cy_px, z_norm) or None if landmarks invalid.
    """
    if landmarks is None:
        return None

    h, w = frame_shape[:2]
    nose_lms = landmarks[NOSE_BRIDGE_IDX]
    cx = int(np.mean(nose_lms[:, 0]) * w)
    cy = int(np.mean(nose_lms[:, 1]) * h)
    z  = float(np.mean(nose_lms[:, 2]))

    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    return cx, cy, z


def _mock_landmarks() -> np.ndarray:
    """Return plausible synthetic landmarks for demo mode."""
    lms = np.zeros((468, 3), dtype=np.float32)
    # Place landmarks in a rough face oval
    for i in range(468):
        angle = (i / 468) * 2 * np.pi
        lms[i] = [
            0.5 + 0.15 * np.cos(angle),
            0.5 + 0.2  * np.sin(angle),
            -0.05 + np.random.randn() * 0.01,
        ]
    return lms
