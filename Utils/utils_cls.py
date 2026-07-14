"""
FaceVault Utils — ImageProcessor class for drawing and affine transforms.

Handles all annotation drawing on live frames and geometric normalisation
of face crops before CNN embedding extraction.

Based on actual utils_cls.py by Tal Toledano.
"""

import cv2
import numpy as np
import logging
from functools import lru_cache

logger = logging.getLogger("facevault.cls")

B, G, R = (255, 0, 0), (0, 255, 0), (0, 0, 255)
WHITE   = (255, 255, 255)
TARGET_SHAPE = (256, 256)


class ImageProcessor:
    """
    Stateless helper class for frame annotation and geometric transforms.

    Used by Thread 2 (MediaPipe overlay), Thread 4 (identity bounding boxes),
    and the analytics reporter (zone visualisation).
    """

    def __init__(self):
        self.images = []

    # ── Drawing ───────────────────────────────────────────────────────────────

    def add_rec(self, image: np.ndarray, coord: tuple,
                color=R, thickness: int = 2, linetype: int = cv2.LINE_AA) -> None:
        """Draw a bounding rectangle. coord = (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = [int(c) for c in coord]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, linetype)

    def add_line(self, image: np.ndarray, pts: tuple,
                 color=WHITE, lineW: int = 2) -> None:
        """Draw a line between two points."""
        p1, p2 = pts
        cv2.line(image, tuple(map(int, p1)), tuple(map(int, p2)), color, lineW)

    def add_circ(self, image: np.ndarray, coord: tuple,
                 color=B, radius: int = 5, line: int = -1) -> None:
        """Draw a filled circle."""
        cv2.circle(image, tuple(map(int, coord)), radius, color, line)

    def add_txt(self, image: np.ndarray, text, coord: tuple,
                color=WHITE, line: int = 2, scale: float = 0.7) -> None:
        """Overlay text on frame. text can be str or float."""
        label = f"{text:.2f}" if isinstance(text, float) else str(text)
        x, y  = int(coord[0]), int(coord[1]) - 10
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, line, cv2.LINE_AA)

    def add_polys(self, image: np.ndarray, rectangles: list) -> None:
        """Draw multiple polygon outlines."""
        for rect in rectangles:
            cv2.polylines(image, [rect], isClosed=True, color=R, thickness=2)

    def draw_landmarks(self, image: np.ndarray,
                        landmarks: np.ndarray,
                        color: tuple = (0, 200, 100)) -> np.ndarray:
        """Draw MediaPipe landmark dots on frame."""
        from Utils.utils_mp import draw_landmarks as mp_draw
        return mp_draw(image, landmarks, color=color)

    def draw_identity_badge(self, image: np.ndarray,
                             bbox: tuple, customer_id: str,
                             confidence: float, is_vip: bool = False) -> None:
        """
        Draw a labelled bounding box with customer ID and confidence score.
        VIP customers get a gold border and star prefix.
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        color = (0, 215, 255) if is_vip else G    # Gold for VIP, green otherwise

        self.add_rec(image, (x1, y1, x2, y2), color=color, thickness=2)

        label = f"{'★ VIP ' if is_vip else ''}{customer_id} ({confidence:.0%})"
        bg_w  = len(label) * 8 + 4
        cv2.rectangle(image, (x1, y1-20), (x1+bg_w, y1), color, -1)
        cv2.putText(image, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_zone_overlay(self, image: np.ndarray,
                           zone_map: dict, active_zone: str = None) -> None:
        """
        Draw translucent zone rectangles on frame.
        Highlights the active zone (where the customer is looking) in yellow.
        """
        overlay = image.copy()
        for zone_name, (x1, y1, x2, y2) in zone_map.items():
            color = YELLOW if zone_name == active_zone else (200, 200, 200)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            self.add_txt(image, zone_name, (x1+5, y1+20), color=(0,0,0), scale=0.5)

        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

        # Draw borders on top
        for zone_name, (x1, y1, x2, y2) in zone_map.items():
            border = (0, 215, 255) if zone_name == active_zone else (150, 150, 150)
            self.add_rec(image, (x1, y1, x2, y2), color=border, thickness=1)

    # ── Affine Transforms ─────────────────────────────────────────────────────

    def affine(self, delta: tuple, theta: float,
               scale: float, vec: np.ndarray) -> np.ndarray:
        """
        Compose rotation, translation, and scaling into one affine transform.

        Args:
            delta:  (dx, dy) translation
            theta:  rotation angle in radians
            scale:  uniform scale factor
            vec:    input vectors, shape (3, N) in homogeneous coords

        Returns:
            Transformed vectors shape (3, N).
        """
        if vec.ndim == 3:
            vec = vec[:, :, 0]

        R_mat  = self.rot(theta, vec)
        T_mat  = self.tran(delta, vec)
        S_mat  = self.scl(scale, vec)

        return (S_mat @ R_mat + T_mat) @ vec.T

    def tran(self, delta: tuple, vec: np.ndarray) -> np.ndarray:
        """3x3 translation matrix."""
        dx, dy = delta if isinstance(delta, (tuple, list)) else (delta, delta)
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)

    def rot(self, theta: float, vec: np.ndarray) -> np.ndarray:
        """3x3 rotation matrix (2D rotation in homogeneous coords)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)

    def scl(self, scale: float, vec: np.ndarray) -> np.ndarray:
        """3x3 uniform scale matrix."""
        return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)

    def normalise_face_orientation(self, image: np.ndarray,
                                    landmarks: np.ndarray) -> np.ndarray:
        """
        Align face crop to canonical frontal orientation using
        eye landmark positions for rotation correction.
        Improves embedding consistency across poses.
        """
        if landmarks is None:
            return image

        h, w = image.shape[:2]

        # Left eye centre (avg of eye landmarks 33, 133)
        # Right eye centre (avg of 362, 263)
        le = landmarks[[33, 133]].mean(axis=0)
        re = landmarks[[362, 263]].mean(axis=0)

        le_px = (int(le[0]*w), int(le[1]*h))
        re_px = (int(re[0]*w), int(re[1]*h))

        # Angle between eye line and horizontal
        dy = re_px[1] - le_px[1]
        dx = re_px[0] - le_px[0]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Rotate image to correct tilt
        M = cv2.getRotationMatrix2D(
            center=(w//2, h//2), angle=angle, scale=1.0
        )
        aligned = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        return aligned


YELLOW = (0, 255, 255)
