"""
FaceVault Utils — Kalman Filter for 3D Face Tracking.

6-DoF tracking with a constant-acceleration motion model.
State vector: [x, y, z, vx, vy, vz, ax, ay, az] — position, velocity, acceleration.

Key innovation: depth-aware measurement uncertainty.
  sigma = z^2 / (baseline * focal_length)
Stereo cameras get noisier with distance, so the Kalman filter
automatically trusts near measurements more than far ones.

Based on actual utils_filters.py implementation by Tal Toledano.
"""

import numpy as np
import logging

logger = logging.getLogger("facevault.filters")


class KalmanFilter:
    """
    3D Kalman filter with constant-acceleration state model.

    State: [pos(dim_z), vel(dim_z), acc(dim_z)]  shape: (3*dim_z, 1)
    Measurement: [pos(dim_z)]                     shape: (dim_z, 1)

    Example:
        z0 = np.array([[320.], [240.], [2500.]])
        kf = KalmanFilter(acc_std=0.5, meas_std=1.2, z=z0, time=0.0)
        kf.predict(dt=1/30)
        kf.update(z_new)
        pos = kf.position   # smoothed XYZ
    """

    def __init__(self, acc_std: float, meas_std: float,
                 z: np.ndarray, time: float):
        """
        Args:
            acc_std:  Process noise — std dev of acceleration changes between frames.
                      Higher = more responsive to sudden movements; lower = smoother.
            meas_std: Measurement noise — depth-derived sigma (z^2 / baseline*focal).
            z:        Initial measurement, shape (dim_z, 1).
            time:     Timestamp of first measurement (monotonic seconds).
        """
        self.dim_z    = z.shape[0]
        self.time     = time
        self.acc_std  = acc_std
        self.meas_std = meas_std

        # Observation matrix H — extracts position from full state
        # H @ x = position   (first dim_z elements of state)
        self.H = np.eye(self.dim_z, 3 * self.dim_z)

        # Initial state: position = z, velocity = 0, acceleration = 0
        self.x = np.vstack([z, np.zeros((2 * self.dim_z, 1))])

        # Initial covariance: high uncertainty (our initial estimate is a guess)
        self.P = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        i, j = np.indices(self.P.shape)
        self.P[(i - j) % self.dim_z == 0] = 1e5

        logger.debug(f"KalmanFilter init: dim={self.dim_z} "
                     f"acc_std={acc_std} meas_std={meas_std:.3f}")

    def predict(self, dt: float) -> None:
        """
        Predict next state assuming constant acceleration.
        F encodes: pos += vel*dt + 0.5*acc*dt^2
                   vel += acc*dt
        """
        # State transition matrix
        F = np.eye(3 * self.dim_z)
        np.fill_diagonal(F[:2 * self.dim_z,     self.dim_z:],     dt)
        np.fill_diagonal(F[:    self.dim_z,  2 * self.dim_z:],  0.5 * dt**2)

        # Process noise: uncertainty grows with assumed acceleration variance
        A = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        np.fill_diagonal(A[2 * self.dim_z:, 2 * self.dim_z:], 1)
        Q = self.acc_std ** 2 * (F @ A @ F.T)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray) -> None:
        """
        Update state with new measurement z (shape: dim_z x 1).
        Skips update gracefully if z is None (e.g. MediaPipe missed a frame).
        """
        if z is None:
            return

        R = self.meas_std ** 2 * np.eye(self.dim_z)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain

        innovation = z - self.H @ self.x
        self.x = self.x + K @ innovation

        I = np.eye(3 * self.dim_z)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T

    # ── Convenience accessors ─────────────────────────────────────────────────
    @property
    def position(self) -> np.ndarray:
        """Smoothed 3D position [x, y, z]."""
        return self.x[:self.dim_z].flatten()

    @property
    def velocity(self) -> np.ndarray:
        """Estimated 3D velocity [vx, vy, vz]."""
        return self.x[self.dim_z:2*self.dim_z].flatten()

    @property
    def uncertainty(self) -> np.ndarray:
        """Position uncertainty (diagonal of P, position block)."""
        return np.diag(self.P[:self.dim_z, :self.dim_z])


class MultiObjectTracker:
    """
    Manages one KalmanFilter per tracked customer face.

    Handles:
      - Track creation for new faces
      - Track association via nearest-centroid matching
      - Track pruning when a customer leaves frame
    """

    def __init__(self, acc_std=0.5, max_age_frames=30, match_thresh_px=80.0):
        self.acc_std        = acc_std
        self.max_age        = max_age_frames
        self.match_thresh   = match_thresh_px
        self._filters       = {}    # track_id -> KalmanFilter
        self._ages          = {}    # track_id -> frames since last update
        self._next_id       = 0

    def update(self, detections: list, dt: float, depth_sigma: float) -> dict:
        """
        Args:
            detections: list of np.ndarray shape (3,1) — measured XYZ centroids
            dt:         time delta in seconds
            depth_sigma: measurement uncertainty (from stereo depth formula)

        Returns:
            dict: track_id -> smoothed position array (3,)
        """
        # Predict all existing tracks
        for tid, kf in self._filters.items():
            kf.predict(dt)
            self._ages[tid] += 1

        # Greedy nearest-centroid assignment
        assigned_det = set()
        for tid, kf in list(self._filters.items()):
            best_dist, best_det_idx = float("inf"), None
            for i, det in enumerate(detections):
                if i in assigned_det:
                    continue
                dist = float(np.linalg.norm(kf.position - det.flatten()))
                if dist < best_dist:
                    best_dist, best_det_idx = dist, i

            if best_det_idx is not None and best_dist < self.match_thresh:
                self._filters[tid].update(detections[best_det_idx])
                self._ages[tid] = 0
                assigned_det.add(best_det_idx)

        # Create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in assigned_det:
                import time
                self._filters[self._next_id] = KalmanFilter(
                    acc_std=self.acc_std,
                    meas_std=depth_sigma,
                    z=det,
                    time=time.monotonic()
                )
                self._ages[self._next_id] = 0
                self._next_id += 1

        # Prune stale tracks
        stale = [tid for tid, age in self._ages.items() if age > self.max_age]
        for tid in stale:
            del self._filters[tid]
            del self._ages[tid]
            logger.debug(f"Track {tid} pruned (age > {self.max_age})")

        return {tid: kf.position for tid, kf in self._filters.items()}


def run_kalman(frame, tracker, kalman_filters: dict) -> dict:
    """
    Convenience wrapper matching the original utils_filters.py signature.
    Extracts tracklets from a DepthAI tracker, applies Kalman smoothing.
    Returns dict of track_id -> smoothed (x, y, z).
    """
    current_time = tracker.getTimestamp().total_seconds()

    for t in tracker.tracklets:
        tid  = t.id
        roi  = t.roi.denormalize(frame.shape[1], frame.shape[0])
        cx   = float((roi.x + roi.width)  / 2)
        cy   = float((roi.y + roi.height) / 2)
        z    = float(t.spatialCoordinates.z) if hasattr(t, "spatialCoordinates") else 2000.0

        centroid = np.array([[cx], [cy], [z]])

        if tid not in kalman_filters:
            kalman_filters[tid] = KalmanFilter(
                acc_std=0.5, meas_std=1.2, z=centroid, time=current_time
            )

        kf = kalman_filters[tid]
        kf.predict(dt=1/30)
        kf.update(centroid)

    return {tid: kf.position for tid, kf in kalman_filters.items()}
