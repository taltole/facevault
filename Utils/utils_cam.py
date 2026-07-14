"""
FaceVault Utils — OAK-D Stereo Camera Calibration & Pipeline.

StereoInference reads calibration data directly from the OAK-D device EEPROM:
  - baseline (mm) between left and right cameras
  - focal length (pixels) at configured resolution
  - intrinsic and extrinsic matrices
  - disparity-to-depth conversion factor

Also provides build_oak_pipeline() which wires up the DepthAI pipeline
for RGB + depth streaming used by Thread 1.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger("facevault.cam")


@dataclass
class CameraCalibration:
    """Immutable calibration bundle read from device EEPROM."""
    baseline_mm:          float
    focal_length_pixels:  float
    hfov_deg:             float
    mono_width:           int
    mono_height:          int
    color_width:          int   = 1920
    color_height:         int   = 1080
    intrinsics_left:      Optional[np.ndarray] = field(default=None, repr=False)
    intrinsics_right:     Optional[np.ndarray] = field(default=None, repr=False)
    extrinsics_lr:        Optional[np.ndarray] = field(default=None, repr=False)


class StereoInference:
    """
    Wraps OAK-D device calibration into a clean interface.

    Usage (real device):
        with dai.Device() as device:
            si = StereoInference(device, resolution_num=400,
                                 mono_heigth=400, mono_width=640)

    Usage (demo / test):
        si = StereoInference.from_defaults()
    """

    def __init__(self, device, resolution_num: int,
                 mono_heigth: int, mono_width: int, verbose: bool = True):
        try:
            import depthai as dai
            calibData = device.readCalibration()

            # Extrinsics between left (CAM_B) and right (CAM_C)
            lr_ext = np.array(
                calibData.getCameraExtrinsics(
                    dai.CameraBoardSocket.CAM_B,
                    dai.CameraBoardSocket.CAM_C,
                    useSpecTranslation=True
                )
            )
            tvec           = lr_ext[0:3, 3:4].flatten()
            baseline_cm    = np.sqrt(np.sum(tvec ** 2))

            self.calib              = CameraCalibration(
                baseline_mm         = baseline_cm * 10,
                focal_length_pixels = calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RIGHT, mono_width, mono_heigth)[0][0],
                hfov_deg            = calibData.getFov(dai.CameraBoardSocket.CAM_B),
                mono_width          = mono_width,
                mono_height         = mono_heigth,
                intrinsics_left     = np.array(calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.LEFT,  mono_width, mono_heigth)),
                intrinsics_right    = np.array(calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RIGHT, mono_width, mono_heigth)),
                extrinsics_lr       = lr_ext,
            )

            self.resize_factor = resolution_num / mono_heigth

        except Exception as e:
            logger.warning(f"OAK-D not available ({e}), using defaults")
            self.calib         = self._default_calib(mono_width, mono_heigth)
            self.resize_factor = 1.0

        if verbose:
            logger.info(f"StereoInference: baseline={self.baseline:.1f}mm "
                        f"focal={self.focal_length_pixels:.1f}px "
                        f"hfov={self.hfov:.1f}°")

    @classmethod
    def from_defaults(cls, mono_width=640, mono_height=400):
        """Mock calibration for demo/test without a physical OAK-D."""
        obj = object.__new__(cls)
        obj.calib         = cls._default_calib(mono_width, mono_height)
        obj.resize_factor = 1.0
        return obj

    @staticmethod
    def _default_calib(mono_width=640, mono_height=400) -> CameraCalibration:
        """Reasonable defaults matching a standard OAK-D Lite."""
        fx = 860.0   # typical focal length at 640x400
        return CameraCalibration(
            baseline_mm=75.0,
            focal_length_pixels=fx,
            hfov_deg=80.0,
            mono_width=mono_width,
            mono_height=mono_height,
            intrinsics_left=np.array([[fx,0,320],[0,fx,200],[0,0,1]],dtype=np.float64),
            intrinsics_right=np.array([[fx,0,320],[0,fx,200],[0,0,1]],dtype=np.float64),
        )

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def baseline(self) -> float:
        return self.calib.baseline_mm

    @property
    def focal_length_pixels(self) -> float:
        return self.calib.focal_length_pixels

    @property
    def hfov(self) -> float:
        return self.calib.hfov_deg

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Convert disparity map (pixels) to depth map (mm)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(
                disparity > 0,
                (self.baseline * self.focal_length_pixels) / disparity,
                0
            )
        return depth.astype(np.float32)

    def depth_measurement_sigma(self, z_mm: float) -> float:
        """
        Depth-aware measurement uncertainty for Kalman filter.
        sigma = z^2 / (baseline * focal_length)
        Grows quadratically with distance — stereo gets noisier far away.
        """
        return max((z_mm ** 2) / (self.baseline * self.focal_length_pixels), 0.5)

    @staticmethod
    def get_focal_length_pixels(width: int, hfov_deg: float) -> float:
        """Compute focal length in pixels from frame width and horizontal FOV."""
        import math
        return (width * 0.5) / math.tan(math.radians(hfov_deg * 0.5))

    def __repr__(self):
        return (f"StereoInference(baseline={self.baseline:.1f}mm, "
                f"focal={self.focal_length_pixels:.1f}px, hfov={self.hfov:.1f}°)")


def build_oak_pipeline(device):
    """
    Wires up a DepthAI pipeline for synchronized RGB + depth streaming.
    Returns (q_rgb, q_depth) output queues ready to call .get() on.

    Falls back to a webcam-based mock when OAK-D is unavailable (demo mode).
    """
    try:
        import depthai as dai

        pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 400)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Stereo depth
        mono_left  = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo     = pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Output queues
        xout_rgb   = pipeline.create(dai.node.XLinkOut)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")

        cam_rgb.preview.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        logger.info("OAK-D pipeline started (RGB + Depth)")
        return q_rgb, q_depth

    except Exception as e:
        logger.warning(f"OAK-D pipeline unavailable ({e}), using webcam mock")
        return _build_webcam_mock()


def _build_webcam_mock():
    """
    Demo fallback: wraps OpenCV webcam in queue-like objects
    so the rest of the pipeline runs identically in demo mode.
    """
    import cv2
    import queue as q_module
    import threading

    rgb_q   = q_module.Queue(maxsize=4)
    depth_q = q_module.Queue(maxsize=4)
    cap     = cv2.VideoCapture(0)

    class FrameWrapper:
        def __init__(self, frame): self._frame = frame
        def getCvFrame(self):      return self._frame
        def getFrame(self):        return self._frame

    class MockQueue:
        def __init__(self, inner): self._q = inner
        def get(self):
            return FrameWrapper(self._q.get())

    def _capture():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            depth = np.full(frame.shape[:2], 2000, dtype=np.uint16)
            if not rgb_q.full():
                rgb_q.put(frame)
            if not depth_q.full():
                depth_q.put(depth)

    threading.Thread(target=_capture, daemon=True).start()
    logger.info("Webcam mock pipeline started")
    return MockQueue(rgb_q), MockQueue(depth_q)
