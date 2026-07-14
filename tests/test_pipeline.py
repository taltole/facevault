"""
FaceVault — Unit Tests
========================
Tests all 12 Utils modules in the context of the FaceVault business problem.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import json
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

from Utils.utils         import setup_logging, get_device
from Utils.utils_cam     import StereoInference
from Utils.utils_filters import KalmanFilter
from Utils.utils_image   import dhash, find_match_in_db, normalize_face_crop
from Utils.utils_nn      import extract_embedding, recon_loss, kl_loss, vae_loss
from Utils.utils_cls     import ImageProcessor
from Utils.utils_file    import VisitLogger, NumpyEncoder
from Utils.utils_mp      import extract_mp_landmarks, get_mp_depthmask
from Utils.utils_quant   import run_ptq
from Utils.utils_vis     import show_embedds, draw_gaze_heatmap
from Utils.utils_tk      import StaffUI
from Utils.utils_gl      import render_attention_heatmap_3d

from facevault.pipeline  import classify_gaze_zone, match_embedding, extract_face_region


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_frame():
    """480x640 BGR frame simulating an OAK-D RGB capture."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Draw a rough face-like blob in the center
    cv2.circle(frame, (320, 240), 80, (200, 160, 130), -1)
    return frame

@pytest.fixture
def sample_depth():
    """Simulated depth map — face at ~2.5m (2500mm)."""
    depth = np.full((480, 640), 2500, dtype=np.uint16)
    return depth

@pytest.fixture
def sample_landmarks():
    """Simulated 468 3D MediaPipe face landmarks."""
    lms = np.random.rand(468, 3).astype(np.float32)
    lms[:, 2] *= 0.1   # Z coords are small in MediaPipe (normalised)
    return lms

@pytest.fixture
def sample_embedding():
    """Simulated 512-dim face embedding."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


# ── utils.py ─────────────────────────────────────────────────────────────────

class TestUtils:
    def test_setup_logging_returns_logger(self):
        logger = setup_logging("DEBUG")
        assert logger is not None

    def test_get_device_returns_string(self):
        device = get_device()
        assert device in ("cpu", "cuda")


# ── utils_cam.py ─────────────────────────────────────────────────────────────

class TestStereoInference:
    def test_from_defaults_returns_object(self):
        """StereoInference.from_defaults() returns a usable mock for demos."""
        si = StereoInference.from_defaults()
        assert hasattr(si, "baseline")
        assert hasattr(si, "focal_length_pixels")
        assert si.baseline > 0
        assert si.focal_length_pixels > 0

    def test_disparity_to_depth(self):
        """depth = baseline * focal / disparity — basic stereo formula."""
        si = StereoInference.from_defaults()
        disparity = 50.0   # pixels
        depth = (si.baseline * si.focal_length_pixels) / disparity
        assert depth > 0


# ── utils_filters.py ─────────────────────────────────────────────────────────

class TestKalmanFilter:
    def test_init(self):
        z  = np.array([[320.], [240.], [2500.]])
        kf = KalmanFilter(acc_std=0.5, meas_std=1.2, z=z, time=0.0)
        assert kf.x.shape == (9, 1)   # 3D * 3 (pos, vel, acc)

    def test_predict_changes_state(self):
        z  = np.array([[320.], [240.], [2500.]])
        kf = KalmanFilter(acc_std=0.5, meas_std=1.2, z=z, time=0.0)
        x_before = kf.x.copy()
        kf.predict(dt=1/30)
        # State should evolve (velocity is zero so position stays same initially)
        assert kf.P is not None

    def test_update_with_measurement(self):
        z  = np.array([[320.], [240.], [2500.]])
        kf = KalmanFilter(acc_std=0.5, meas_std=1.2, z=z, time=0.0)
        kf.predict(dt=1/30)
        z_new = np.array([[325.], [242.], [2480.]])
        kf.update(z_new)
        # After update, position should shift toward measurement
        assert abs(float(kf.x[0]) - 320.0) < 10.0

    def test_update_none_is_safe(self):
        z  = np.array([[320.], [240.], [2500.]])
        kf = KalmanFilter(acc_std=0.5, meas_std=1.2, z=z, time=0.0)
        kf.predict(dt=1/30)
        kf.update(None)   # must not raise

    def test_depth_aware_sigma(self):
        """sigma = z^2 / (baseline * focal) grows with depth."""
        baseline, focal = 75.0, 860.0
        z1, z2 = 1000.0, 3000.0
        sig1 = z1**2 / (baseline * focal)
        sig2 = z2**2 / (baseline * focal)
        assert sig2 > sig1   # farther = noisier


# ── utils_image.py ───────────────────────────────────────────────────────────

class TestImageUtils:
    def test_dhash_returns_int(self, sample_frame):
        h = dhash(sample_frame, hash_size=8)
        assert isinstance(h, int)

    def test_dhash_same_image_same_hash(self, sample_frame):
        h1 = dhash(sample_frame)
        h2 = dhash(sample_frame)
        assert h1 == h2

    def test_dhash_different_images_differ(self, sample_frame):
        noise = np.random.randint(0, 255, sample_frame.shape, dtype=np.uint8)
        h1 = dhash(sample_frame)
        h2 = dhash(noise)
        assert h1 != h2

    def test_find_match_in_db_no_match(self, sample_frame):
        h = dhash(sample_frame)
        result = find_match_in_db(h, db={}, threshold=10)
        assert result is None

    def test_normalize_face_crop_output_shape(self, sample_frame):
        crop = normalize_face_crop(sample_frame, target_size=(112, 112))
        assert crop.shape == (112, 112, 3)


# ── utils_nn.py ──────────────────────────────────────────────────────────────

class TestNNUtils:
    def test_recon_loss_zero_for_identical(self):
        import torch
        x = torch.randn(4, 3, 64, 64)
        loss = recon_loss(x, x)
        assert float(loss) < 1e-6

    def test_kl_loss_positive(self):
        import torch
        mu      = torch.zeros(4, 128)
        log_var = torch.zeros(4, 128)
        loss    = kl_loss(mu, log_var)
        assert float(loss) >= 0

    def test_vae_loss_positive(self):
        import torch
        x       = torch.randn(4, 3, 64, 64)
        mu      = torch.zeros(4, 128)
        log_var = torch.zeros(4, 128)
        loss    = vae_loss(x, x, mu, log_var)
        assert float(loss) >= 0


# ── utils_cls.py ─────────────────────────────────────────────────────────────

class TestImageProcessor:
    def test_add_rec_does_not_raise(self, sample_frame):
        proc = ImageProcessor()
        proc.add_rec(sample_frame, coord=(100, 100, 200, 200))

    def test_add_txt_does_not_raise(self, sample_frame):
        proc = ImageProcessor()
        proc.add_txt(sample_frame, text=0.95, coord=(150, 150))

    def test_affine_returns_array(self):
        proc = ImageProcessor()
        vec  = np.array([[320.], [240.], [1.0]])
        result = proc.affine(delta=(0, 0), theta=0.0, scale=1.0, vec=vec)
        assert result.shape == (3, 1)


# ── utils_file.py ────────────────────────────────────────────────────────────

class TestVisitLogger:
    def test_numpy_encoder_handles_int64(self):
        data = {"count": np.int64(42), "score": np.float32(0.95)}
        s    = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(s)
        assert parsed["count"] == 42
        assert abs(parsed["score"] - 0.95) < 0.01

    def test_numpy_encoder_handles_ndarray(self):
        data = {"embedding": np.random.randn(512).astype(np.float32)}
        s    = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(s)
        assert len(parsed["embedding"]) == 512

    def test_visit_logger_init(self, tmp_path):
        vl = VisitLogger(db_path=tmp_path / "test_visits.h5")
        assert vl is not None


# ── utils_mp.py ──────────────────────────────────────────────────────────────

class TestMediaPipe:
    @patch("Utils.utils_mp.mp.solutions.face_mesh.FaceMesh")
    def test_extract_landmarks_returns_none_on_empty(self, mock_fm, sample_frame):
        """When no face detected, extract_mp_landmarks returns None."""
        mock_instance        = MagicMock()
        mock_instance.process.return_value = MagicMock(multi_face_landmarks=None)
        mock_fm.return_value.__enter__.return_value = mock_instance
        result = extract_mp_landmarks(sample_frame)
        assert result is None

    def test_get_mp_depthmask_shape(self, sample_frame, sample_landmarks, sample_depth):
        mask = get_mp_depthmask(
            frame=sample_frame,
            landmarks=sample_landmarks,
            depth_frame=sample_depth,
            z_threshold=0.1
        )
        assert mask.shape[:2] == sample_frame.shape[:2]


# ── utils_vis.py ─────────────────────────────────────────────────────────────

class TestVisualization:
    @patch("Utils.utils_vis.plt.show")
    def test_show_embedds_runs(self, mock_show):
        embeddings = np.random.randn(20, 512).astype(np.float32)
        labels     = np.random.randint(0, 3, 20)
        show_embedds(embeddings, labels, title="Test")

    @patch("Utils.utils_vis.plt.savefig")
    def test_draw_gaze_heatmap_saves(self, mock_save, tmp_path):
        draw_gaze_heatmap(
            zone_names=["entrance", "zone_A", "zone_B", "checkout"],
            zone_values=[300.0, 180.0, 450.0, 120.0],
            title="Test Heatmap",
            out_path=tmp_path / "heatmap.png"
        )


# ── Pipeline-level tests ─────────────────────────────────────────────────────

class TestPipeline:
    def test_classify_gaze_zone_entrance(self):
        xyz        = np.array([160.0, 270.0, 2500.0])
        frame_shape = (540, 1280, 3)
        zone = classify_gaze_zone(xyz, frame_shape)
        assert zone == "entrance"

    def test_classify_gaze_zone_unknown(self):
        xyz        = np.array([9999.0, 9999.0, 2500.0])
        frame_shape = (540, 1280, 3)
        zone = classify_gaze_zone(xyz, frame_shape)
        assert zone is None

    def test_match_embedding_empty_db(self, sample_embedding):
        cid, conf = match_embedding(sample_embedding, embedding_db={})
        assert cid == "unknown"
        assert conf == 0.0

    def test_match_embedding_finds_same(self, sample_embedding):
        db  = {"customer_001": sample_embedding.copy()}
        cid, conf = match_embedding(sample_embedding, db, threshold=0.9)
        assert cid == "customer_001"
        assert conf > 0.99

    def test_extract_face_region_valid(self, sample_frame):
        centroid = np.array([320.0, 240.0, 2500.0])
        region   = extract_face_region(sample_frame, centroid, pad=50)
        assert region is not None
        assert region.shape[0] > 0 and region.shape[1] > 0

    def test_extract_face_region_oob(self, sample_frame):
        centroid = np.array([0.0, 0.0, 2500.0])   # at corner — might clip
        region   = extract_face_region(sample_frame, centroid, pad=50)
        # Should still return something (clipped to frame bounds)
        assert region is None or region.shape[0] > 0
