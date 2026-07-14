"""
FaceVault Utils — Image Processing, Perceptual Hashing, Face Crop Normalisation.

Core utilities for the identity pre-filter pipeline:
  1. dhash()           — perceptual hash (1000x faster than CNN)
  2. find_match_in_db()— hamming-distance search in hash database
  3. normalize_face_crop() — affine normalisation before embedding
  4. resize_pad()      — aspect-ratio-safe resize with padding
"""

import cv2
import numpy as np
import logging
from PIL import Image
from pathlib import Path
from typing import Optional

logger = logging.getLogger("facevault.image")

# ── Constants ─────────────────────────────────────────────────────────────────
B, G, R = (255, 0, 0), (0, 255, 0), (0, 0, 255)
WHITE   = (255, 255, 255)
TARGET_SHAPE = (224, 224)


# ── Perceptual Hashing ────────────────────────────────────────────────────────

def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    Difference hash (dhash) — fast perceptual fingerprint for near-duplicate detection.

    Algorithm:
      1. Resize to (hash_size+1, hash_size) — one extra column for horizontal diff
      2. Convert to greyscale
      3. Compare adjacent pixels horizontally — produces hash_size*hash_size bits
      4. Pack into integer

    Hamming distance between two hashes:
      - 0   = identical
      - <10 = very similar (same person, different lighting)
      - >20 = different person

    Complexity: O(1) — ~0.3ms per image vs ~28ms for CNN embedding.
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image

    img = img.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.array(img, dtype=np.int16)

    # Horizontal difference — True where left > right
    diff   = pixels[:, :-1] > pixels[:, 1:]
    bits   = diff.flatten()

    # Pack 64 bits into a single integer
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def hamming_distance(h1: int, h2: int) -> int:
    """Number of differing bits between two dhash values."""
    return bin(h1 ^ h2).count("1")


def find_match_in_db(query_hash: int, db: dict,
                     threshold: int = 10) -> Optional[str]:
    """
    Linear scan of hash database to find the closest matching customer.

    Args:
        query_hash: dhash of the current face crop
        db:         dict of {customer_id: dhash_value}
        threshold:  max Hamming distance to count as a match

    Returns:
        customer_id string if match found, else None.

    Note: For >10K customers, replace with a vantage-point tree for O(log n) search.
    """
    best_id, best_dist = None, threshold + 1

    for cid, stored_hash in db.items():
        dist = hamming_distance(query_hash, stored_hash)
        if dist < best_dist:
            best_dist = dist
            best_id   = cid

    if best_id is not None and best_dist <= threshold:
        logger.debug(f"Hash match: {best_id} (hamming={best_dist})")
        return best_id

    return None


# ── Face Crop Normalisation ───────────────────────────────────────────────────

def normalize_face_crop(face: np.ndarray,
                         target_size: tuple = (112, 112)) -> np.ndarray:
    """
    Prepare a face crop for CNN embedding:
      1. Convert BGR -> RGB
      2. Resize with padding to preserve aspect ratio
      3. Normalise pixel values to [-1, 1]

    Returns float32 numpy array (H, W, 3).
    """
    if face is None or face.size == 0:
        return np.zeros((*target_size, 3), dtype=np.float32)

    # Resize with padding
    padded = resize_pad(face, target_size)

    # BGR -> RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Normalise to [-1, 1] (matches ResNet50 preprocessing)
    norm = (rgb.astype(np.float32) / 127.5) - 1.0
    return norm


def resize_pad(image: np.ndarray, target: tuple) -> np.ndarray:
    """
    Resize an image to target (H, W) with letterbox padding.
    Preserves aspect ratio — no squashing.
    """
    th, tw = target
    h, w   = image.shape[:2]

    scale  = min(tw / w, th / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre on black canvas
    canvas  = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_y   = (th - new_h) // 2
    pad_x   = (tw - new_w) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas


# ── Duplicate Detection ───────────────────────────────────────────────────────

def find_duplicates(image_paths: list, threshold: int = 5) -> list:
    """
    Find near-duplicate images in a list using dhash.

    Returns list of (path_a, path_b, hamming_dist) tuples.
    O(n^2) — suitable for up to ~10K images.
    """
    hashes = []
    for p in image_paths:
        try:
            img  = cv2.imread(str(p))
            if img is not None:
                hashes.append((p, dhash(img)))
        except Exception as e:
            logger.warning(f"Could not hash {p}: {e}")

    duplicates = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming_distance(hashes[i][1], hashes[j][1])
            if dist <= threshold:
                duplicates.append((hashes[i][0], hashes[j][0], dist))
                logger.debug(f"Duplicate pair: {hashes[i][0]} <-> {hashes[j][0]} dist={dist}")

    logger.info(f"Found {len(duplicates)} duplicate pairs among {len(hashes)} images")
    return duplicates


# ── Misc helpers ─────────────────────────────────────────────────────────────

def load_jxr(path: str) -> Optional[np.ndarray]:
    """Load a Windows JPEG XR (.jxr) encoded image via imagecodecs."""
    try:
        import imagecodecs
        data = open(path, "rb").read()
        arr  = imagecodecs.jxr_decode(data)
        return arr
    except Exception as e:
        logger.warning(f"JXR load failed for {path}: {e}")
        return None
