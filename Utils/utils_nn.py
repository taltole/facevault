"""
FaceVault Utils — Neural Network helpers, embeddings, losses.

Provides:
  - extract_embedding()  CNN face embedding (ResNet50, 512-dim, L2-normalised)
  - load_backbone()      loads float32 or quantized int8 model
  - recon_loss()         MSE reconstruction loss (VAE)
  - kl_loss()            KL divergence regularisation (VAE)
  - vae_loss()           combined VAE loss
  - count_parameters()   human-readable param count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("facevault.nn")

_conf_threshold = 0.5

# Standard ImageNet normalisation — matches ResNet50 training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ── Backbone Loading ──────────────────────────────────────────────────────────

def load_backbone(model_name: str = "resnet50",
                  quantized: bool = False,
                  weights_path: Optional[str] = None) -> nn.Module:
    """
    Load face embedding backbone, optionally from a saved int8 checkpoint.

    Args:
        model_name:   'resnet50' | 'mobilenet_v3_small' | 'vgg16'
        quantized:    If True, tries to load int8 model from MODEL_PATH.
                      Falls back to float32 if not found.
        weights_path: Explicit path to .pt file; overrides auto-discovery.

    Returns:
        nn.Module in eval mode on CPU (int8 models don't run on CUDA).
    """
    # Try loading quantized model from disk
    if quantized and weights_path is None:
        from config.config import MODEL_PATH
        candidate = Path(MODEL_PATH) / f"{model_name}_int8_facevault.pt"
        if candidate.exists():
            weights_path = str(candidate)
            logger.info(f"Loading int8 model from {weights_path}")

    # Build architecture
    model = _build_embedding_model(model_name)

    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded weights from {weights_path}")
    else:
        if quantized:
            logger.warning("Int8 weights not found, using float32 pretrained")
        logger.info(f"Using pretrained {model_name} (float32)")

    model.eval()
    logger.info(f"Backbone: {model_name} — {count_parameters(model)} params")
    return model


def _build_embedding_model(backbone_name: str = "resnet50") -> nn.Module:
    """Replace classifier head with 512-dim embedding head."""
    if backbone_name == "resnet50":
        try:
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            base = models.resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
    elif backbone_name == "mobilenet_v3_small":
        try:
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            base = models.mobilenet_v3_small(weights=None)
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, 512)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return base


# ── Embedding Extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(model: nn.Module,
                      face_crop: np.ndarray) -> np.ndarray:
    """
    Extract L2-normalised 512-dim embedding from a face crop.

    Args:
        model:      Loaded backbone (output of load_backbone)
        face_crop:  BGR numpy array, any size

    Returns:
        np.ndarray shape (512,), unit-normalised.
    """
    if face_crop is None or face_crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    # RGB conversion + preprocessing
    rgb   = face_crop[:, :, ::-1]     # BGR -> RGB
    tensor = _preprocess(rgb).unsqueeze(0)  # (1, 3, 224, 224)

    out   = model(tensor)              # (1, 512)
    emb   = out.squeeze().numpy()      # (512,)

    # L2 normalise — cosine similarity becomes dot product
    norm  = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm

    return emb.astype(np.float32)


# ── VAE / GAN Losses (used in training) ──────────────────────────────────────

def recon_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss for VAE decoder."""
    return F.mse_loss(y_pred, y_true)


def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from N(mu, sigma^2) to N(0, 1).
    Used to regularise the VAE latent space.
    """
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def vae_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
    """
    Beta-VAE combined loss: reconstruction + beta * KL divergence.
    beta > 1 encourages more disentangled representations.
    """
    r_loss = recon_loss(y_true, y_pred)
    k_loss = kl_loss(mu, log_var)
    img_size = y_true.shape[-1] ** 2
    return r_loss + beta * (1.0 / img_size) * k_loss


def gan_loss(real_output: torch.Tensor,
             fake_output: torch.Tensor) -> tuple:
    """
    Non-saturating GAN losses (Goodfellow et al.).
    Returns (generator_loss, discriminator_loss).
    """
    real_loss = F.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output))

    d_loss = (real_loss + fake_loss) / 2
    g_loss = F.binary_cross_entropy_with_logits(
        fake_output, torch.ones_like(fake_output))

    return g_loss, d_loss


# ── Utilities ─────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> str:
    """Human-readable total trainable parameter count, e.g. '4.2M'."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1_000_000:
        return f"{total/1e6:.1f}M"
    if total >= 1_000:
        return f"{total/1e3:.1f}K"
    return str(total)


def num_flat_features(x: torch.Tensor) -> int:
    """Product of all dimensions except the batch dimension."""
    size = x.size()[1:]
    num  = 1
    for s in size:
        num *= s
    return num
