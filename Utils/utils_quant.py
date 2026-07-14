"""
FaceVault Utils — Quantisation-Aware Training (QAT) & Static PTQ.

Compresses the float32 face embedding backbone to int8 for edge deployment.

QAT schedule (from actual utils_quant.py by Tal Toledano):
  Epochs 0-2:  Full QAT with observers active
  Epoch  3:    Disable observers (lock quantisation ranges)
  Epoch  5:    Freeze BatchNorm statistics
  Epoch  15:   torch.quantization.convert() — real int8 ops

Why QAT over PTQ for faces:
  Face embedding activations are narrower and more sensitive than ImageNet.
  PTQ introduces 2-5% accuracy degradation. QAT keeps delta < 0.5%.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("facevault.quant")


# ── QAT ───────────────────────────────────────────────────────────────────────

def run_qat(model: nn.Module,
            train_loader,
            val_loader=None,
            epochs: int = 15,
            lr: float = 0.001,
            observer_off_epoch: int = 3,
            bn_freeze_epoch: int = 5,
            backend: str = "fbgemm") -> nn.Module:
    """
    Quantisation-Aware Training pipeline.

    Steps:
      1. Fuse Conv + BN + ReLU layers (required before QAT)
      2. Set QAT qconfig
      3. Insert fake-quantisation modules
      4. Train loop with scheduled observer/BN disabling
      5. Convert to real int8

    Args:
        model:              float32 PyTorch model
        train_loader:       DataLoader for training data
        val_loader:         Optional DataLoader for per-epoch validation
        epochs:             Total QAT training epochs
        lr:                 Learning rate (usually lower than original training)
        observer_off_epoch: Epoch to disable activation/weight observers
        bn_freeze_epoch:    Epoch to freeze BatchNorm running stats
        backend:            'fbgemm' (x86) or 'qnnpack' (ARM/mobile)

    Returns:
        Quantized model with real int8 operators.
    """
    logger.info(f"Starting QAT: {epochs} epochs, backend={backend}")

    torch.backends.quantized.engine = backend
    model.train()

    # 1. Fuse Conv + BN + ReLU — must happen before inserting fake-quant
    try:
        model.fuse_model()
        logger.info("Layer fusion: Conv+BN+ReLU fused")
    except AttributeError:
        logger.warning("model.fuse_model() not available — skipping fusion")

    # 2. QAT config
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # 3. Prepare for QAT (inserts fake-quantisation modules)
    model_prepared = torch.quantization.prepare_qat(model)

    optimizer = optim.SGD(model_prepared.parameters(), lr=lr,
                          momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── Observer & BN management ─────────────────────────────────────────
        if epoch == observer_off_epoch:
            model_prepared.apply(torch.quantization.disable_observer)
            logger.info(f"Epoch {epoch}: observers disabled — quantisation ranges locked")

        if epoch == bn_freeze_epoch:
            freeze_bn_stats(model_prepared)
            logger.info(f"Epoch {epoch}: BatchNorm statistics frozen")

        # ── Train ─────────────────────────────────────────────────────────────
        model_prepared.train()
        running_loss = 0.0
        n_batches    = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping — important for QAT stability
            torch.nn.utils.clip_grad_norm_(model_prepared.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # ── Validate ──────────────────────────────────────────────────────────
        val_acc = 0.0
        if val_loader is not None:
            val_acc = _quick_eval(model_prepared, val_loader)
            history["val_acc"].append(val_acc)

        scheduler.step()

        logger.info(f"QAT Epoch {epoch+1:02d}/{epochs} — "
                    f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.5f}")

    # 5. Convert to real int8 operators
    model_prepared.eval()
    int8_model = torch.quantization.convert(model_prepared)
    logger.info("Model converted to real int8 operators")

    return int8_model


# ── Static PTQ ────────────────────────────────────────────────────────────────

def run_ptq(model: nn.Module,
            calibration_loader,
            backend: str = "fbgemm",
            num_batches: int = 100) -> nn.Module:
    """
    Post-Training Quantisation with calibration data.

    Faster than QAT (no retraining), but ~2-5% accuracy drop on face embeddings.
    Use when training data is unavailable or time is limited.

    Args:
        model:               float32 model
        calibration_loader:  DataLoader with representative samples
        backend:             'fbgemm' or 'qnnpack'
        num_batches:         Number of calibration batches (100 is usually enough)
    """
    logger.info(f"Starting static PTQ: calibrating on {num_batches} batches")

    torch.backends.quantized.engine = backend
    model.eval()

    try:
        model.fuse_model()
    except AttributeError:
        pass

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)

    # Calibration — feed representative inputs to observe activation ranges
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            model(inputs)
            if i >= num_batches:
                break

    int8_model = torch.quantization.convert(model)
    logger.info("PTQ complete — model converted to int8")
    return int8_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def freeze_bn_stats(model: nn.Module) -> None:
    """
    Freeze BatchNorm running mean/variance so they don't update during training.
    Called at bn_freeze_epoch in QAT to improve quantisation stability.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            module.weight.requires_grad = True   # still update scale/bias
            module.bias.requires_grad   = True


def plot_activation_histograms(float_model: nn.Module,
                                int8_model: nn.Module,
                                sample_loader,
                                out_path: Optional[Path] = None,
                                layer_name: str = "layer1") -> None:
    """
    Plot activation distributions before and after quantisation.
    Shows how well the quantisation ranges were calibrated.

    Args:
        float_model:   Original float32 model
        int8_model:    Quantized int8 model
        sample_loader: DataLoader with sample inputs
        out_path:      Save path for the figure
        layer_name:    Which layer to hook into
    """
    float_acts, int8_acts = [], []

    def _hook_float(m, inp, out):
        float_acts.append(out.detach().cpu().numpy().flatten())

    def _hook_int8(m, inp, out):
        int8_acts.append(out.detach().cpu().numpy().flatten())

    # Hook into a comparable layer in both models
    fh = None
    ih = None
    for name, module in float_model.named_modules():
        if layer_name in name and isinstance(module, nn.ReLU):
            fh = module.register_forward_hook(_hook_float)
            break
    for name, module in int8_model.named_modules():
        if layer_name in name:
            ih = module.register_forward_hook(_hook_int8)
            break

    with torch.no_grad():
        for i, (inputs, _) in enumerate(sample_loader):
            float_model(inputs)
            int8_model(inputs)
            if i >= 10:
                break

    if fh: fh.remove()
    if ih: ih.remove()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if float_acts:
        all_float = np.concatenate(float_acts)
        axes[0].hist(all_float, bins=100, color="steelblue", alpha=0.8)
        axes[0].set_title(f"Float32 activations ({layer_name})")
        axes[0].set_xlabel("Activation value")
        axes[0].set_ylabel("Count")

    if int8_acts:
        all_int8 = np.concatenate(int8_acts)
        axes[1].hist(all_int8, bins=100, color="coral", alpha=0.8)
        axes[1].set_title(f"Int8 activations ({layer_name})")
        axes[1].set_xlabel("Activation value")

    fig.suptitle("Activation Histograms: Float32 vs Int8", fontsize=13)
    plt.tight_layout()

    if out_path:
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info(f"Activation histogram saved to {out_path}")
        plt.close()
    else:
        plt.show()


def _quick_eval(model: nn.Module, dataloader) -> float:
    """Quick accuracy pass — used per-epoch during QAT."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            out   = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    model.train()
    return correct / max(total, 1)
