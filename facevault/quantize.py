"""
FaceVault — Model Quantisation for Edge Deployment
====================================================
Compresses the face embedding backbone (ResNet50) to int8
using Quantisation-Aware Training (QAT) so FaceVault can run
at real-time speeds on the in-store edge device (no server needed).

Faithfully wraps utils_quant.py patterns:
  - Conv + BN + ReLU fusion
  - 15-epoch QAT loop
  - Observer disable at epoch 3
  - BatchNorm freeze at epoch 5
  - torch.quantization.convert() to real int8

Usage:
  python -m facevault.quantize --epochs 15 --backbone resnet50
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path

from config.config import (
    QAT_EPOCHS, QAT_LR, QAT_OBSERVER_OFF, QAT_BN_FREEZE,
    QUANT_BACKEND, MODEL_PATH
)
from Utils.utils_quant import (
    run_qat, run_ptq, freeze_bn_stats, plot_activation_histograms
)

logger = logging.getLogger("facevault.quantize")


def get_face_dataloader(data_root, batch_size=32, img_size=224, split="train"):
    """
    Builds a DataLoader for the face embedding training set.
    Expects ImageFolder structure:
      data_root/
        customer_001/  img1.jpg img2.jpg ...
        customer_002/  ...
    """
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset    = ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"),
                            num_workers=4, pin_memory=True)
    logger.info(f"DataLoader: {len(dataset)} images, {len(dataset.classes)} classes")
    return dataloader


def build_embedding_model(backbone_name="resnet50", num_classes=1000):
    """
    Loads a pretrained backbone and modifies the final layer to output
    EMBEDDING_DIM-dimensional face embeddings (L2-normalised).
    """
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features

    # Replace classifier with embedding head
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),       # 512-dim face embedding
    )
    return model


def quantize_for_edge(data_root, backbone_name="resnet50", use_qat=True):
    """
    Full pipeline:
      1. Load pretrained backbone with embedding head
      2. Run QAT (or PTQ) using utils_quant
      3. Save quantized model to MODEL_PATH
      4. Compare float32 vs int8 accuracy

    Returns path to saved int8 model.
    """
    logger.info(f"Starting {'QAT' if use_qat else 'PTQ'} for {backbone_name}")

    # 1. Model
    model      = build_embedding_model(backbone_name)
    train_ldr  = get_face_dataloader(data_root, split="train")
    val_ldr    = get_face_dataloader(data_root, split="val")

    # 2. Baseline float32 accuracy
    float_acc = evaluate(model, val_ldr)
    logger.info(f"Float32 accuracy: {float_acc:.4f}")

    # 3. Quantise
    if use_qat:
        int8_model = run_qat(
            model=model,
            train_loader=train_ldr,
            val_loader=val_ldr,
            epochs=QAT_EPOCHS,
            lr=QAT_LR,
            observer_off_epoch=QAT_OBSERVER_OFF,
            bn_freeze_epoch=QAT_BN_FREEZE,
            backend=QUANT_BACKEND,
        )
    else:
        # Static PTQ — calibrate on representative data, no retraining
        int8_model = run_ptq(
            model=model,
            calibration_loader=train_ldr,
            backend=QUANT_BACKEND,
        )

    # 4. Post-quantisation accuracy
    int8_acc = evaluate(int8_model, val_ldr)
    logger.info(f"Int8 accuracy:    {int8_acc:.4f}")
    logger.info(f"Accuracy delta:   {(float_acc - int8_acc)*100:.2f}%")

    # 5. Plot activation histograms (before vs after) using utils_quant
    plot_activation_histograms(
        float_model=model,
        int8_model=int8_model,
        sample_loader=val_ldr,
        out_path=MODEL_PATH / "activation_hist.png"
    )

    # 6. Save
    out_path = MODEL_PATH / f"{backbone_name}_int8_facevault.pt"
    torch.save(int8_model.state_dict(), out_path)
    logger.info(f"Saved int8 model to {out_path}")

    # 7. Model size comparison
    float_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    int8_mb  = Path(out_path).stat().st_size / 1e6
    logger.info(f"Float32 model size: {float_mb:.1f} MB")
    logger.info(f"Int8 model size:    {int8_mb:.1f} MB  ({float_mb/int8_mb:.1f}x compression)")

    return out_path


def evaluate(model, dataloader, device=None):
    """Quick accuracy evaluation loop."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / max(total, 1)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceVault model quantisation")
    parser.add_argument("--data",     type=str, required=True, help="Path to face image dataset (ImageFolder format)")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--epochs",   type=int, default=QAT_EPOCHS)
    parser.add_argument("--ptq",      action="store_true", help="Use static PTQ instead of QAT")
    args = parser.parse_args()

    out = quantize_for_edge(
        data_root=args.data,
        backbone_name=args.backbone,
        use_qat=not args.ptq,
    )
    print(f"\nDone. Int8 model saved to: {out}")
