"""
FaceVault Utils — Visualisation: t-SNE, heatmaps, subplot grids, match viewer.

Based on actual utils_vis.py by Tal Toledano.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path
from typing import Optional

logger = logging.getLogger("facevault.vis")


def show_embedds(embeddings: np.ndarray,
                  labels: Optional[np.ndarray] = None,
                  title: str = "t-SNE Embedding Space",
                  out_path: Optional[Path] = None) -> None:
    """
    Run t-SNE on high-dim embeddings and produce a 2D scatter plot.
    Each point = one customer; colour = visit frequency (labels).

    Used in analytics/report.py to visualise customer segments.
    """
    from sklearn.manifold import TSNE

    n = len(embeddings)
    perplexity = min(30, max(5, n // 3))

    logger.info(f"Running t-SNE on {n} embeddings (perplexity={perplexity})")
    tsne    = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                   max_iter=1000, init="pca")
    coords  = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap="plasma",
            s=60, alpha=0.8, edgecolors="white", linewidths=0.5
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Visit frequency", fontsize=11)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=60, alpha=0.8)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info(f"t-SNE plot saved to {out_path}")
        plt.close()
    else:
        plt.show()


def draw_gaze_heatmap(zone_names: list,
                       zone_values: list,
                       title: str = "Zone Attention Heatmap",
                       out_path: Optional[Path] = None) -> None:
    """
    Bar heatmap of customer attention seconds per store zone.
    Colour encodes attention intensity.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    vals    = np.array(zone_values, dtype=float)
    norm    = vals / (vals.max() + 1e-8)
    colours = cm.YlOrRd(norm)

    bars = ax.bar(zone_names, vals, color=colours, edgecolor="white", linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Store Zone", fontsize=12)
    ax.set_ylabel("Total Dwell Time (seconds)", fontsize=12)
    ax.set_ylim(0, vals.max() * 1.15 + 1)
    ax.grid(axis="y", alpha=0.3)

    # Colourbar
    sm = cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, vals.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8, label="Relative attention")

    plt.tight_layout()

    if out_path:
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info(f"Zone heatmap saved to {out_path}")
        plt.close()
    else:
        plt.show()


def show_match(img_paths: list, title: str = "Match Result") -> None:
    """Side-by-side display of two matched face crops."""
    if len(img_paths) < 2:
        logger.warning("show_match needs at least 2 images")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, path in zip(axes, img_paths[:2]):
        if isinstance(path, str):
            img = cv2.imread(path)[:, :, ::-1]
        else:
            img = path[:, :, ::-1] if path.shape[2] == 3 else path
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def show_subframes(frames: list, titles: list = None, cols: int = 4) -> None:
    """Grid display of multiple frames — used for debugging pipeline stages."""
    n    = len(frames)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).flatten()

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        if frame.ndim == 2:
            ax.imshow(frame, cmap="gray")
        else:
            ax.imshow(frame[:, :, ::-1])
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=9)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def imshow(image: np.ndarray, title: str = "Image",
            wait: int = 0) -> None:
    """Quick OpenCV display for debugging."""
    cv2.imshow(title, image)
    cv2.waitKey(wait)
