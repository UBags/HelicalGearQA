"""
display/visualiser.py
======================
Matplotlib-based display helpers.

These functions are used to render the three-panel figure that appears
in the notebook output:
  axes[0]  — "Processed Image"  (yellow/black enhanced)
  axes[1]  — "Final Mask"       (edge + contact + contour image)
  axes[2]  — "Image with arcs"  (annotated photo + dimensions table)

All functions accept a matplotlib ``Axes`` or ``Figure`` object and
return nothing (they draw in place).

``render_results`` is the single public entry point used by ``main.py``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Low-level helpers (preserved from original notebook)
# ---------------------------------------------------------------------------

def show_mask(mask: np.ndarray, ax, random_color: bool = False) -> None:
    """Overlay a semi-transparent coloured mask on a matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords: np.ndarray,
                 labels: np.ndarray,
                 ax,
                 marker_size: int = 375) -> None:
    """Scatter positive (green) and negative (red) SAM prompt points."""
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg[:, 0], neg[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box: np.ndarray, ax) -> None:
    """Draw a bounding box rectangle on a matplotlib axis."""
    x0, y0 = box[0], box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor='green',
                       facecolor=(0, 0, 0, 0), lw=2)
    )


def show_anns(anns: list, ax=None) -> None:
    """Render all SAM automatic-mask annotations with random colours."""
    if not anns:
        return
    if ax is None:
        ax = plt.gca()
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax.set_autoscale_on(False)
    img = np.ones((
        sorted_anns[0]['segmentation'].shape[0],
        sorted_anns[0]['segmentation'].shape[1],
        4
    ))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_results(enhanced_image:         np.ndarray,
                    edge_contact_image:     np.ndarray,
                    final_annotated_image:  np.ndarray,
                    filename:              str = "") -> None:
    """
    Render the standard three-panel figure.

    Parameters
    ----------
    enhanced_image : np.ndarray
        Yellow/black enhanced image for axes[0] ("Processed Image").
    edge_contact_image : np.ndarray
        Grayscale edge + contact + contour image for axes[1] ("Final Mask").
    final_annotated_image : np.ndarray
        Annotated photo + dimensions table for axes[2] ("Image with arcs").
    filename : str
        Optional image filename used as the figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))

    axes[0].imshow(enhanced_image)
    axes[0].set_title("Processed Image")
    axes[0].axis("off")

    axes[1].imshow(edge_contact_image, cmap="gray")
    axes[1].set_title("Final Mask")
    axes[1].axis("off")

    axes[2].imshow(final_annotated_image)
    axes[2].set_title("Image with arcs")
    axes[2].axis("off")

    if filename:
        fig.suptitle(filename, fontsize=10)

    plt.tight_layout()
    plt.show()