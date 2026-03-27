"""
model/inferencer.py
====================
Low-level SAM inference helpers.  Every raw call to the model tensor
operations lives here, so that the segmentation modules stay free of
PyTorch/HuggingFace boilerplate.

Functions
---------
run_prompted_inference
    Run a single SAM forward pass given an image and a set of prompt
    points.  Returns raw predicted masks and IOU scores.

post_process_masks
    Convert raw SAM output tensors to numpy boolean arrays at the
    original image resolution.

get_image_embeddings
    Compute the SAM image embedding for an image (needed when the same
    image is re-used across multiple prompted calls).

run_prompted_inference_with_embeddings
    Like ``run_prompted_inference`` but reuses pre-computed embeddings,
    which avoids the expensive vision-encoder pass on every call.
"""

from __future__ import annotations

import numpy as np
import torch

from model.loader import SamComponents


# ---------------------------------------------------------------------------
# Image embedding
# ---------------------------------------------------------------------------

def get_image_embeddings(image_rgb: np.ndarray,
                          sam: SamComponents) -> torch.Tensor:
    """
    Compute the SAM ViT image embedding for *image_rgb*.

    This is the computationally expensive step.  Pre-computing it once
    and reusing it for multiple prompted calls (via
    ``run_prompted_inference_with_embeddings``) significantly reduces
    total inference time.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    sam : SamComponents
        Loaded SAM model components (from ``model.loader``).

    Returns
    -------
    torch.Tensor
        Image embedding tensor on the same device as the model.
    """
    inputs = sam.processor(image_rgb, return_tensors="pt").to(device=sam.device)
    image_embeddings = sam.model.get_image_embeddings(inputs["pixel_values"])
    return image_embeddings


# ---------------------------------------------------------------------------
# Prompted inference (fresh embedding)
# ---------------------------------------------------------------------------

def run_prompted_inference(image_rgb: np.ndarray,
                            input_points: list,
                            sam: SamComponents) -> tuple:
    """
    Run a full SAM forward pass (embedding + decoder) for *image_rgb*
    with *input_points* as prompts.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.
    input_points : list
        Prompt points in SAM format, e.g.
        ``[ [[x1,y1], [x2,y2], [x3,y3]] ]``.
    sam : SamComponents
        Loaded SAM model components.

    Returns
    -------
    masks_list : list
        Nested list of mask tensors as returned by
        ``processor.image_processor.post_process_masks``.
    scores : torch.Tensor
        IOU scores, shape (1, n_predictions, n_masks_per_prediction).
    """
    # Build embedding
    inputs_embed = sam.processor(image_rgb, return_tensors="pt").to(device=sam.device)
    image_embeddings = sam.model.get_image_embeddings(inputs_embed["pixel_values"])

    # Build prompted inputs (no pixel_values needed — use pre-computed embedding)
    inputs = sam.processor(
        image_rgb, input_points=input_points, return_tensors="pt"
    ).to(device=sam.device)
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = sam.model(**inputs)

    masks_list = sam.processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    return masks_list, scores


# ---------------------------------------------------------------------------
# Prompted inference (reused embedding)
# ---------------------------------------------------------------------------

def run_prompted_inference_with_embeddings(image_rgb: np.ndarray,
                                            input_points: list,
                                            image_embeddings: torch.Tensor,
                                            sam: SamComponents) -> tuple:
    """
    Run the SAM decoder with pre-computed *image_embeddings*.

    Skips the ViT encoder pass, making this much faster when the same
    image is used with different prompt sets.

    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image used only to build the processor inputs for shape
        information (pixel_values are discarded).
    input_points : list
        Prompt points in SAM format.
    image_embeddings : torch.Tensor
        Pre-computed embedding from ``get_image_embeddings``.
    sam : SamComponents
        Loaded SAM model components.

    Returns
    -------
    masks_list : list
        Nested list of post-processed mask tensors.
    scores : torch.Tensor
        IOU scores tensor.
    """
    inputs = sam.processor(
        image_rgb, input_points=input_points, return_tensors="pt"
    ).to(device=sam.device)
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = sam.model(**inputs)

    masks_list = sam.processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    return masks_list, scores


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def post_process_masks(masks_list: list,
                        scores: torch.Tensor,
                        area_limits: list | None = None) -> list[np.ndarray]:
    """
    Flatten the nested SAM mask output into a plain list of numpy boolean
    arrays, optionally filtering by area.

    SAM returns masks in a 3-level structure:
    ``masks_list[0][prediction_idx][mask_variant]``

    This function iterates over all predictions and, for each, takes the
    first mask variant whose pixel-count falls within one of the
    *area_limits* ranges (if provided).

    Parameters
    ----------
    masks_list : list
        Output of ``processor.image_processor.post_process_masks``.
    scores : torch.Tensor
        IOU scores, shape (1, n_predictions, n_variants).
    area_limits : list of [float, float] or None
        Each entry is [min_area, max_area].  If provided, a mask is only
        accepted if its pixel count falls within at least one range.
        If None, the highest-scoring variant per prediction is used.

    Returns
    -------
    list of np.ndarray
        Each element is a boolean (or uint8) numpy mask, shape (H, W).
    """
    _, n_predictions, _ = scores.shape
    final_masks: list[np.ndarray] = []

    for i in range(n_predictions):
        found = False
        for mask_tensor in masks_list[0][i]:
            if found:
                break
            mask_np = mask_tensor.cpu().detach().numpy()
            current_area = float(np.sum(mask_np))

            if area_limits is not None:
                area_ok = any(
                    lo <= current_area <= hi
                    for lo, hi in area_limits
                )
                if area_ok:
                    final_masks.append(mask_np)
                    found = True
            else:
                # No area filter: take first variant
                final_masks.append(mask_np)
                found = True

    return final_masks