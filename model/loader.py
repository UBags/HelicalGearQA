"""
model/loader.py
================
Responsible for loading the Facebook SAM model, its processor, config,
image processor, and the HuggingFace mask-generation pipeline — exactly
once per run.

All paths and model settings come from ``config.settings``.  The module
exposes a single public function, ``load_sam_components``, which returns
a ``SamComponents`` dataclass holding every object that the rest of the
pipeline needs.

Usage
-----
    from model.loader import load_sam_components
    sam = load_sam_components()
    # sam.model, sam.processor, sam.generator, sam.device are now ready
"""

import os
from dataclasses import dataclass
from datetime import datetime

import torch
from transformers import (
    SamModel,
    SamConfig,
    SamProcessor,
    SamImageProcessor,
    pipeline,
)

from config.settings import FACEBOOK_MODELS_DIRECTORY, FACEBOOK_MODEL_SUBFOLDER


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SamComponents:
    """
    Groups all SAM-related objects so they can be passed as a single
    argument through the pipeline.

    Attributes
    ----------
    model : SamModel
        The loaded SAM model (on the correct device).
    processor : SamProcessor
        The SAM input/output processor.
    config : SamConfig
        SAM configuration object.
    image_processor : SamImageProcessor
        The SAM image pre-processor.
    generator : pipeline
        HuggingFace mask-generation pipeline (used for prompt-free inference).
    device : torch.device
        The device the model is loaded onto (cuda:0 or cpu).
    model_home_path : str
        The filesystem path from which the model was loaded.
    """
    model:           SamModel
    processor:       SamProcessor
    config:          SamConfig
    image_processor: SamImageProcessor
    generator:       object          # transformers pipeline object
    device:          torch.device
    model_home_path: str


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_sam_components() -> SamComponents:
    """
    Load the SAM model and all associated components from the local
    filesystem path defined in ``config.settings``.

    The function:
      1. Detects whether a CUDA GPU is available and selects the device.
      2. Loads ``SamModel`` from ``FACEBOOK_MODELS_DIRECTORY / FACEBOOK_MODEL_SUBFOLDER``.
      3. Loads ``SamProcessor``, ``SamConfig``, and ``SamImageProcessor``
         from the same path.
      4. Constructs a HuggingFace ``mask-generation`` pipeline wrapping all
         the above.
      5. Returns a ``SamComponents`` dataclass with every loaded object.

    Returns
    -------
    SamComponents
        All model components, ready for inference.

    Raises
    ------
    OSError
        If the model directory does not exist or is missing required files.
    """
    model_home_path = os.path.join(FACEBOOK_MODELS_DIRECTORY, FACEBOOK_MODEL_SUBFOLDER)

    if not os.path.isdir(model_home_path):
        raise OSError(
            f"SAM model directory not found: '{model_home_path}'.\n"
            "Check FACEBOOK_MODELS_DIRECTORY and FACEBOOK_MODEL_SUBFOLDER "
            "in config/settings.py."
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[loader] Using device: {device}")

    # --- Load model ---
    t0 = datetime.now()
    print(f"[loader] Loading SamModel from '{model_home_path}' …")
    current_model = SamModel.from_pretrained(model_home_path).to(device=device)
    print(f"[loader] SamModel loaded in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Load processor, config, image processor ---
    t0 = datetime.now()
    print(f"[loader] Loading SamProcessor / SamConfig / SamImageProcessor …")
    current_processor       = SamProcessor.from_pretrained(model_home_path)
    current_config          = SamConfig.from_pretrained(model_home_path)
    current_image_processor = SamImageProcessor(model_home_path)
    print(f"[loader] Processor components loaded in {(datetime.now() - t0).total_seconds():.1f}s")

    # --- Build mask-generation pipeline ---
    t0 = datetime.now()
    print(f"[loader] Building mask-generation pipeline …")
    mask_generator = pipeline(
        "mask-generation",
        model=current_model,
        config=current_config,
        image_processor=current_image_processor,
        device=device,
    )
    print(f"[loader] Pipeline ready in {(datetime.now() - t0).total_seconds():.1f}s")

    return SamComponents(
        model=current_model,
        processor=current_processor,
        config=current_config,
        image_processor=current_image_processor,
        generator=mask_generator,
        device=device,
        model_home_path=model_home_path,
    )