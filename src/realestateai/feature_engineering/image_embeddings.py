from __future__ import annotations

import io

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def clip_image_embeddings_from_urls(
    image_urls: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    timeout: int = 10,
    max_images: int | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    Download images by URLs and compute CLIP image embeddings.
    Returns: np.ndarray of shape [N, D] (L2-normalized).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    urls = image_urls[:max_images] if max_images else list(image_urls)

    # --- download ---
    images = []
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            images.append(img)
        except Exception:
            # skip broken urls
            continue

    if not images:
        return np.empty((0, model.config.projection_dim), dtype=np.float32)

    # --- embed ---
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)  # [B, D]
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # L2 norm
            all_embs.append(feats.detach().cpu().numpy().astype(np.float32))

    return np.vstack(all_embs)
