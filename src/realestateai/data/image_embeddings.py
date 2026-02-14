from __future__ import annotations

from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)


def encode_images_and_captions(
    image_urls: list[str],
    *,
    # Fast baseline models
    embed_model_name: str = "google/siglip-base-patch16-224",
    caption_model_name: str = "Salesforce/blip-image-captioning-base",
    # Runtime
    batch_size: int = 8,
    timeout_s: int = 12,
    max_caption_tokens: int = 40,
    device: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Download images by URLs, produce:
      1) image embeddings (SigLIP) as numpy array [N, D]
      2) captions (BLIP) as list[str] length N

    Notes:
      - If an image fails to load, returns zero-vector embedding and empty caption for that slot.
      - Embeddings are L2-normalized (good for cosine similarity + pgvector).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Load models ----------
    # SigLIP encoder
    embed_processor = AutoProcessor.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
    embed_model.eval()

    # BLIP captioner (fast baseline)
    cap_processor = BlipProcessor.from_pretrained(caption_model_name)
    cap_model = BlipForConditionalGeneration.from_pretrained(caption_model_name).to(device)
    cap_model.eval()

    # ---------- Helpers ----------
    def _load_image(url: str) -> Image.Image | None:
        try:
            r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "realestate-ai/0.1"})
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            return None

    # Pre-download all images (keeps alignment with input urls)
    images: list[Image.Image | None] = [_load_image(u) for u in image_urls]
    n = len(image_urls)

    # We’ll fill outputs aligned with input order
    captions: list[str] = [""] * n

    # We need embedding dimension to allocate output array.
    # Quick way: run 1 dummy forward on any valid image; if none, we fallback to model config.
    embedding_dim: int
    any_img = next((im for im in images if im is not None), None)
    if any_img is not None:
        with torch.no_grad():
            tmp = embed_processor(images=any_img, return_tensors="pt").to(device)
            out = embed_model(**tmp)
            embedding_dim = out.image_embeds.shape[-1]
    else:
        # best-effort fallback
        embedding_dim = getattr(getattr(embed_model, "config", None), "projection_dim", 768) or 768

    embeddings = np.zeros((n, embedding_dim), dtype=np.float32)

    # ---------- Batched inference ----------
    def _iter_batches(idxs: list[int], bs: int):
        for i in range(0, len(idxs), bs):
            yield idxs[i : i + bs]

    valid_idxs = [i for i, im in enumerate(images) if im is not None]

    with torch.no_grad():
        # 1) Embeddings (SigLIP)
        for b in _iter_batches(valid_idxs, batch_size):
            batch_imgs = [images[i] for i in b]  # type: ignore[list-item]
            inputs = embed_processor(images=batch_imgs, return_tensors="pt").to(device)
            out = embed_model(**inputs)
            img_emb = out.image_embeds  # [B, D]
            img_emb = F.normalize(img_emb, dim=-1)
            embeddings[b, :] = img_emb.detach().cpu().numpy().astype(np.float32)

        # 2) Captions (BLIP)
        # BLIP captioning is a bit slower; keep batch_size smaller if needed.
        for b in _iter_batches(valid_idxs, max(1, min(batch_size, 4))):
            batch_imgs = [images[i] for i in b]  # type: ignore[list-item]
            cap_inputs = cap_processor(images=batch_imgs, return_tensors="pt").to(device)
            generated = cap_model.generate(
                **cap_inputs,
                max_new_tokens=max_caption_tokens,
                do_sample=False,
            )
            batch_caps = cap_processor.batch_decode(generated, skip_special_tokens=True)
            for idx, cap in zip(b, batch_caps):
                captions[idx] = cap.strip()

    return embeddings, captions


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    urls = [
        "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",
        "https://images.unsplash.com/photo-1501183638710-841dd1904471",
    ]
    emb, caps = encode_images_and_captions(urls, batch_size=8)
    print("Embeddings shape:", emb.shape)
    print("Captions:")
    for u, c in zip(urls, caps):
        print("-", c)
