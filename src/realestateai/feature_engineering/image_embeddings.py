from __future__ import annotations

import io

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPImageEmbedder:
    """
    Reusable CLIP image embedder (image-only).

    - Loads model once
    - embed_urls() returns L2-normalized np.ndarray [N, D]
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        use_fast_processor: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        self.model = (
            CLIPModel.from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_fast=use_fast_processor,
        )

        self.embedding_dim = self.model.config.projection_dim

    @staticmethod
    def _download_images(
        image_urls: list[str],
        timeout: int,
        max_images: int | None,
    ) -> list[Image.Image]:
        urls = image_urls[:max_images] if max_images else list(image_urls)
        images: list[Image.Image] = []

        for url in urls:
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                images.append(img)
            except Exception:
                continue

        return images

    def _image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Compute image embeddings tensor [B, D] without requiring text inputs.
        """
        # vision_model returns BaseModelOutputWithPooling; we need pooler_output
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output  # [B, hidden]
        feats = self.model.visual_projection(pooled)  # [B, D]
        return feats

    def embed_urls(
        self,
        image_urls: list[str],
        batch_size: int = 16,
        timeout: int = 10,
        max_images: int | None = None,
    ) -> np.ndarray:
        images = self._download_images(image_urls, timeout=timeout, max_images=max_images)

        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embs: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i : i + batch_size]

                inputs = self.processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                print(pixel_values.shape)

                feats = self._image_features(pixel_values)  # Tensor [B, D]
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)

                all_embs.append(feats.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_embs)
    
    



if __name__ == "__main__":
    from realestateai.data.postgres.utils import query_to_dataframe

    df = query_to_dataframe(
        """SELECT ad_id, images_small FROM extracted_payload"""
    )

    embedder = CLIPImageEmbedder(device="cuda")  # или не указывать

    embeddings = embedder.embed_urls(df.iloc[0]["images_small"])

    print(embeddings.shape)
    print(embeddings[0])
