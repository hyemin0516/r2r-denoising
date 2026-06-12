from __future__ import annotations
from pathlib import Path
from typing import Dict
from PIL import Image

import torch
from torch.utils.data import Dataset

from ..utils.noise import get_noise_fns

class DIV2KPairs(Dataset):
    def __init__(
        self,
        img_dir: Path,
        transform,
        distribution: str,
        noise_level: float,
        exts=(".png", ".jpg", ".jpeg"),
    ):
        self.img_dir = Path(img_dir)
        self.paths = []
        for e in exts:
            self.paths += sorted(self.img_dir.glob(f"*{e}"))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.img_dir}")
        self.transform = transform
        self.add_noise, _ = get_noise_fns(distribution)
        self.noise_level = noise_level

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)  # [0,1], (3,H,W)
        y = self.add_noise(x, self.noise_level)

        return {"x": x, "y": y}