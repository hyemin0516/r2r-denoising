from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .datasets.div2k import DIV2KPairs
from .models.autoencoder import Autoencoder
from .models.dncnn import DnCNN
from .models.drunet import DRUNet
from .models.unet import UNet
from .utils.noise import get_noise_fns
from .utils.transform import build_transforms


def load_init_weights(model: torch.nn.Module, ckpt_path: str, strict: bool = True) -> None:
    """
    Load initial weights ONLY (no optimizer/scheduler/scaler).
    Supports:
      - plain state_dict (.pth/.pt)  (keys match model)
      - lightning checkpoint (.ckpt) (state_dict keys like "model.xxx" or "xxx")
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # case A) lightning ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    # case B) common wrappers
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # strip common prefixes
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        elif k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if len(missing) or len(unexpected):
        print(f"[init_weights] missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing):
            print("  missing keys (first 20):", missing[:20])
        if len(unexpected):
            print("  unexpected keys (first 20):", unexpected[:20])


def build_model(model_name: str) -> torch.nn.Module:
    m = model_name.lower()
    if m == "autoencoder":
        return Autoencoder(in_ch=3, base=32)
    if m == "dncnn":
        # DnCNN: input/output same channels
        return DnCNN(in_channels=3, out_channels=3, depth=20, nf=64, bias=True)
    if m == "drunet":
        # DRUNet expects sigma channel inside; we pass sigma via forward(x, sigma=...)
        return DRUNet(in_channels=3, out_channels=3)
    if m == "unet":
        return UNet(in_channels=3, out_channels=3)
    raise ValueError(f"Unknown model: {model_name} (choose from autoencoder|dncnn|drunet)")


class GR2RLightning(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        distribution: str,
        noise_level: float,
        alpha: float,
        lr: float,
        mc_samples: int,
        sigma_mode: str = "noise_level",   # how to set sigma for DRUNet: noise_level or fixed
        sigma_value: float = 0.1,          # used when sigma_mode="fixed"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(model_name=model_name)

        # GR2R corruptor depends on distribution
        _, self.corruptor = get_noise_fns(distribution)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))
    
    def forward(self, x):
        return self.model(x)

    def _maybe_forward(self, x: torch.Tensor):
        if self.hparams.model_name.lower() != "drunet":
            return self.model(x)
    
        if self.hparams.sigma_mode == "fixed":
            sigma_val = float(self.hparams.sigma_value)
        else:
            sigma_val = float(self.hparams.noise_level)
    
        sigma = torch.full((x.size(0),), sigma_val, device=x.device, dtype=x.dtype)  # (B,)
        return self.model(x, sigma=sigma)
        # return self.model(x)

    def r2r_loss(self, y: torch.Tensor) -> torch.Tensor:
        alpha = float(self.hparams.alpha)
        nl = float(self.hparams.noise_level)

        y1 = self.corruptor(y, alpha, nl)
        y2 = (1.0 / alpha) * (y - y1 * (1.0 - alpha))

        pred = self._maybe_forward(y1.clamp(0,1))
        loss = torch.nn.functional.mse_loss(pred, y2)

        return loss, y2, pred

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        
        pred = self.forward(y)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]

        pred = self.forward(y)
        pred = torch.clamp(pred, 0.0, 1.0)
        
        self.log("val_psnr", self.psnr(pred, x), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ssim", self.ssim(pred, x), prog_bar=False, on_step=False, on_epoch=True)

def build_run_name(args):
    # 너무 길어지는 거 방지하면서, 실험 구분에 필요한 것만 핵심적으로 넣음
    # 예: dncnn_gaussian_nl0.10_a0.20_img256_bs8_ep30_mc5
    return (
        f"{args.model}_"
        f"{args.distribution}_"
        f"nl{args.noise_level:.2f}_"
        f"a{args.alpha:.2f}_"
        f"img{args.img_size}_"
        f"bs{args.batch_size}_"
        f"ep{args.epochs}_"
        f"mc{args.mc_samples}"
    )

def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--div2k_root", type=str, default="/workspace02/hyemin/nas/datasets/DIV2K") #required=True, 
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)

    # GR2R
    p.add_argument("--distribution", type=str, default="gaussian", choices=["gaussian", "poisson", "gamma", "correlated_poisson"])
    p.add_argument("--noise_level", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mc_samples", type=int, default=5)

    # model
    p.add_argument("--model", type=str, default="unet", choices=["autoencoder", "dncnn", "drunet", "unet"])

    # DRUNet sigma control (kept generic; only used when model=drunet)
    p.add_argument("--sigma_mode", type=str, default="noise_level", choices=["noise_level", "fixed"])
    p.add_argument("--sigma_value", type=float, default=0.1)

    # optim/training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)

    # checkpoints/logs
    p.add_argument("--ckpt_dir", type=str, default="./ckpts_gr2r")
    p.add_argument("--init_weights", type=str, default=None,
                   help="Optional: path to weights to initialize model (NOT resume).")
    p.add_argument("--resume", type=str, default=None,
                   help="Optional: lightning checkpoint path to resume training (optimizer/scaler included).")

    p.add_argument("--wandb_project", type=str, default="gr2r-denoising")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)

    args = p.parse_args()

    root = Path(args.div2k_root)
    train_dir = root / "DIV2K_train_HR"
    val_dir = root / "DIV2K_valid_HR"
    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_dir.exists(), f"Missing: {val_dir}"

    train_tf, val_tf = build_transforms(args.img_size)

    train_ds = DIV2KPairs(train_dir, train_tf, args.distribution, args.noise_level)
    val_ds = DIV2KPairs(val_dir, val_tf, args.distribution, args.noise_level)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    lit_model = GR2RLightning(
        model_name=args.model,
        distribution=args.distribution,
        noise_level=args.noise_level,
        alpha=args.alpha,
        lr=args.lr,
        mc_samples=args.mc_samples,
        sigma_mode=args.sigma_mode,
        sigma_value=args.sigma_value,
    )

    # initialize weights (optional) BEFORE training
    if args.init_weights is not None:
        print(f"[init_weights] loading from: {args.init_weights}")
        load_init_weights(lit_model.model, args.init_weights, strict=False)  # strict=False safer for your first try

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_psnr",
            mode="max",
            save_top_k=1,
        )
    ]

    if args.wandb_name is None or str(args.wandb_name).strip() == "":
        args.wandb_name = build_run_name(args)
        
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        save_dir=str(ckpt_dir),   # 로컬에도 wandb 관련 파일 저장
        log_model=False,          # 필요하면 True
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.5,       # 기울기(Gradient)의 최대 허용 크기
        gradient_clip_algorithm="norm", # 'norm'은 벡터의 전체 길이를 기준으로 자름 (권장)
        # precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=callbacks,
        log_every_n_steps=20,
        enable_progress_bar=True,
        logger=logger,
    )

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
