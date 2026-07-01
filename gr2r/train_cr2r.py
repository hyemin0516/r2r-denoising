from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import DictConfig
from omegaconf import OmegaConf


import numpy as np
import matplotlib.pyplot as plt
import math
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .datasets.div2k import DIV2KPairs
from .datasets.sidd import SIDDTrainDataset, SIDDValidationDataset
from .models.autoencoder import Autoencoder
from .models.dncnn import DnCNN
from .models.drunet import DRUNet
from .models.unet import UNet
from .models.unet_v2 import UNet_v2
from .utils.noise import get_noise_fns
from .utils.transform import build_transforms
from .utils.pixel_downsampling import pixel_unshuffle, pixel_shuffle
from .utils.estimate_sigma import estimate_sigma_mad, estimate_sigma_mad_map, estimate_smart_sigma, estimate_sigma, estimate_sigma_structure_aware, LearnableNoiseEstimator
from .utils.grid_injection import get_grid_noise_map
from .utils.mic import mic
from .utils.frequency_band_mixup import apply_residual_frequency_band_mixup
torch.set_float32_matmul_precision('high')


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
        return DnCNN(in_channels=3, out_channels=3, depth=20, nf=64)
    if m == "drunet":
        # DRUNet expects sigma channel inside; we pass sigma via forward(x, sigma=...)
        return DRUNet(in_channels=3, out_channels=3)
    if m == "unet":
        return UNet(in_channels=3, out_channels=3)
    if m == "unet_v2":
        return UNet_v2(in_channels=3, out_channels=3)
    raise ValueError(f"Unknown model: {model_name} (choose from autoencoder|dncnn|drunet)")

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, weight=None):
        diff = x - y
        # L1과 비슷하지만 미분 가능 (sqrt(x^2 + eps^2))
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if weight is not None:
            w = weight
            C = x.shape[1]
            loss = (loss * w).sum() / (w.sum() * C + 1e-8) 
            return loss
        return torch.mean(loss)

def avg_pool_reflect(x, kernel_size):
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=0)

def highpass(x, kernel_size=15):
    return x - avg_pool_reflect(x, kernel_size)

def fft_phase_loss(pred, ref, hp_kernel=15, eps=1e-8):
    pred_hp = highpass(pred, hp_kernel)
    ref_hp = highpass(ref.detach(), hp_kernel)

    Fp = torch.fft.rfft2(pred_hp, norm="ortho")
    Fr = torch.fft.rfft2(ref_hp, norm="ortho")

    dot = Fp.real * Fr.real + Fp.imag * Fr.imag
    denom = torch.abs(Fp) * torch.abs(Fr) + eps
    phase_sim = dot / denom

    # ref에 실제 high-frequency structure가 있는 부분을 더 신뢰
    amp_w = torch.abs(Fr).detach()
    amp_w = amp_w / (amp_w.mean(dim=(-2, -1), keepdim=True) + eps)
    amp_w = amp_w.clamp(0.1, 5.0)

    loss = ((1.0 - phase_sim) * amp_w).mean()
    return loss


def refine_pd_with_full_anchor(
    teacher_pd,
    teacher_full,
    gate,
    gamma=0.3,
    lp_kernel=15,
):
    # PD-Full difference
    delta = teacher_pd - teacher_full

    # remove low-frequency color/context shift
    delta_low = avg_pool_reflect(delta, lp_kernel)
    delta_hp = delta - delta_low

    # full context + gated PD detail
    refined = teacher_full + gamma * gate * delta_hp
    return refined.clamp(0, 1)

def calc_normalized_gram_matrix(feat):
    """공간 해상도에 무관하게 질감 통계량만 추출하는 정규화된 그람 행렬"""
    b, c, h, w = feat.size()
    feat_flat = feat.reshape(b, c, h * w)
    G = torch.bmm(feat_flat, feat_flat.transpose(1, 2))
    return G.div(c * h * w)  # 🚨 픽셀 수 정규화 필수

def crop_and_get_gram(feat, original_input_size, original_pad_size):
    """스케일을 역산하여 오염된 패딩을 걷어내고 순수 그람 행렬 반환"""
    scale_factor = original_input_size // feat.shape[2]
    feat_pad = original_pad_size // scale_factor
    
    if feat_pad > 0:
        feat_cropped = feat[..., feat_pad:-feat_pad, feat_pad:-feat_pad]
    else:
        feat_cropped = feat
        
    return calc_normalized_gram_matrix(feat_cropped)

def pd_subpatch_phase_mean_loss(residual, stride=4):
    """
    residual: [B * stride^2, C, H, W]
    subpatch-as-batch PD 구조 기준
    """
    Bp, C, H, W = residual.shape
    s2 = stride * stride
    assert Bp % s2 == 0

    B = Bp // s2
    r = residual.reshape(B, s2, C, H, W)

    phase_mean = r.mean(dim=(-2, -1))          # [B, s^2, C]
    mean_center = phase_mean.mean(dim=1, keepdim=True)

    return (phase_mean - mean_center).abs().mean()

class ConfidencePCViewAmpGenerator(nn.Module):
    def __init__(
        self,
        num_sources=10,
        hidden=32,
        min_amp=0.005,
        max_amp=0.15,
        smooth_kernel=9,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.smooth_kernel = smooth_kernel

        # inputs:
        # conf_smooth    : 1ch
        # mc_std_norm    : 1ch
        # base_amp       : num_sources ch
        in_ch = 2 + num_sources

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_sources, 3, padding=1),
        )

    def forward(self, conf_smooth, mc_std_norm, base_amp):
        """
        conf_smooth : [B, 1, H, W]
        mc_std_norm : [B, 1, H, W]
        base_amp    : [B, 10, H, W]

        return:
            pc_amp  : [B, 10, H, W]
        """

        x = torch.cat(
            [
                conf_smooth.detach(),
                mc_std_norm.detach(),
                base_amp.detach(),
            ],
            dim=1,
        )

        raw = self.net(x)

        pc_amp = self.min_amp + (
            self.max_amp - self.min_amp
        ) * torch.sigmoid(raw)

        if self.smooth_kernel is not None and self.smooth_kernel > 1:
            pc_amp = avg_pool_reflect(pc_amp, self.smooth_kernel)

        return pc_amp


class PD_GR2R(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config

        self.teacher, self.student, self.feature = self.build_models(model_name=self.cfg.model.name)
        # self.teacher, self.student = self.build_models(model_name=self.cfg.model.name)
        self.noise_estimator = LearnableNoiseEstimator()

        # GR2R corruptor depends on distribution
        _, self.corruptor = get_noise_fns(self.cfg.r2r.noise_dist)

        self.psnr_student = PeakSignalNoiseRatio(data_range=1.0)
        self.psnr_teacher = PeakSignalNoiseRatio(data_range=1.0)
        self.psnr_teacher_src = PeakSignalNoiseRatio(data_range=1.0)
        
        self.ssim_student = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_teacher = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_teacher_src = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.pc_amp_generator = ConfidencePCViewAmpGenerator(
            num_sources=10,
            hidden=32,
            min_amp=0.005,
            max_amp=0.15,
            smooth_kernel=9,
        )

        self.automatic_optimization = False

    def build_models(self, model_name):
        student = build_model(model_name)
        teacher = build_model(model_name)
        feature = build_model(model_name)

        teacher.load_state_dict(student.state_dict())
        feature.load_state_dict(student.state_dict())

        for p in teacher.parameters():
            p.requires_grad = False

        for p in feature.parameters():
            p.requires_grad = False

        return teacher, student, feature
        # return teacher, student

    def configure_optimizers(self):
        main_params = [p for n, p in self.student.named_parameters()]
        
        if self.cfg.r2r.use_learned_estimator:
            estimator_params = [p for n, p in self.noise_estimator.named_parameters()]
            optimizer = torch.optim.Adam([
                {'params': main_params, 'lr': self.cfg.solver.lr},
                {'params': estimator_params, 'lr': 1e-6}
            ], weight_decay=self.cfg.solver.wd)
        else:
            optimizer = torch.optim.Adam(
                main_params, lr=self.cfg.solver.lr, weight_decay=self.cfg.solver.wd
            )
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.solver.lr, weight_decay=self.cfg.solver.wd)

        # total_steps = self.cfg.solver.max_steps
        # warmup_steps = self.cfg.solver.warmup_steps

        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=0.01, total_iters=warmup_steps
        # )

        # cosine_steps = total_steps - warmup_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.solver.max_steps,  # 예: 100
            eta_min=self.cfg.solver.min_lr  # 최소 LR (0이 되지 않게 안전장치)
        )

        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, 
        #     schedulers=[warmup_scheduler, cosine_scheduler], 
        #     milestones=[warmup_steps]
        # )
        
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         optimizer,
        #         milestones=self.cfg.solver.lr_steps,  
        #         gamma=0.5                        
        # )  


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # ★ 매우 중요: 'epoch'이 아니라 'step'마다 갱신!
                "frequency": 1,
                # "monitor": "val_psnr", # (선택사항) ReduceLROnPlateau 쓸 때만 필요
                "strict": True,
            }
        }


    def update_ema_variables_cosine(self, current_step, rampup_end=1000):
        """
        DINO의 철학을 2k 스텝에 압축한 SOTA 코사인 모멘텀 스케줄러
        """
        base_m = 0.99   # 초기 귀가 얇은 상태 (빨리 배우기)
        max_m = 0.999   # 최종 철벽 방어 상태
        
        # if current_step < rampup_end:
        #     # 0 ~ 2k 스텝: DINO 스타일의 Half-Cosine 곡선을 그리며 부드럽게 증가
        #     m = max_m - (max_m - base_m) * 0.5 * (1.0 + math.cos(math.pi * current_step / rampup_end))
        # else:
        #     # 2k 스텝 이후: 0.999로 영구 고정 (Student의 Blur 타협을 완벽 방어!)
        #     m = 0.9995

        m = self.cfg.uda.momentum
        
        with torch.no_grad():
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.mul_(m).add_((1 - m) * param_q.detach())


    def get_pseudo_confidence_weight(self, current_step, rampup_length=1500, max_w=0.5):
        """
        [Gaussian Ramp-up 스케줄러]
        학생이 타겟 노이즈에 과적합 되기 전(1500 step)까지 
        스승의 피드백 가중치를 S자 곡선으로 부드럽게 끌어올립니다.
        """
        if current_step >= rampup_length:
            return max_w
            
        # 0.0 ~ 1.0 사이의 진행률
        phase = current_step / rampup_length
        
        # Gaussian Ramp-up 핵심 공식: exp(-5 * (1 - phase)^2)
        # 초반에는 0에 가깝게 기어다가, 1000 step 부근에서 급격히 솟구침
        return max_w * math.exp(-5.0 * (1.0 - phase) ** 2)
        
        # Mean Teacher 논문 오리지널 공식: exp(-5 * (1 - x)^2)
        # 초반(phase=0)에는 0.006으로 거의 0에 가깝다가 후반에 급상승
        return max_w * math.exp(-5.0 * (1.0 - phase) ** 2)

    def get_current_target_weight(self, current_step, rampup_starts=500, rampup_ends=1500, max_weight=2.0, cfg=True):
        """
        1000스텝까지는 weight = 0 (학생 스스로 뼈대 잡는 시간)
        1000 ~ 4000스텝 동안 0 -> 5.0으로 부드럽게 증가 (스승의 개입 시작)
        4000스텝 이후는 5.0 고정 (스승의 정답지에 풀-집중)
        """
        if cfg:
           rampup_starts=self.cfg.uda.rampup_starts 
           rampup_ends=self.cfg.uda.rampup_starts 
           max_weight=self.cfg.uda.max_weight
        if current_step < rampup_starts:
            return 0.0
        elif current_step >= rampup_ends:
            return max_weight
        else:
            # 0 ~ 1 사이로 진행률 계산
            progress = (current_step - rampup_starts) / (rampup_ends - rampup_starts)
            return max_weight * progress

    def get_current_consistency_weight(self, current_step, rampup_starts=2000, rampup_ends=10000, max_weight=1.5, min_weight=0.1):
        """
        초반 (0 ~ 2000): min_weight (0.1) 유지 (Mini-MC로 학생이 스스로 깨끗한 뼈대를 깎아내는 시간)
        중반 (2000 ~ 10000): 0.1 -> 1.5로 아주 천천히 우상향 (Slow Cooking, 스무스한 지식 이식)
        후반 (10000 이후): max_weight (1.5) 고정 (깨끗해진 PD 뼈대를 Full 해상도에 강력하게 쐐기 박기)
        """
        if current_step <= rampup_starts:
            return min_weight
        elif current_step >= rampup_ends:
            return max_weight
        else:
            # 시작점과 끝점 사이의 정확한 진행률(0.0 ~ 1.0) 계산
            rampup_length = float(rampup_ends - rampup_starts)
            progress = (current_step - rampup_starts) / rampup_length
            
            # (옵션) 만약 Linear보다 더 부드러운 곡선을 원하시면 아래 Cosine 방식을 쓸 수도 있습니다.
            # import math
            # progress = 0.5 * (1.0 - math.cos(math.pi * progress))
            
            return min_weight + (max_weight - min_weight) * progress

    def get_current_src_weight(self, current_step, decay_starts=8000, decay_ends=15000, max_weight=1.0, min_weight=0.1):
        """
        [Phase 1] 초반~중반 (0 ~ 8000): max_weight (1.0) 유지 
                  -> '야생의 불도저' 모드. 거친 재오염 노이즈 덩어리를 파괴하는 데 풀파워 집중.
        [Phase 2] 권력 교체기 (8000 ~ 15000): 1.0 -> 0.1로 부드럽게 우하향 (Cosine Decay)
                  -> w_cons가 1.0을 돌파하며 강력해지는 시점에 맞춰, src의 힘을 서서히 빼줌.
        [Phase 3] 극후반 (15000 이후): min_weight (0.1) 고정 
                  -> 노이즈 억제는 끝났음. 남은 학습은 MC 황금 정답지(cons)와 스승(trg)에게 온전히 맡김.
        """
        if current_step <= decay_starts:
            return max_weight
        elif current_step >= decay_ends:
            return min_weight
        else:
            # 시작점과 끝점 사이의 진행률 (0.0 ~ 1.0)
            progress = (current_step - decay_starts) / float(decay_ends - decay_starts)
            
            # Cosine Decay: 1.0에서 0.0으로 떨어지는 매우 부드러운 'S자' 곡선
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # 최종 가중치 계산 (1.0부터 시작해서 0.1에 부드럽게 안착)
            return min_weight + (max_weight - min_weight) * cosine_decay
        
    def forward(self, x: torch.Tensor, sigma_map=None, padding=False):
        # ==========================================================
        # [수정] Zero-Padding Artifact 방지 (Ghosting 해결 핵심)
        # ==========================================================
        # 모델(DnCNN 등)이 3x3 Conv를 쓸 때 테두리에 0을 채우는데,
        # PD로 쪼개진 이미지에서는 이 테두리가 나중에 합쳐질 때 
        # 이미지 정중앙에 '검은 격자'나 '위치 어긋남'을 만듭니다.
        # 이를 막기 위해 입력 이미지를 미리 거울처럼 반사(Reflect)해서 넓혀줍니다.

        if padding:
            pad_size = 16  # 8픽셀 정도면 충분합니다 (Receptive Field 고려)
            x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        else:
            x_pad = x

        # ----------------------------------------------------------
        # 기존 로직 (Sigma 처리 등)
        # ----------------------------------------------------------
        # 1. DRUNet이 아니면 Sigma 필요 없음
        if self.cfg.model.name.lower() != "drunet":
            out_pad = self.model(x_pad)
        else:
            # 2. Sigma 값 결정
            if sigma_map is not None:
                sigma = sigma_map
            else:
                if self.cfg.r2r.sigma_mode.lower() == "fixed":
                    sigma_val = float(self.cfg.r2r.sigma_value)
                else:
                    sigma_val = float(self.cfg.r2r.noise_level)
                sigma = torch.full((x.size(0),), sigma_val, device=x.device, dtype=x.dtype)

            # 3. 차원 불일치 자동 보정 (x_pad 사이즈 기준)
            if isinstance(sigma, torch.Tensor) and sigma.ndim == 1:
                # x가 PD로 인해 배치가 늘어났는데 sigma가 원본 배치 사이즈라면
                if x_pad.size(0) != sigma.size(0):
                    if x_pad.size(0) > sigma.size(0) and x_pad.size(0) % sigma.size(0) == 0:
                        repeat_factor = x_pad.size(0) // sigma.size(0)
                        sigma = sigma.repeat_interleave(repeat_factor)
            
            out_pad = self.model(x_pad, sigma=sigma)

        # ==========================================================
        # [복구] 아까 붙였던 패딩만큼 다시 잘라내기 (Center Crop)
        # ==========================================================
        if padding:
            out = out_pad[..., pad_size:-pad_size, pad_size:-pad_size]
        else:
            out = out_pad
        
        return out

    def r2r_loss(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor) -> torch.Tensor:
        if self.cfg.solver.loss_type == "mse":
            loss = torch.nn.functional.mse_loss(pred, target)

        elif self.cfg.solver.loss_type == "cbl":
            loss = self.cbl_loss(pred, target)

        elif self.cfg.solver.loss_type == "nll":
            mu, log_var = torch.chunk(pred, 2, dim=1)
            log_var = torch.clamp(log_var, min=-10, max=10)
            var = torch.exp(log_var)

            loss = torch.nn.functional.gaussian_nll_loss(mu, target, var, reduction='mean')

        return loss

    def smooth_l1_loss(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor,
                 weight=None) -> torch.Tensor:
        if weight is not None:
            raw_loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none')
            masked_loss = raw_loss * weight

            valid_pixels = weight.sum() + 1e-8
            loss = masked_loss.sum() / valid_pixels

        else:
            loss =  torch.nn.functional.smooth_l1_loss(pred, target)

        return loss

    def tta_forward(self, img, idx):
        if idx == 0: return img
        elif idx == 1: return torch.rot90(img, 1, [2, 3])
        elif idx == 2: return torch.rot90(img, 2, [2, 3])
        elif idx == 3: return torch.rot90(img, 3, [2, 3])
        elif idx == 4: return torch.flip(img, [3])
        elif idx == 5: return torch.rot90(torch.flip(img, [3]), 1, [2, 3])
        elif idx == 6: return torch.rot90(torch.flip(img, [3]), 2, [2, 3])
        elif idx == 7: return torch.rot90(torch.flip(img, [3]), 3, [2, 3])

    def tta_backward(self, img, idx):
        if idx == 0: return img
        elif idx == 1: return torch.rot90(img, -1, [2, 3])
        elif idx == 2: return torch.rot90(img, -2, [2, 3])
        elif idx == 3: return torch.rot90(img, -3, [2, 3])
        elif idx == 4: return torch.flip(img, [3])
        elif idx == 5: return torch.flip(torch.rot90(img, -1, [2, 3]), [3])
        elif idx == 6: return torch.flip(torch.rot90(img, -2, [2, 3]), [3])
        elif idx == 7: return torch.flip(torch.rot90(img, -3, [2, 3]), [3])

    def training_step(self, batch, batch_idx):       
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        y = batch["y"] # High-Res Real Noisy
        pad_size = 16

        # ===========================================================
        # 1. MC sampling
        # ===========================================================
        MC_STEPS = 8
        self.teacher.eval()
        with torch.no_grad():
            mc_preds = []
            for i in range(MC_STEPS):
                aug_idx = i % 8
                y_aug = self.tta_forward(y, aug_idx)
                y_sub_aug = pixel_unshuffle(y_aug, 2)
                y_aug_pad = F.pad(y_sub_aug, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

                # Heuristic 기반 (estimator flag 무관, MC sampling은 보조 기능)
                sigma_map_aug, _ = estimate_sigma_structure_aware(y_aug_pad)
                
                if self.cfg.r2r.use_learned_estimator:
                    aug_variance = sigma_map_aug ** 2
                    log_var_aug = self.noise_estimator(y_aug_pad, aug_variance)
                    sigma_map_for_aug = torch.exp(0.5 * log_var_aug).detach()
                else:
                    sigma_map_for_aug = sigma_map_aug.detach()
                
                root_noise_b = torch.randn_like(y_aug_pad)
                micro_noise = root_noise_b * (sigma_map_for_aug * 0.1)
                y_input = y_aug_pad + micro_noise
                
                src_residual_b = self.teacher(y_input.clamp(0, 1))
                src_pred_b_full = src_residual_b + y_input.clamp(0, 1)
                src_pred_b = src_pred_b_full[..., pad_size:-pad_size, pad_size:-pad_size]

                assembled_src_pred_b = pixel_shuffle(src_pred_b, 2)
                restored_pred = self.tta_backward(assembled_src_pred_b, aug_idx)

                
                mc_preds.append(restored_pred.clamp(0, 1))
            
            pred_mc_stack = torch.stack(mc_preds)
            pred_mc_clean = pred_mc_stack.mean(dim=0)
            mc_variance = pred_mc_stack.var(dim=0, unbiased=False)   # [B, C, H, W] full resolution
        
        # MC variance 정규화 (full resolution)
        mc_var_lf = torch.nn.functional.avg_pool2d(
            mc_variance.mean(1, keepdim=True), kernel_size=5, stride=1, padding=2
        )
        mc_var_q95 = mc_var_lf.flatten(2).quantile(0.95, dim=2)[..., None, None]
        mc_var_norm = (mc_var_lf / (mc_var_q95 + 1e-8)).clamp(0, 1)
        
        # MC variance를 src PD stride=4 도메인으로 변환
        mc_var_norm_sub = pixel_unshuffle(mc_var_norm, self.cfg.r2r.pd_stride).mean(dim=1, keepdim=True).detach()
        mc_var_norm_sub_pad = F.pad(mc_var_norm_sub, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        # mc_var_norm_sub shape: [B*16, 1, H/4, W/4] (B*16 = batch_size after PD)
        
        # Confidence (target loss용)
        confidence = 1.0 / (1.0 + mc_variance * self.cfg.solver.conf_scale)
        conf_threshold = self.cfg.solver.conf_thd
        hard_conf_mask = (confidence > conf_threshold).float()
        
        # Var 로깅
        var_flat = mc_variance.view(-1)
        var_99th = torch.quantile(var_flat, 0.99)
        self.log("debug/var_mean", var_flat.mean())
        self.log("debug/var_max", var_flat.max())
        self.log("debug/var_99th", var_99th)

        # ===========================================================
        # 2. Target loss (Patch-Craft based)
        # ===========================================================
        with torch.no_grad():
            B, C, H, W = y.shape
            patch_size = 16
            half = patch_size // 2
            pad = half  # 8
            
            # teacher_pd
            teacher_pd = pred_mc_clean
            base_idx = torch.randint(0, len(mc_preds), (1,)).item()
            base_pd = mc_preds[base_idx].detach()

            
            # confidence map smoothing
            y_pad = F.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            teacher_full_residual, trg_feat_pad = self.teacher(y_pad.clamp(0, 1), return_features='enc')
            teacher_full_pad = teacher_full_residual + y_pad.clamp(0, 1)
            teacher_full = teacher_full_pad[..., pad_size:-pad_size, pad_size:-pad_size]

            #teacher gram
            conf_map = confidence.mean(dim=1, keepdim=True).detach()

            conf_smooth = F.avg_pool2d(
                conf_map,
                kernel_size=patch_size,
                stride=1,
                padding=patch_size // 2
            )
            conf_smooth = conf_smooth[..., :H, :W]

            mc_stack = torch.stack([p.detach() for p in mc_preds], dim=0)
            mc_std = mc_stack.std(dim=0).mean(dim=1, keepdim=True)
        
            mc_q95 = mc_std.flatten(2).quantile(0.95, dim=2)[..., None, None]
            mc_std_norm = (mc_std / (mc_q95 + 1e-8)).clamp(0, 1)

            # n_hat residual
            n_hat_full = y - teacher_full
            n_hat_pd = y - teacher_pd

            # Batch shuffle indices (각 source마다 다른 permutation)
            perm1 = torch.randperm(B, device=y.device)
            perm2 = torch.randperm(B, device=y.device)
            perm3 = torch.randperm(B, device=y.device)
            perm4 = torch.randperm(B, device=y.device)

            # Cross-batch noise
            n_hat_full_cross1 = n_hat_full[perm1]
            n_hat_full_cross2 = n_hat_full[perm2]
            n_hat_pd_cross1 = n_hat_pd[perm3]
            n_hat_pd_cross2 = n_hat_pd[perm4]

            base_amp = (
                (0.15 - 0.05)
                * torch.rand(B, 10, 1, 1, device=y.device)
                + 0.05
            )
            base_amp = base_amp.expand(-1, -1, H, W)
        
            # old reference only for regularization
            min_gate = 0.3
            old_noise_gate = min_gate + (1.0 - min_gate) * conf_smooth
            amp_ref = base_amp * old_noise_gate

        pc_amp = self.pc_amp_generator(
            conf_smooth=conf_smooth,
            mc_std_norm=mc_std_norm,
            base_amp=base_amp,
        )
        
        sigma_1 = pc_amp[:, 0:1]
        sigma_2 = pc_amp[:, 1:2]
        sigma_3 = pc_amp[:, 2:3]
        sigma_4 = pc_amp[:, 3:4]
        sigma_5 = pc_amp[:, 4:5]
        sigma_6 = pc_amp[:, 5:6]
        sigma_7 = pc_amp[:, 6:7]
        sigma_8 = pc_amp[:, 7:8]
        sigma_9 = pc_amp[:, 8:9]
        sigma_10 = pc_amp[:, 9:10]
            
        # Type 1: n_hat residual based
        source_pos1 = teacher_full + sigma_1 * n_hat_full_cross1
        source_neg1 = teacher_full - sigma_2 * n_hat_full_cross2
        
        # Type 2: Variance-preserving stochastic noise mixing
        # 8개의 MC 샘플 중 무작위로 2개의 뷰 선택
        mc_idx_1 = torch.randint(1, len(mc_preds), (1,)).item()
        mc_idx_2 = torch.randint(1, len(mc_preds), (1,)).item()
        mc_idx_3 = torch.randint(1, len(mc_preds), (1,)).item()
        mc_idx_4 = torch.randint(1, len(mc_preds), (1,)).item()
        
        # 순수 인식적 편차 추출 (Zero-mean deviation)
        mc_dev_1 = mc_preds[mc_idx_1].detach() - pred_mc_clean
        mc_dev_2 = mc_preds[mc_idx_2].detach() - pred_mc_clean
        mc_dev_3 = mc_preds[mc_idx_3].detach() - pred_mc_clean
        mc_dev_4 = mc_preds[mc_idx_4].detach() - pred_mc_clean
        
        # 2. Phase-Shifted Decorrelation (격자 파괴 로직)
        # -3 ~ +3 픽셀 단위의 Random Roll을 통해 PD의 고주파 주기성(Grid)을 완벽히 파괴
        shift_x1, shift_y1 = torch.randint(-3, 4, (2,)).tolist()
        shift_x2, shift_y2 = torch.randint(-3, 4, (2,)).tolist()
        shift_x3, shift_y3 = torch.randint(-3, 4, (2,)).tolist()
        shift_x4, shift_y4 = torch.randint(-3, 4, (2,)).tolist()
        
        mc_dev_1_jittered = torch.roll(mc_dev_1, shifts=(shift_y1, shift_x1), dims=(2, 3))
        mc_dev_2_jittered = torch.roll(mc_dev_2, shifts=(shift_y2, shift_x2), dims=(2, 3))
        mc_dev_3_jittered = torch.roll(mc_dev_3, shifts=(shift_y3, shift_x3), dims=(2, 3))
        mc_dev_4_jittered = torch.roll(mc_dev_4, shifts=(shift_y4, shift_x4), dims=(2, 3))
        
        # 3. Variance Matching (스케일 동기화)
        n_hat_std = n_hat_pd.std(dim=(2, 3), keepdim=True) + 1e-8
        mc_dev_std_1 = mc_dev_1_jittered.std(dim=(2, 3), keepdim=True) + 1e-8
        mc_dev_std_2 = mc_dev_2_jittered.std(dim=(2, 3), keepdim=True) + 1e-8
        mc_dev_std_3 = mc_dev_3_jittered.std(dim=(2, 3), keepdim=True) + 1e-8
        mc_dev_std_4 = mc_dev_4_jittered.std(dim=(2, 3), keepdim=True) + 1e-8
        
        n_new_1 = (mc_dev_1_jittered / mc_dev_std_1) * n_hat_std
        n_new_2 = (mc_dev_2_jittered / mc_dev_std_2) * n_hat_std
        n_new_3 = (mc_dev_3_jittered / mc_dev_std_3) * n_hat_std
        n_new_4 = (mc_dev_4_jittered / mc_dev_std_4) * n_hat_std
        
        # 4. Variance-preserving Spherical Interpolation
        alpha = confidence.clamp(min=0.3, max=1.0)
        beta = torch.sqrt((1.0 - alpha**2).clamp(min=0))
        
        n_mixed_1 = alpha * n_hat_pd + beta * n_new_1
        n_mixed_2 = alpha * n_hat_pd + beta * n_new_2
        n_mixed_3 = alpha * n_hat_pd + beta * n_new_3
        n_mixed_4 = alpha * n_hat_pd + beta * n_new_4
        
        # 5. 타겟 생성 (Blur 없는 teacher_full을 베이스로 사용)
        source_pos2 = base_pd + sigma_3 * n_mixed_1
        source_neg2 = base_pd - sigma_4 * n_mixed_2
        source_pos3 = base_pd + sigma_5 * n_mixed_3
        source_neg3 = base_pd - sigma_6 * n_mixed_4

        # type 3: Zero-Artifact Extrapolation + frequency-mixed artifact direction
        artifact_delta = teacher_pd - teacher_full
        
        delta_std = artifact_delta.std(dim=(2, 3), keepdim=True) + 1e-8
        artifact_delta_unit = artifact_delta / delta_std
        
        
        artifact_delta_matched = artifact_delta_unit * n_hat_std
        
        source_pos4 = teacher_full + sigma_7 * artifact_delta_matched
        source_neg4 = teacher_full - sigma_8 * artifact_delta_matched

        # Type 4: n_hat_pd residual based
        source_pos5 = teacher_pd + sigma_9 * n_hat_pd_cross1
        source_neg5 = teacher_pd - sigma_10 * n_hat_pd_cross2
        
        # Sources list (6 sources)
        sources = [
            source_pos1,     # 0: n_hat_full +
            source_neg1,     # 1: n_hat_full -
            source_pos2,     # 2: n_hat_pd +
            source_neg2,     # 3: n_hat_pd -
            source_pos3,     # 4: n_hat_pd +
            source_neg3,     # 5: n_hat_pd -
            source_pos4,     # 6: extrapolated_anchor +
            source_neg4,     # 7: extrapolated_anchor -
            source_pos5,
            source_neg5,
            
        ]
            
        # ====== Padding 추가 ======
        # 모든 sources와 mask를 mirror padding
        sources_padded = [F.pad(s, (pad, pad, pad, pad), mode='reflect') for s in sources]
        teacher_pd_padded = F.pad(teacher_pd, (pad, pad, pad, pad), mode='reflect')
        conf_smooth_padded = F.pad(conf_smooth, (pad, pad, pad, pad), mode='reflect')
        
        H_pad = H + 2 * pad
        W_pad = W + 2 * pad
        
        # Grid 크기 (padded 기준)
        n_grid_h = H_pad // patch_size
        n_grid_w = W_pad // patch_size
        
        # Random offset (padded image 안에서)
        offset_h = torch.randint(0, patch_size, (1,)).item()
        offset_w = torch.randint(0, patch_size, (1,)).item()
        
        # Padded matched_target
        # matched_target_padded = teacher_pd_padded.clone()

        matched_target_padded = teacher_pd_padded.clone()
        selection_mask_padded = torch.zeros(B, 1, H_pad, W_pad, device=y.device)
        
        for b in range(B):
            for gi in range(n_grid_h):
                for gj in range(n_grid_w):
                    cy_val = offset_h + gi * patch_size + half
                    cx_val = offset_w + gj * patch_size + half
                    cy_val = min(cy_val, H_pad - 1)
                    cx_val = min(cx_val, W_pad - 1)
                    
                    # Clean check (padded mask)
                    # if hard_conf_mask_padded[b, 0, cy_val, cx_val] < 0.5:
                    #     continue
                    
                    src_idx = torch.randint(0, len(sources_padded), (1,)).item()
                    source = sources_padded[src_idx]
                    
                    cy_s = max(0, cy_val - half)
                    cy_e = min(H_pad, cy_val + half)
                    cx_s = max(0, cx_val - half)
                    cx_e = min(W_pad, cx_val + half)
                    
                    matched_target_padded[b, :, cy_s:cy_e, cx_s:cx_e] = \
                        source[b, :, cy_s:cy_e, cx_s:cx_e]
                    selection_mask_padded[b, 0, cy_s:cy_e, cx_s:cx_e] = 1
        
        # Crop back to original size
        matched_target = matched_target_padded[:, :, pad:pad+H, pad:pad+W]
        selection_mask = selection_mask_padded[:, :, pad:pad+H, pad:pad+W]
        
        patch_craft_target = matched_target.detach()
                
        
        # Loss
        # with torch.no_grad():
        #     sigma_map, _ = estimate_sigma_structure_aware(y)
        #     sv_input_trg = self.cfg.r2r.sv_input_trg

        #     confidence_factor = 1.0 + (1.0 - confidence.clamp(min=conf_threshold))
        #     root_noise_trg = torch.randn_like(y)
        #     y_input_trg = (y + root_noise_trg * sigma_map * sv_input_trg * confidence_factor).clamp(0, 1)

        y_pad = F.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        res_orig = self.student(y_pad.clamp(0, 1))
        pred_orig_pad = res_orig + y_pad.clamp(0, 1)
        pred_orig = pred_orig_pad[..., pad_size:-pad_size, pad_size:-pad_size]
        # loss_pc = torch.nn.functional.mse_loss(pred_orig, patch_craft_target)
        loss_pc = self.r2r_loss(pred_orig, patch_craft_target)
        # loss_mask = (confidence > 0.5).float()
        # loss_pc = (((pred_orig - patch_craft_target) ** 2) * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        pred_detached = teacher_full.detach()

        second_pass_residual = self.student(pred_detached)
        second_pass_pred = second_pass_residual + pred_detached

        loss_idem = F.mse_loss(second_pass_pred, pred_detached, reduction='none')
        w_idem = (1-confidence).detach().clamp(min=0.1) # * 3.0

        # loss_pc_amp_prior = F.smooth_l1_loss(
        #     pc_amp,
        #     amp_ref.detach(),
        # )
        
        loss_pc_amp_tv = (
            (pc_amp[..., 1:, :] - pc_amp[..., :-1, :]).abs().mean()
            + (pc_amp[..., :, 1:] - pc_amp[..., :, :-1]).abs().mean()
        )

        w_pc_amp_prior = 0
        w_pc_amp_tv = 0.001
        
        trg_loss = loss_pc + (loss_idem * w_idem).mean() + (w_pc_amp_tv * loss_pc_amp_tv)
        # trg_loss = loss_pc
        
        # Logging
        self.log('loss_idem', loss_idem.mean(), on_step=True)
        self.log('target_loss', trg_loss, on_step=True, logger=True)
        self.log('debug/selection_mask_ratio', selection_mask.mean(), on_step=True, logger=True)
        self.log("train/pc_amp_mean", pc_amp.mean())
        self.log("train/pc_amp_std", pc_amp.std())
        self.log("train/pc_amp_min", pc_amp.min())
        self.log("train/pc_amp_max", pc_amp.max())
        self.log("train/pc_amp_tv", loss_pc_amp_tv)

        # ===========================================================
        # 3. Source self-Supervised Loss
        # ===========================================================
        y_sub = pixel_unshuffle(y, self.cfg.r2r.pd_stride)
        y_sub_pad = F.pad(y_sub, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        sigma_map_s, edge_gate_s = estimate_sigma_structure_aware(
            y_sub_pad,
            window_size=7,
            edge_gate_min=0.5
        )
        
        if self.cfg.r2r.use_learned_estimator:
            # Learnable estimator 사용
            s_variance = sigma_map_s ** 2
            log_var = self.noise_estimator(y_sub_pad, s_variance)
            sigma_map_for_est = torch.exp(0.5 * log_var)
            sigma_map_for_src = sigma_map_for_est.detach()
        else:
            # Heuristic만 사용
            sigma_map_for_est = None
            sigma_map_for_src = sigma_map_s.detach()        
        
        root_noise = torch.randn_like(y_sub_pad)
        heavy_noise = root_noise * (sigma_map_for_src * self.cfg.r2r.sigma_value) 
        target_noise = root_noise * (sigma_map_for_src * 0.5)
        
        y1_sub = (y_sub_pad + heavy_noise).clamp(0, 1)
        y2_sub_pad = (y_sub_pad - target_noise).clamp(0, 1)
        y2_sub = y2_sub_pad[..., pad_size:-pad_size, pad_size:-pad_size]
        
        # Student 예측
        src_residual_pad, src_feat_pad = self.student(y1_sub.clamp(0, 1), return_features='enc')
        src_pred_pad = src_residual_pad + y1_sub.clamp(0, 1)

        src_residual = src_residual_pad[..., pad_size:-pad_size, pad_size:-pad_size]
        src_pred = src_pred_pad[..., pad_size:-pad_size, pad_size:-pad_size]
        src_pred_full = pixel_shuffle(src_pred, self.cfg.r2r.pd_stride)

        # phase anchor loss
        # loss_src_phase_anchor = fft_phase_loss(
        #     src_pred_full,
        #     teacher_full.detach(),
        #     hp_kernel=15
        # )
        # w_src_phase_anchor = 0.001

        # gram-consistency
        _, _, H_padded, _ = y1_sub.shape
        G_src_batch = crop_and_get_gram(src_feat_pad, H_padded, pad_size)

        b_new, c_gram, _ = G_src_batch.shape
        stride = self.cfg.r2r.pd_stride
        b_original = b_new // (stride * stride)

        G_src_grouped = G_src_batch.reshape(b_original, stride * stride, c_gram, c_gram)
        G_center = G_src_grouped.mean(dim=1, keepdim=True)
        loss_gram = torch.nn.functional.l1_loss(G_src_grouped, G_center.expand_as(G_src_grouped))
        w_gram = self.cfg.solver.w_gram

        # phase consistency
        loss_phase = pd_subpatch_phase_mean_loss(
            src_residual,
            stride=self.cfg.r2r.pd_stride
        )
        w_phase = self.cfg.solver.w_phase
        
        # Main PD Loss
        src_loss_main = self.r2r_loss(src_pred, y2_sub)
        src_loss = src_loss_main + (loss_gram * w_gram) + (loss_phase * w_phase) # + (loss_src_phase_anchor * w_src_phase_anchor)
        # src_loss = src_loss_main
        self.log('src_loss', src_loss, on_step=True, logger=True)
        self.log('gram_loss',loss_gram*w_gram, on_step=True)
        self.log('phase_loss',loss_phase * w_phase, on_step=True)
        # self.log('src-phase_loss',loss_src_phase_anchor * w_src_phase_anchor, on_step=True)
        # self.log('debug/mc_boost_mean', mc_boost.mean(), on_step=True)
        # self.log('debug/mc_boost_max', mc_boost.max(), on_step=True)
            
        # ===========================================================
        # 4. Estimator Loss (flag 분기)
        # ===========================================================
        if self.cfg.r2r.use_learned_estimator:
            def get_robust_flat_weight(img):
                B, C, H, W = img.shape
                sobel_x = torch.tensor([[-1., 0., 1.],
                                        [-2., 0., 2.],
                                        [-1., 0., 1.]], device=img.device, dtype=img.dtype)
                sobel_y = sobel_x.T
                sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
                sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
                
                gx = torch.nn.functional.conv2d(img, sobel_x, padding=1, groups=C)
                gy = torch.nn.functional.conv2d(img, sobel_y, padding=1, groups=C)
                grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
                
                grad_norm = grad_mag / (grad_mag.flatten(2).quantile(0.95, dim=2)[..., None, None] + 1e-8)
                structure_weight = torch.exp(-5.0 * grad_norm.clamp(0, 1))
                
                return structure_weight.mean(dim=1, keepdim=True)
            
            flat_weight = get_robust_flat_weight(x_hat.detach())
            
            # Base variance
            n_hat_sq_mean = torch.nn.functional.avg_pool2d(n_hat**2, kernel_size=7, stride=1, padding=3)
            n_hat_mean = torch.nn.functional.avg_pool2d(n_hat, kernel_size=7, stride=1, padding=3)
            base_variance = torch.clamp(n_hat_sq_mean - n_hat_mean**2, min=1e-8)
            
            # Boost factor (mc_var_norm은 위에서 계산됨)
            boost_scale = 1.5
            boost_factor = 1.0 + (mc_var_norm * flat_weight * boost_scale)
            refined_target = base_variance * boost_factor.detach()
            
            # PD domain
            refined_target_sub = pixel_unshuffle(refined_target, self.cfg.r2r.pd_stride)
            flat_weight_sub = pixel_unshuffle(flat_weight, self.cfg.r2r.pd_stride).detach()
            
            # KL loss
            pred_var_safe = torch.clamp(sigma_map_for_est**2, min=1e-8)
            target_var_safe = torch.clamp(refined_target_sub, min=1e-8)
            
            kl_per_pixel = 0.5 * (
                torch.log(pred_var_safe / target_var_safe) + target_var_safe / pred_var_safe - 1.0
            )
            kl_per_pixel = kl_per_pixel.mean(dim=1, keepdim=True)
            
            weight_sum = flat_weight_sub.sum() + 1e-8
            loss_estimator = (kl_per_pixel * flat_weight_sub).sum() / weight_sum
            
            w_est = self.cfg.solver.w_est if hasattr(self.cfg.solver, 'w_est') else 1e-3
            
            self.log('loss_estimator', loss_estimator, on_step=True, logger=True)
            self.log('debug/estimator_boost_max', boost_factor.max(), on_step=True)
            self.log('debug/sigma_map_mean', sigma_map_for_est.mean(), on_step=True)
        else:
            loss_estimator = torch.tensor(0.0, device=y.device)
            w_est = 0.0

        # ===========================================================
        # 5. Train loss & backward
        # ===========================================================
        w_uda = self.cfg.solver.w_uda
        w_src = self.cfg.solver.w_src

        train_loss = (w_src * src_loss) + (w_uda * trg_loss) + (w_est * loss_estimator)
        self.log('train_loss', train_loss, on_step=True, logger=True)

        opt.zero_grad()
        self.manual_backward(train_loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt.step()
        sch.step()

        # ===========================================================
        # 6. EMA update
        # ===========================================================
        self.update_ema_variables_cosine(self.global_step)
                
        return train_loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]

        pad_size=16
        y_pad = F.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        student_residual = self.student(y_pad.clamp(0, 1))
        teacher_residual = self.teacher(y_pad.clamp(0, 1))

        student_pred = student_residual + y_pad.clamp(0, 1)
        teacher_pred = teacher_residual + y_pad.clamp(0, 1)
        
        student_out = student_pred[..., pad_size:-pad_size, pad_size:-pad_size]
        teacher_out = teacher_pred[..., pad_size:-pad_size, pad_size:-pad_size]
        
        student_eval = student_out.clamp(0, 1)
        teacher_eval = teacher_out.clamp(0, 1)
        
        self.log("student_psnr", self.psnr_student(student_eval, x),
                 prog_bar=True, on_step=False, on_epoch=True)
        
        self.log("student_ssim", self.ssim_student(student_eval, x),
                 prog_bar=False, on_step=False, on_epoch=True)
        
        self.log("teacher_psnr", self.psnr_teacher(teacher_eval, x),
                 prog_bar=True, on_step=False, on_epoch=True)
        
        self.log("teacher_ssim", self.ssim_teacher(teacher_eval, x),
                 prog_bar=False, on_step=False, on_epoch=True)

        y_sub = pixel_unshuffle(y, 2)
        y_sub_pad = F.pad(y_sub, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        with torch.no_grad():
            teacher_residual_src = self.teacher(y_sub_pad.clamp(0, 1))
            teacher_pred_sub_pad = (teacher_residual_src + y_sub_pad).clamp(0, 1)
            teacher_pred_sub = teacher_pred_sub_pad[..., pad_size:-pad_size, pad_size:-pad_size]
        teacher_pred_src = pixel_shuffle(teacher_pred_sub, 2)
        teacher_src_eval = teacher_pred_src.clamp(0, 1)
        
        
        self.log("teacher_psnr_source", self.psnr_teacher_src(teacher_src_eval, x),
                 on_step=False, on_epoch=True)
        
        self.log("teacher_ssim_source", self.ssim_teacher_src(teacher_src_eval, x),
                 on_step=False, on_epoch=True)


        if batch_idx == 349: # 349
            # 깔끔하게 내부 메서드 호출!
            self.log_images(x[0:1], y[0:1], student_out[0:1], key="student_samples")
            self.log_images(x[0:1], y[0:1], teacher_out[0:1], key="teacher_samples")
            self._log_mc_confidence_map(y, batch_idx, target_batch=349)

    def log_images(self, clean, noisy, denoised, key="val_samples"):
        """
        WandB에 이미지를 로깅하는 내부 헬퍼 메서드
        """
        # 로거가 없으면 조기 종료
        if self.logger is None:
            return

        # 1. 텐서 -> Numpy 변환 (CPU 이동 포함)
        def to_np(t):
            return np.clip(t[0].detach().cpu().permute(1, 2, 0).numpy(), 0, 1)

        clean_img = to_np(clean)
        noisy_img = to_np(noisy)
        denoised_img = to_np(denoised)

        # 2. 맵 계산 (Error map, Noise map)
        noise_map = np.abs(noisy_img - clean_img)
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min() + 1e-8)

        residual_map = np.abs(clean_img - denoised_img)
        # 잘 보이게 Contrast 3배 강조 (선택 사항)
        residual_map = np.clip(residual_map * 3, 0, 1) 

        # 3. Plot 생성
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        titles = ["GT", "Noisy", "Noise Pattern", "Result", "Error Map"]
        images = [clean_img, noisy_img, noise_map, denoised_img, residual_map]
        cmaps = [None, None, 'gray', None, 'inferno']

        for ax, img, title, cmap in zip(axes, images, titles, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()

        # 4. WandB 로깅
        try:
            # WandbLogger인지 확인하고 로깅
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    f"{key}/epoch_{self.current_epoch}": [wandb.Image(fig, caption=f"PSNR: {-10*np.log10(((clean_img-denoised_img)**2).mean()):.2f}dB")]
                })
        except Exception as e:
            print(f"Logging failed: {e}")
        finally:
            plt.close(fig) # 메모리 정리

    def _log_mc_confidence_map(self, y, batch_idx, target_batch=349, num_samples=8):
        if batch_idx != target_batch:
            return
        
        # ----------------------------------------------------
        # 1. Teacher MC Confidence (현재 생명줄)
        # ----------------------------------------------------
        mc_preds = []
        alpha = float(self.cfg.r2r.alpha)
        pad_size = 16
        
        with torch.no_grad():
            for i in range(num_samples):
                aug_idx = i % 8
                y_aug = self.tta_forward(y, aug_idx)
                y_sub_aug = pixel_unshuffle(y_aug, 2)
                y_aug_pad = F.pad(y_sub_aug, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
                
                # Heuristic sigma는 항상 계산
                sigma_map, _ = estimate_sigma_structure_aware(y_aug_pad)
                
                # Learnable estimator 분기
                if self.cfg.r2r.use_learned_estimator:
                    h_var_aug = sigma_map ** 2
                    log_var = self.noise_estimator(y_sub_aug.detach(), h_var_aug.detach())
                    sigma_map_aug = torch.exp(0.5 * log_var)
                else:
                    sigma_map_aug = sigma_map.detach()
                
                root_noise = torch.randn_like(y_aug_pad)
                heavy_noise = root_noise * (sigma_map_aug * 0.1)
                y1 = y_aug_pad + heavy_noise
                
                pred_residual = self.teacher(y1.clamp(0, 1))
                pred_pad = (pred_residual + y1).clamp(0, 1)
                pred = pred_pad[..., pad_size:-pad_size, pad_size:-pad_size]
                assembled_pred = pixel_shuffle(pred, 2)
                restored_pred = self.tta_backward(assembled_pred, aug_idx)
                mc_preds.append(restored_pred.detach())
        
        mc_stack = torch.stack(mc_preds)
        mc_variance = mc_stack.var(dim=0, unbiased=False)
        
        # 공간적 평활화 (Smoothing) 적용
        confidence = 1.0 / (1.0 + mc_variance * self.cfg.solver.conf_scale)
        var_99th = torch.quantile(mc_variance.view(-1), 0.99)
        self.log("debug/vis_image_var_99th", var_99th, on_step=False, on_epoch=True)
        
        # ----------------------------------------------------
        # 2. 시각화를 위한 텐서 -> Numpy 변환
        # ----------------------------------------------------
        if confidence.size(1) > 1:
            conf_1ch = confidence.mean(dim=1).squeeze().cpu().numpy()
        else:
            conf_1ch = confidence.squeeze().cpu().numpy()
        
        mc_mean_pred = mc_stack[0]
        src_rgb = mc_mean_pred[0].cpu().permute(1, 2, 0).numpy()
        src_rgb = np.clip(src_rgb, 0.0, 1.0)
        
        # ----------------------------------------------------
        # 3. Matplotlib Plotting & Logging
        # ----------------------------------------------------
        fig_conf, ax_conf = plt.subplots(figsize=(8, 8))
        im_conf = ax_conf.imshow(conf_1ch, cmap='viridis', vmin=0.0, vmax=1.0)
        ax_conf.axis('off')
        fig_conf.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
        
        fig_pred, ax_pred = plt.subplots(figsize=(8, 8))
        im_pred = ax_pred.imshow(src_rgb)
        ax_pred.axis('off')
        
        # Caption에 estimator 사용 여부 표시 (디버깅 편의)
        est_tag = "EST_ON" if self.cfg.r2r.use_learned_estimator else "EST_OFF"
        
        try:
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    "Debug_Visuals/1_Confidence": [
                        wandb.Image(fig_conf, caption=f"[{est_tag}] Epoch {self.current_epoch} | Smoothed Conf | Var 99th: {var_99th:.6f}")
                    ],
                    "Debug_Visuals/2_MC_Mean_Prediction": [
                        wandb.Image(fig_pred, caption=f"[{est_tag}] Epoch {self.current_epoch} | Teacher MC Mean Output")
                    ]
                })
        except Exception as e:
            print(f"Visualization logging failed: {e}")
        finally:
            plt.close(fig_conf)
            plt.close(fig_pred)


def build_run_name(cfg, version="v1"):
    # args 대신 cfg 객체를 받아서 이름 생성
    return (
        f"{cfg.model.name}_"
        f"a{cfg.r2r.alpha:.2f}_"
        f"img{cfg.data.img_size}_"
        f"stride{cfg.r2r.pd_stride}_"
        f"bs{cfg.data.batch_size}_"
        f"ep{cfg.train.max_epochs}_"
        f"mc{cfg.r2r.mc_samples}_"
        f"{version}"
    )

def main():
    p = argparse.ArgumentParser()
    
    # 1. 런타임 인자 (YAML에 넣기 애매한 실행 시점의 변수들만 남김)
    p.add_argument("--config", type=str, default="exp1.yaml", help="Path to config file")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--version", type=str, default="v1", help="Experiment version string")
    p.add_argument("--wandb_project", type=str, default="gr2r-denoising")
    p.add_argument("--ckpt_dir", type=str, default="./ckpts_gr2r")

    
    args = p.parse_args()

    # 2. Config YAML 파일 로드
    cfg = OmegaConf.load(args.config)
    
    # (선택) 터미널 명령어로 YAML 설정 덮어쓰기 허용 (예: python main.py --config exp1.yaml r2r.alpha=0.1)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    print(f"🚀 Starting Experiment with Config: {args.config}")
    print(OmegaConf.to_yaml(cfg)) # 로드된 설정 터미널에 출력

    # 3. 경로 설정
    train_dir = Path("/hyemin/nas/datasets/prep/SIDD_s512_o128")
    val_dir = Path("/hyemin/nas/datasets/SIDD")
    
    assert train_dir.exists(), f"Missing: {train_dir}"
    assert val_dir.exists(), f"Missing: {val_dir}"

    # 4. 데이터셋 & 데이터로더 설정 (cfg.data 참조)
    train_ds = SIDDTrainDataset(
        root_dir=train_dir,
        crop_size=cfg.data.img_size
    )
    val_ds = SIDDValidationDataset(
        noisy_file_path=val_dir / "ValidationNoisyBlocksSrgb.mat",
        gt_file_path=val_dir / "ValidationGtBlocksSrgb.mat"
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True
    )

    # 5. 모델 초기화 (cfg.r2r, cfg.train 등 참조)
    # ※ 만약 PD_GR2R __init__이 cfg 객체 통째로 받도록 수정되었다면 
    # lit_model = PD_GR2R(cfg) 로 한 줄로 끝낼 수 있습니다.
    # 아래는 기존 __init__ 파라미터에 맞춰서 넘겨주는 방식입니다.
    lit_model = PD_GR2R(cfg)

    # if cfg.model.checkpoint and str(cfg.model.checkpoint).lower() != "none":
    #     print(f"[init_weights] loading from: {cfg.model.checkpoint}")
    #     load_init_weights(lit_model.student, cfg.model.checkpoint, strict=True)
    #     lit_model.teacher.load_state_dict(lit_model.student.state_dict())
    #     lit_model.feature.load_state_dict(lit_model.student.state_dict())

    if cfg.model.checkpoint and str(cfg.model.checkpoint).lower() != "none":
        print(f"[init_weights] loading from: {cfg.model.checkpoint}")
        load_init_weights(lit_model.feature, cfg.model.checkpoint, strict=True)


    # train_cr2r.py main()에 추가 (load 직후)
    load_init_weights(lit_model.student, cfg.model.checkpoint, strict=True)
    
    # # 디버그
    # sd = lit_model.student.state_dict()
    # first_key = list(sd.keys())[0]
    # print(f"Student first key: {first_key}")
    # print(f"Student first weight norm: {sd[first_key].norm():.4f}")

    # ckpt = torch.load(cfg.model.checkpoint, map_location="cpu")
    # if "state_dict" in ckpt:
    #     print(f"Ckpt keys (first 5): {list(ckpt['state_dict'].keys())[:5]}")

    # 6. 콜백 및 로거 설정
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-psnr-{step:05d}-{student_psnr:.2f}", 
            monitor="student_psnr",
            mode="max",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-ssim-{step:05d}-{student_ssim:.2f}", 
            monitor="teacher_psnr", 
            mode="max",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # wandb_name = build_run_name(cfg, version=args.version)
    wandb_name = ckpt_dir.name

        
    logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_name,
        save_dir=str(ckpt_dir),   
        log_model=False,
        # W&B에 YAML config 전체를 업로드해서 나중에 하이퍼파라미터 추적하기 쉽게 만들기
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=True,
        settings=wandb.Settings(code_dir=os.getcwd()),
    )

    # 7. 트레이너 실행
    trainer = L.Trainer(
        max_epochs=-1,
        max_steps=cfg.solver.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        val_check_interval=100,
        check_val_every_n_epoch=None,
        devices=1,
        # gradient_clip_val=1.0,       
        # gradient_clip_algorithm="norm", 
        callbacks=callbacks,
        log_every_n_steps=20,
        enable_progress_bar=True,
        logger=logger,
    )
    # trainer.validate(lit_model, val_loader, ckpt_path=args.resume)
    # args.resume = '/hyemin/denoising/ckpts/pcr2r/w_uda1.0_w_src1.0_m0.99_unet_ign_src_inter-gram1e6-l1_phase_test/best-psnr-step=11300-student_psnr=36.65.ckpt'
    # trainer.validate(lit_model, val_loader, ckpt_path=args.resume)
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
