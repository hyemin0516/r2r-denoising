import torch
import torch.nn as nn
import torch.nn.functional as F

class PreWhitener(nn.Module):
    """
    y_tilde = clamp(y + g(y), 0, 1)
    g(y)는 depthwise conv 기반의 아주 가벼운 residual 필터.
    초기 weight=0 -> 시작은 거의 identity.
    """
    def __init__(self, channels: int, ksize: int = 3):
        super().__init__()
        pad = ksize // 2
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=ksize,
            padding=pad, groups=channels, bias=True
        )
        # identity init: weight=0, bias=0
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

    def forward(self, y):
        return torch.clamp(y + self.dw(y), 0.0, 1.0)


def loss_identity(y_tilde, y):
    # L1이 색/톤 유지에 안정적
    return (y_tilde - y).abs().mean()

def loss_whiten(y_tilde, lpf_ksize=5):
    # LPF: 평균풀링(아주 간단/안정)
    pad = lpf_ksize // 2
    y_lpf = F.avg_pool2d(y_tilde, kernel_size=lpf_ksize, stride=1, padding=pad)
    r = y_tilde - y_lpf  # high-pass residual

    # 인접 픽셀 상관(수평/수직). 절댓값으로 상관 크기만 줄이기
    corr_h = (r[:, :, :, 1:] * r[:, :, :, :-1]).mean().abs()
    corr_v = (r[:, :, 1:, :] * r[:, :, :-1, :]).mean().abs()
    return corr_h + corr_v