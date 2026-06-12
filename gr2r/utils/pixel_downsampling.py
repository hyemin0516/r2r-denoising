import torch
import torch.nn as nn
import torch.nn.functional as F

def pixel_unshuffle(x, stride=2):
    """
    (B, C, H, W) -> (B * stride^2, C, H/stride, W/stride)
    Batch 차원으로 Sub-image를 쌓아버리는 방식입니다.
    """
    b, c, h, w = x.shape
    # 1. Reshape to separate stride blocks
    x = x.view(b, c, h // stride, stride, w // stride, stride)
    # 2. Permute to bring stride dimensions together
    # (B, stride, stride, C, H/s, W/s)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    # 3. Merge into Batch dimension
    # (B * s * s, C, H/s, W/s)
    x = x.view(b * stride * stride, c, h // stride, w // stride)
    return x

def pixel_shuffle(x, stride=2):
    """
    (B * stride^2, C, H/stride, W/stride) -> (B, C, H, W)
    위 과정을 역으로 수행하여 원본 해상도로 복원합니다.
    """
    b_new, c, h_new, w_new = x.shape
    b = b_new // (stride * stride)
    
    x = x.view(b, stride, stride, c, h_new, w_new)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    x = x.view(b, c, h_new * stride, w_new * stride)
    return x