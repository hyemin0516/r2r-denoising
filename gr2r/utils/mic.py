import torch
import torch.nn.functional as F

def mic(model, y_original, ratio=0.05, patch_size=2): # ratio를 5%로, patch_size를 2~4로 대폭 축소
    y_masked, mask = apply_mask_noise_fill(y_original, ratio, patch_size)
    
    residual = model(y_masked)
    pred_mic = residual + y_masked
    pred_mic = torch.clamp(pred_mic, 0, 1)
    
    # 마스킹된 영역(inv_mask)에서만 loss 계산
    inv_mask = 1.0 - mask
    
    return pred_mic, inv_mask

def apply_mask_noise_fill(x, ratio=0.05, patch_size=2):
    B, C, H, W = x.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    
    # 1. 마스크 생성
    rand_tensor = torch.rand(B, 1, h_patches, w_patches, device=x.device)
    mask = (rand_tensor > ratio).float()
    mask = F.interpolate(mask, size=(H, W), mode='nearest')
    
    # =========================================================
    # [수석 연구원의 수정 포인트: Shuffled Patch 제거, Pure Noise 채우기]
    # =========================================================
    # 단순히 0으로 채우는 대신, 현재 이미지의 노이즈 스케일과 유사한 
    # 가우시안 노이즈(결측치)를 채워 CNN의 엣지 쇼크를 방지합니다.
    # (이미지가 0~1로 정규화되어 있다면 randn 텍스처를 살짝 입힙니다)
    
    noise_fill = torch.randn_like(x) * x.std() + x.mean() 
    
    # 마스킹 영역(0인 곳)은 가우시안 노이즈로, 나머지는 원본으로 유지
    x_masked = x * mask + noise_fill * (1 - mask)
    
    return x_masked, mask