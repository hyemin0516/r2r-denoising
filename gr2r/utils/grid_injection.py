import torch

def get_grid_noise_map(img, stride=5, intensity_range=(0.01, 0.05)):
    """
    (B, C, H, W) 이미지 크기에 맞는 격자 노이즈 맵 생성
    """
    b, c, h, w = img.shape
    device = img.device
    
    # 1. 기본 커널 (2x2 Checkerboard)
    kernel_base = torch.tensor([[0, 1], [1, 0]], device=device, dtype=img.dtype)
    k_h, k_w = kernel_base.shape # 2, 2
    
    # 2. 반복 횟수 계산 (Safe Repeat)
    # 이미지를 덮으려면 (이미지크기 / 커널크기) 보다 조금 더 많이 반복해야 함
    # +1을 해줘서 무조건 이미지보다 크게 만듭니다.
    rep_h = (h // k_h) + 1
    rep_w = (w // k_w) + 1
    
    # 3. Tiling & Slicing (핵심!)
    mask = kernel_base.repeat(rep_h, rep_w) # 일단 크게 만듦
    mask = mask[:h, :w] # 정확히 이미지 크기만큼 잘라냄 (Crop)
    
    # 4. 차원 확장
    # 이제 mask shape은 (h, w)가 보장되므로 view에서 에러가 안 납니다.
    mask = mask.view(1, 1, h, w)
    
    # 5. Random Alpha & Sign 적용
    alpha = torch.empty(b, c, 1, 1, device=device).uniform_(*intensity_range)
    sign = torch.randint(0, 2, (b, c, 1, 1), device=device).float() * 2 - 1
    
    grid_map = mask * alpha * sign
    
    return grid_map

# ----------------------------------------------------------
def estimate_frequency_sigma(y, kernel_size=5):
    """
    [Contribution 핵심]
    주파수 분리(Frequency Separation)를 이용한 강건한 노이즈 레벨 추정
    - 구조(Structure)와 노이즈(Noise)를 분리하여, 텍스처를 노이즈로 오인하거나 
      노이즈를 엣지로 오인하는 문제를 해결함.
    """
    b, c, h, w = y.shape
    pad = kernel_size // 2
    
    # 1. Low-Pass Filtering (구조 성분 추출)
    # 간단한 Box Filter나 Gaussian Filter를 사용해 이미지를 뭉갭니다.
    # 여기서는 계산 효율성을 위해 AvgPool을 사용하여 부드러운 배경을 만듭니다.
    # (커널 사이즈가 클수록 더 큰 구조만 남습니다)
    y_low = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=pad)
    
    # 2. High-Pass Filtering (노이즈 + 미세 텍스처 추출)
    # 원본에서 저주파를 빼면 순수한 '변동성'만 남습니다.
    # 배경색이나 밝기 편차가 제거된 상태입니다.
    y_high = y - y_low
    
    # 3. HPF 영역에서 Local MAD 계산
    # 이제 y_high 위에서 MAD를 구합니다.
    # 이전 함수(estimate_sigma_mad_map)의 로직을 여기에 적용합니다.
    
    # (1) Unfold로 패치 뜯기 (Local Statistics)
    stride = 1 # 촘촘하게 봅니다
    # unfold shape: (B, C*K*K, L) where L = H*W
    y_patches = F.unfold(y_high, kernel_size=3, padding=1, stride=stride)
    y_patches = y_patches.view(b, c, 9, h, w) # 3x3=9 neighbors
    
    # (2) MAD 계산
    # 중앙값 (High freq 성분의 중심은 0에 가까움)
    median = torch.median(y_patches, dim=2, keepdim=True).values
    # 절대 편차
    diff = torch.abs(y_patches - median)
    # MAD
    mad = torch.median(diff, dim=2).values.squeeze(2) # (B, C, H, W)
    
    # (3) Sigma 추정
    sigma_est = mad / 0.6745
    
    # 4. Texture Suppression (텍스처 보호 - 여기가 중요!)
    # y_high의 절댓값이 유난히 크다면, 그건 노이즈가 아니라 '강한 엣지'입니다.
    # 노이즈는 가우시안 분포를 따르지만, 엣지는 Outlier처럼 튑니다.
    
    # 엣지 에너지 계산
    local_energy = torch.mean(y_patches.abs(), dim=2) # (B, C, H, W)
    
    # 에너지가 너무 크면(텍스처) Sigma를 낮춰서 보호해야 합니다.
    # Weighting Function: 에너지가 클수록 0에 가깝게, 작으면 1에 가깝게
    # 민감도(sensitivity) 조절: 이 값이 클수록 텍스처를 더 강하게 보호
    sensitivity = 5.0 
    texture_weight = torch.exp(-local_energy * sensitivity)
    
    # 5. 최종 맵 합성
    final_sigma = sigma_est * texture_weight
    
    # 최소한의 노이즈 바닥(Floor) 설정 (Identity 방지)
    # 아주 매끈한 영역이라도 최소한 0.005(약 1/255) 정도는 지우게 함
    return final_sigma.clamp(min=0.005)
