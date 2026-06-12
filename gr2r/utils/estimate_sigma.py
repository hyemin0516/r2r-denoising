import torch
import torch.nn as nn
import torch.nn.functional as F

def estimate_sigma_mad(img_sub):
    """
    Robust Noise Level Estimation using MAD (Median Absolute Deviation)
    입력: PD가 적용된 Sub-image (B*s^2, C, H/s, W/s) - 노이즈가 독립적이라고 가정
    출력: 추정된 Sigma 값 (B,)
    """
    # 1. 고주파 성분 추출 (High-pass filtering)
    # 이미지의 텍스처를 배제하고 노이즈만 남기기 위해 인접 픽셀과 뺄셈을 합니다.
    # (B, C, H, W-1)
    diff = img_sub[..., 1:] - img_sub[..., :-1] 
    
    # 2. MAD 계산
    # 공식: sigma = median(|x - median(x)|) / 0.6745
    # 정규분포(Gaussian) 가정 하에 표준편차의 근사값을 구합니다.
    
    # 배치를 제외한 나머지 차원으로 평탄화 (B, -1)
    diff_flat = diff.view(diff.size(0), -1)
    
    # 중앙값(Median) 계산 (각 배치별로)
    median_val = torch.median(diff_flat, dim=1, keepdim=True).values
    
    # 편차의 절댓값의 중앙값
    mad = torch.median(torch.abs(diff_flat - median_val), dim=1).values
    
    # 정규분포 상수(0.6745)로 보정하여 Sigma 도출
    sigma_est = mad / 0.6745
    
    return sigma_est

def estimate_sigma_mad_map(img, stride=2):
    """
    Robust Noise Level Map Estimation using MAD (Spatially Local)
    
    [수정 사항]
    입력: 원본 해상도 이미지 (B, C, H, W)
    출력: 추정된 Sigma Map (B, C, H, W) - 입력과 동일한 크기로 복원됨
    """
    b, c, h, w = img.shape

    # 1. 이웃 픽셀 모으기 (Unfold / Pixel Unshuffle)
    # (B, C, H, W) -> (B, C, H//s, s, W//s, s)로 차원을 쪼갭니다.
    # 이렇게 해야 H와 W를 stride로 나눈 '패치' 개념이 생깁니다.
    img_unfolded = img.view(b, c, h // stride, stride, w // stride, stride)

    # 2. 이웃끼리 마지막 차원으로 몰아넣기
    # (B, C, H//s, W//s, s, s) -> (B, C, H//s, W//s, s*s)
    # 이제 마지막 차원(dim=-1)에 4개(stride=2일 때)의 이웃 픽셀이 모입니다.
    img_neighbors = img_unfolded.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h // stride, w // stride, stride**2)

    # 3. 이웃 간의 MAD 계산 (Local Estimation)
    
    # 이웃들의 중앙값 (Signal 추정) -> dim=-1 기준
    median_val = torch.median(img_neighbors, dim=-1, keepdim=True).values
    
    # (값 - 중앙값)의 절댓값
    diff = torch.abs(img_neighbors - median_val)
    
    # MAD 계산 (중앙값의 중앙값)
    mad = torch.median(diff, dim=-1, keepdim=True).values
    
    # 4. Sigma 도출
    # (B, C, H//s, W//s, 1) -> (B, C, H//s, W//s)
    sigma_low_res = (mad / 0.6745).squeeze(-1)

    # 5. 원래 해상도로 복원 (Upsample)
    # (B, C, H//s, W//s) -> (B, C, H, W)
    # Nearest로 늘려야 '패치 단위'의 노이즈 레벨이 유지됩니다.
    sigma_map = F.interpolate(sigma_low_res, size=(h, w), mode='nearest')
    
    return sigma_map

def estimate_sigma_structure_aware(y, window_size=7, edge_gate_min=0.5):
    B, C, H, W = y.shape

    def avg_pool_reflect(x, kernel_size):
        pad = kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=0)

    y_ref = F.pad(y, (1, 1, 1, 1), mode='reflect')

    laplacian_kernel = torch.tensor(
        [[ 1.,  2.,  1.],
         [ 2., -12., 2.],
         [ 1.,  2.,  1.]],
        device=y.device,
        dtype=y.dtype
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    high_freq = F.conv2d(y_ref, laplacian_kernel, padding=0, groups=C)

    sobel_x = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=y.device,
        dtype=y.dtype
    )
    sobel_y = sobel_x.T

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    gx = F.conv2d(y_ref, sobel_x, padding=0, groups=C)
    gy = F.conv2d(y_ref, sobel_y, padding=0, groups=C)

    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    grad_q95 = grad_mag.flatten(2).quantile(0.95, dim=2)[..., None, None]
    grad_norm = (grad_mag / (grad_q95 + 1e-8)).clamp(0, 1)

    structure_weight = torch.exp(-5.0 * grad_norm)

    edge_weight_soft = torch.exp(-2.0 * grad_norm)
    edge_gate = edge_gate_min + (1.0 - edge_gate_min) * edge_weight_soft
    edge_gate = edge_gate.clamp(edge_gate_min, 1.0)

    weighted_var = avg_pool_reflect(
        high_freq ** 2 * structure_weight,
        window_size
    )

    weight_sum = avg_pool_reflect(
        structure_weight,
        window_size
    )

    local_variance = weighted_var / (weight_sum + 1e-8)

    correction_factor = 164.0
    sigma_map = torch.sqrt(
        torch.clamp(local_variance / correction_factor, min=1e-8)
    )

    return sigma_map, edge_gate

def estimate_smart_sigma(y, window_size=7, edge_sensitivity=1.0):
    """
    Self-supervised 상황에서 입력 이미지 y의 공간적 노이즈 레벨(sigma map)을 추정합니다.
    
    Args:
        y (Tensor): 입력 노이즈 이미지 (B, C, H, W)
        window_size (int): 국소 노이즈를 추정할 윈도우 크기 (보통 5 또는 7)
        edge_sensitivity (float): 엣지 영역의 과대 추정을 방지하기 위한 민감도 조절 파라미터
    Returns:
        sigma_map (Tensor): 픽셀별 노이즈 레벨이 추정된 맵 (B, C, H, W)
    """
    B, C, H, W = y.shape
    
    # 1. Laplacian 필터로 저주파(구조적 배경) 제거 및 고주파(노이즈+엣지) 추출
    # 표준 3x3 라플라시안 커널 적용
    laplacian_kernel = torch.tensor([[ 1.,  2.,  1.],
                                     [ 2., -12., 2.],
                                     [ 1.,  2.,  1.]], device=y.device, dtype=y.dtype)
    # (Out_C, In_C/groups, kH, kW) 형태로 변환
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    
    # 패딩을 주어 크기 유지 (groups=C 로 채널별 독립 연산)
    high_freq = F.conv2d(y, laplacian_kernel, padding=1, groups=C)
    
    # 2. 국소 영역의 분산(Variance) 추정
    # high_freq의 제곱을 구한 뒤, window_size 만큼 Average Pooling을 하여 국소 분산을 구함
    local_variance = F.avg_pool2d(high_freq ** 2, kernel_size=window_size, stride=1, padding=window_size//2)
    
    # 3. 라플라시안 필터의 통계적 보정 계수 적용
    # 라플라시안 커널을 통과한 가우시안 노이즈의 분산은 원래 분산의 약 34배(위 커널 기준)가 됨
    # 이를 역산하여 원래의 sigma 값을 복원 (수학적 증명 기반)
    # 상수 보정 (엣지에서의 과대 추정을 edge_sensitivity로 조절)
    correction_factor = 34.0 * edge_sensitivity 
    
    # 분산에서 표준편차(sigma)로 변환
    sigma_map = torch.sqrt(torch.clamp(local_variance / correction_factor, min=1e-8))

    # 각 이미지의 전반적인 베이스라인 노이즈(중간값) 추출
    global_median = torch.median(sigma_map.view(B, -1), dim=1).values
    global_median = global_median.view(B, 1, 1, 1)

    # 글씨(Edge) 영역의 과도한 sigma 값을 베이스라인의 1.5배로 제한
    sigma_map = torch.clamp(sigma_map, max=global_median * 1.5)
    
    return sigma_map

def _grad_mag(y: torch.Tensor) -> torch.Tensor:
    # y: (B,C,H,W) in [0,1]
    # Sobel-like gradients (채널별)
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=y.device, dtype=y.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=y.device, dtype=y.dtype).view(1,1,3,3)
    B,C,H,W = y.shape
    kx = kx.repeat(C,1,1,1)
    ky = ky.repeat(C,1,1,1)
    gx = F.conv2d(y, kx, padding=1, groups=C)
    gy = F.conv2d(y, ky, padding=1, groups=C)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def estimate_sigma(y: torch.Tensor,
                        window: int = 7,
                        flat_q: float = 0.3,
                        smooth_ksize: int = 7,
                        eps: float = 1e-8):
    """
    SIDD용 sigma_map 추정:
      1) gradient 낮은 영역만 사용(텍스처 배제)
      2) local mean/var로 (a*mu + b) 모델 robust fit
      3) sigma_map = sqrt(a*mu + b), smoothing + clip
    Args:
      y: (B,C,H,W), [0,1]
      window: local stats 윈도우
      flat_q: gradient 하위 q 비율만 flat으로 사용
      smooth_ksize: sigma_map smoothing용 평균풀 커널
    Returns:
      sigma_map: (B,1,H,W)  (채널 공통으로 쓰는 버전)
    """
    B,C,H,W = y.shape

    # (1) luminance(채널 통합)으로 노이즈 레벨 추정이 더 안정적인 경우가 많음
    #     SIDD sRGB면 대체로 OK. RAW면 채널 분리 버전도 가능.
    if C == 3:
        mu_img = 0.299*y[:,0:1] + 0.587*y[:,1:2] + 0.114*y[:,2:3]
    else:
        mu_img = y.mean(dim=1, keepdim=True)

    # (2) local mean/var
    pad = window//2
    local_mean = F.avg_pool2d(mu_img, window, stride=1, padding=pad)
    local_mean2 = F.avg_pool2d(mu_img*mu_img, window, stride=1, padding=pad)
    local_var = torch.clamp(local_mean2 - local_mean*local_mean, min=0.0)

    # (3) flat mask: gradient 작은 곳만
    g = _grad_mag(mu_img)  # (B,1,H,W)
    # 배치별 threshold: 하위 flat_q quantile
    g_flat = g.view(B, -1)
    thr = torch.quantile(g_flat, flat_q, dim=1).view(B,1,1,1)
    mask = (g <= thr).float()  # flat 영역 1

    # (4) (mu, var) 샘플로 a,b 추정: var ≈ a*mu + b
    #     robust를 위해 clamp & weighted mean 사용
    #     (간단 버전: 가중 최소제곱의 closed-form)
    mu_s = local_mean
    var_s = local_var

    w = mask
    # sums
    S0 = (w).sum(dim=(2,3)) + eps
    S1 = (w*mu_s).sum(dim=(2,3))
    S2 = (w*mu_s*mu_s).sum(dim=(2,3))
    T0 = (w*var_s).sum(dim=(2,3))
    T1 = (w*mu_s*var_s).sum(dim=(2,3))

    # solve [S2 S1; S1 S0] [a;b] = [T1;T0]
    det = (S2*S0 - S1*S1) + eps
    a = ( T1*S0 - T0*S1) / det
    b = ( T0*S2 - T1*S1) / det

    # 안정화: 음수 방지
    a = torch.clamp(a, min=0.0)
    b = torch.clamp(b, min=0.0)

    # (5) sigma_map 생성
    sigma2 = a.view(B,1,1,1)*mu_img + b.view(B,1,1,1)
    sigma_map = torch.sqrt(torch.clamp(sigma2, min=1e-8))

    # (6) smoothing + clip
    if smooth_ksize > 1:
        sp = smooth_ksize//2
        sigma_map = F.avg_pool2d(sigma_map, smooth_ksize, stride=1, padding=sp)

    # clip: 중앙값 기준 과도한 값 제한
    med = torch.median(sigma_map.view(B, -1), dim=1).values.view(B,1,1,1)
    sigma_map = torch.clamp(sigma_map, max=med*2.0)

    return sigma_map

# class LearnableNoiseEstimator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Poisson-Gaussian 파라미터: 초기에는 0에 가깝게 설정하여 
#         # 기존 휴리스틱 값을 그대로 따르도록 (Warm-start) 유도
#         self.beta_1 = nn.Parameter(torch.full((1, 3, 1, 1), 1e-6)) # 신호 의존적 계수
#         self.beta_2 = nn.Parameter(torch.full((1, 3, 1, 1), 1e-6)) # 가우시안 기저 계수
        
#         # 기존 알고리즘의 반영 비중 (초기값 1.0)
#         self.heuristic_weight = nn.Parameter(torch.tensor(1.0))

#     def forward(self, y, heuristic_variance):
#         """
#         y: 입력 이미지 (Poisson 성분 추정용)
#         heuristic_variance: 기존 estimate_sigma_structure_aware 함수가 뱉은 분산 값
#         """
#         # 1. 학습 가능한 Poisson-Gaussian 분산 모델
#         # Variance = beta_1 * y + beta_2
#         pg_variance = self.beta_1 * y + self.beta_2
#         # pg_variance = F.softplus(self.beta_1) * y + F.softplus(self.beta_2)
        
#         # 2. 잔차 결합 구조
#         # 초기에는 pg_variance가 거의 0이므로 heuristic_variance가 100% 지배함
#         # hw = torch.sigmoid(self.heuristic_weight) * 2.0
#         # hw = 0.5 + torch.sigmoid(self.heuristic_weight)
#         combined_variance = (self.heuristic_weight * heuristic_variance) + pg_variance
        
#         # 3. 최종 Sigma Map (안전장치 clamp 포함)
#         sigma_map = torch.sqrt(torch.clamp(combined_variance, min=1e-8))
#         return sigma_map

class LearnableNoiseEstimator(nn.Module):
    """
    Log-space residual estimator.
    - heuristic variance를 base로, 보정값(delta)만 학습
    - softplus/sigmoid 없음 — log space에서 unconstrained
    - 마지막 layer zero-init → 초기 출력 = heuristic 그대로
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),   # delta in log-space
        )
        # zero-init: delta=0 → 처음엔 heuristic 100% 의존
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, heuristic_var):
        """
        x: noisy input sub-patch [B, 3, H, W]
        heuristic_var: estimate_sigma_structure_aware의 sigma² [B, 1 or 3, H, W]
        Returns: log_variance (unconstrained)
        """
        log_heuristic = torch.log(heuristic_var + 1e-8)
        delta = self.net(x)                    # 보정값 (양수/음수 자유)
        log_var = log_heuristic + delta        # log(최종분산) = log(heuristic) + delta
        return log_var


# ============================================================
# Heteroscedastic Poisson-Gaussian Sigma Estimator (사전학습용)
#   - Var(y|x) = a*mu + b  를 IRLS + Huber 로 픽셀당 회귀
#   - Flat-region sampling: multi-scale gradient + saturation guard
#   - ACF correction: SIDD spatially-correlated noise 보상
#   - Full-resolution 추정 후 PD-domain broadcast 헬퍼 제공
# ============================================================

def _sobel_kernels(C, device, dtype):
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    return kx, ky


def _wls_fit(mu_s, var_s, w, eps=1e-8):
    """Weighted Least Squares for var = a*mu + b, per-batch closed-form."""
    B = mu_s.size(0)
    S0 = w.sum(dim=(1, 2, 3)) + eps
    S1 = (w * mu_s).sum(dim=(1, 2, 3))
    S2 = (w * mu_s * mu_s).sum(dim=(1, 2, 3))
    T0 = (w * var_s).sum(dim=(1, 2, 3))
    T1 = (w * mu_s * var_s).sum(dim=(1, 2, 3))
    det = (S2 * S0 - S1 * S1) + eps
    a = (T1 * S0 - T0 * S1) / det
    b = (T0 * S2 - T1 * S1) / det
    return a, b


def _estimate_acf_correction(mu_img, flat_mask, eps=1e-8):
    """
    Self-calibrating ACF correction (도메인 비의존).

    flat region 안에서 노이즈의 1-step 자기상관 rho_x, rho_y 를 측정.
    AWGN(i.i.d.) → rho≈0 → correction≈1.0
    SIDD/DND(ISP-correlated) → rho>0 → correction<1.0 (분산이 deflate된 만큼 보정)

    measured_var ≈ true_var * (1 - rho_x) * (1 - rho_y) 의 근사
    (인접 픽셀이 양의 상관을 가지면 (y_i - mean)^2 평균이 작아짐)
    """
    B = mu_img.size(0)
    # local mean 제거 (window=5 근사)
    lm = F.avg_pool2d(mu_img, kernel_size=5, stride=1, padding=2)
    n = mu_img - lm  # 노이즈 추정 (flat region 가정)

    # 1-step shifted 곱 (수평/수직)
    nx = n[..., :, 1:] * n[..., :, :-1]
    ny = n[..., 1:, :] * n[..., :-1, :]
    nn = n * n

    # flat mask로 가중평균 (배경/엣지 픽셀 배제)
    mx = flat_mask[..., :, 1:] * flat_mask[..., :, :-1]
    my = flat_mask[..., 1:, :] * flat_mask[..., :-1, :]

    num_x = (nx * mx).sum(dim=(1, 2, 3))
    num_y = (ny * my).sum(dim=(1, 2, 3))
    den_x = (nn[..., :, 1:] * mx).sum(dim=(1, 2, 3)) + eps
    den_y = (nn[..., 1:, :] * my).sum(dim=(1, 2, 3)) + eps

    rho_x = (num_x / den_x).clamp(0.0, 0.95)
    rho_y = (num_y / den_y).clamp(0.0, 0.95)

    # effective correction = (1 - rho_x) * (1 - rho_y)  (Wiener-Khinchin 근사)
    correction = ((1.0 - rho_x) * (1.0 - rho_y)).clamp(0.15, 1.0)
    return correction  # (B,)


def estimate_sigma_pg_hetero(y,
                             window=15,
                             flat_q=0.25,
                             smooth_ksize=9,
                             acf_correction=None,
                             irls_iters=2,
                             per_channel=True,
                             saturation_range=(0.02, 0.98),
                             eps=1e-8):
    """
    Poisson-Gaussian heteroscedastic sigma estimator (no learnable params).

    Args:
        y: (B, C, H, W), [0,1]   -- full-resolution noisy image (PD 적용 전)
        window: local mean/var window (>=11 권장; SIDD는 15~21)
        flat_q: 하위 q-quantile gradient 픽셀만 flat 으로 사용
        smooth_ksize: 최종 sigma map smoothing 커널
        acf_correction: None이면 데이터에서 자동 추정 (도메인 비의존). float이면 강제 사용.
                         AWGN≈1.0, SIDD≈0.5, DND≈0.7. None 권장.
        irls_iters: Huber-IRLS 반복 횟수 (2면 충분)
        per_channel: True면 채널별 독립 σ map 반환

    Returns:
        sigma_map: (B, C, H, W) — 픽셀별 sigma
        structure_weight: (B, 1, H, W) — flat=1, edge=0 (loss reweighting용)
    """
    B, C, H, W = y.shape

    # ------------------------------------------------------------------
    # 1) Luminance로 (μ, var) 추정 — 채널 평균은 SNR 최대
    # ------------------------------------------------------------------
    if C == 3:
        mu_img = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    else:
        mu_img = y.mean(dim=1, keepdim=True)

    pad = window // 2
    local_mean = F.avg_pool2d(mu_img, window, stride=1, padding=pad)
    local_mean2 = F.avg_pool2d(mu_img * mu_img, window, stride=1, padding=pad)
    local_var = (local_mean2 - local_mean * local_mean).clamp_min(0.0)

    # ------------------------------------------------------------------
    # 2) Multi-scale flat mask (3x3 sobel + 7x7 dilated 평균)
    #    medium-frequency texture (천 패턴, 글자 stem) 도 걸러짐
    # ------------------------------------------------------------------
    kx, ky = _sobel_kernels(1, y.device, y.dtype)
    gx = F.conv2d(mu_img, kx, padding=1)
    gy = F.conv2d(mu_img, ky, padding=1)
    grad_s = torch.sqrt(gx * gx + gy * gy + eps)
    grad_l = F.avg_pool2d(grad_s, kernel_size=7, stride=1, padding=3)
    grad = torch.maximum(grad_s, grad_l)

    g_flat = grad.view(B, -1)
    thr = torch.quantile(g_flat, flat_q, dim=1).view(B, 1, 1, 1)

    # 포화/암부는 분산 추정에서 제외 (도메인별 dynamic range 대응)
    smin, smax = saturation_range
    sat_mask = ((mu_img > smin) & (mu_img < smax)).float()
    flat_mask = (grad <= thr).float() * sat_mask

    # ------------------------------------------------------------------
    # 3) IRLS + Huber 로 (a, b) 강건 추정
    # ------------------------------------------------------------------
    mu_s = local_mean
    var_s = local_var
    w = flat_mask

    a, b = _wls_fit(mu_s, var_s, w, eps)

    for _ in range(irls_iters):
        pred_var = (a.view(B, 1, 1, 1) * mu_s + b.view(B, 1, 1, 1)).clamp_min(eps)
        # standardized residual
        r = (var_s - pred_var) / pred_var.clamp_min(eps).sqrt()
        # Huber weight (k=1.345)
        huber = 1.0 / (1.0 + (r.abs() / 1.345).clamp_min(1.0))
        w = flat_mask * huber
        a, b = _wls_fit(mu_s, var_s, w, eps)

    a = a.clamp_min(0.0)
    b = b.clamp_min(0.0)

    # ------------------------------------------------------------------
    # 4) ACF correction — spatially-correlated noise는 local var을 deflate시킴
    #    measured_var ≈ acf_correction * true_var  →  나눠서 복원
    #    acf_correction=None 이면 데이터에서 자동 추정 (도메인 비의존)
    # ------------------------------------------------------------------
    if acf_correction is None:
        acf_b = _estimate_acf_correction(mu_img, flat_mask, eps)  # (B,)
    else:
        acf_b = torch.full((B,), float(acf_correction), device=y.device, dtype=y.dtype)

    a = a / acf_b
    b = b / acf_b

    # ------------------------------------------------------------------
    # 5) Sigma map 합성 — per-channel intensity 사용
    # ------------------------------------------------------------------
    a_b = a.view(B, 1, 1, 1)
    b_b = b.view(B, 1, 1, 1)
    if per_channel:
        sigma2 = a_b * y + b_b               # (B, C, H, W)
    else:
        sigma2 = (a_b * mu_img + b_b).expand(-1, C, -1, -1)

    sigma_map = sigma2.clamp_min(eps).sqrt()

    if smooth_ksize > 1:
        sp = smooth_ksize // 2
        sigma_map = F.avg_pool2d(sigma_map, smooth_ksize, stride=1, padding=sp)

    # safety clip — 중앙값의 3배 이상은 텍스처 누수로 간주
    med = sigma_map.view(B, -1).median(dim=1).values.view(B, 1, 1, 1)
    sigma_map = sigma_map.clamp(max=med * 3.0)

    # structure weight (loss reweighting/시각화용)
    g_norm = grad / (grad.flatten(2).quantile(0.95, dim=2)[..., None, None] + eps)
    structure_weight = torch.exp(-5.0 * g_norm.clamp(0, 1))

    return sigma_map, structure_weight


def sigma_full_to_pd(sigma_map_full, stride):
    """
    Full-res sigma map → PD sub-image batch broadcast.

    프로젝트의 pixel_unshuffle 이 batch 확장형 (B → B·s²) 으로
    구현되어 있다는 전제. σ는 non-negative scalar 이므로
    동일 unshuffle 을 그대로 적용하면 sub-patch 정렬이 y_sub 와 100% 일치.

    sigma_map_full: (B, C, H, W)
    return:         (B·stride², C, H/stride, W/stride)
    """
    from .pixel_downsampling import pixel_unshuffle as _pu
    return _pu(sigma_map_full, stride)

def estimate_sigma_structure_aware_v2(y, window_size=7, intensity_gamma=0.5):
    """
    structure_aware의 안정성 + heteroscedastic intensity scaling 결합.
 
    핵심 변경:
    (1) 기존 Laplacian+structure_weight 메커니즘 그대로 유지 (검증된 안정성)
    (2) flat region luminance로 intensity-dependent scaling factor 계산
        σ²(p) = base_var × (1 + γ·μ(p))
        γ=0 이면 기존 structure_aware와 100% 동일 (safe fallback)
    (3) ACF auto-estimation 없음 — correction_factor 하나로 통제 (ablation-friendly)
 
    Args:
        y: (B, C, H, W), [0,1]
        window_size: local variance pooling 커널
        intensity_gamma: intensity 의존성 강도.
            0.0 = homoscedastic (기존 structure_aware 동일)
            0.5 = 약한 hetero (SIDD sRGB 권장)
            1.0 = 강한 hetero (raw sensor 용)
    """
    B, C, H, W = y.shape
 
    # ── 기존 structure_aware 로직 그대로 ──
    laplacian_kernel = torch.tensor([[ 1.,  2.,  1.],
                                     [ 2., -12., 2.],
                                     [ 1.,  2.,  1.]], device=y.device, dtype=y.dtype)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    high_freq = F.conv2d(y, laplacian_kernel, padding=1, groups=C)
 
    sobel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]], device=y.device, dtype=y.dtype)
    sobel_y = sobel_x.T
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
 
    gx = F.conv2d(y, sobel_x, padding=1, groups=C)
    gy = F.conv2d(y, sobel_y, padding=1, groups=C)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
 
    grad_norm = grad_mag / (grad_mag.flatten(2).quantile(0.95, dim=2)[..., None, None] + 1e-8)
    structure_weight = torch.exp(-5.0 * grad_norm.clamp(0, 1))
 
    weighted_var = F.avg_pool2d(
        high_freq**2 * structure_weight,
        kernel_size=window_size, stride=1, padding=window_size // 2
    )
    weight_sum = F.avg_pool2d(
        structure_weight,
        kernel_size=window_size, stride=1, padding=window_size // 2
    )
    local_variance = weighted_var / (weight_sum + 1e-8)
 
    correction_factor = 164.0
    base_sigma_sq = local_variance / correction_factor
 
    # ── 여기서부터 v2 추가분: intensity-dependent scaling ──
    # luminance 기반 local mean (per-channel보다 안정적)
    if C == 3:
        mu = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    else:
        mu = y.mean(dim=1, keepdim=True)
 
    # smooth luminance (노이즈가 intensity 추정을 흔드는 것 방지)
    mu_smooth = F.avg_pool2d(mu, kernel_size=window_size, stride=1, padding=window_size // 2)
 
    # intensity scaling: Poisson-Gaussian의 signal-dependent 항을 근사
    # (1 + γ·μ) → μ=0(어두움)에서 1.0, μ=1(밝음)에서 1+γ
    # γ=0.5면 밝은 영역이 어두운 영역보다 σ가 ~22% 더 큼 (sqrt(1.5)/sqrt(1.0))
    intensity_scale = (1.0 + intensity_gamma * mu_smooth).expand(-1, C, -1, -1)
 
    sigma_map = torch.sqrt(torch.clamp(base_sigma_sq * intensity_scale, min=1e-8))
 
    return sigma_map, structure_weight