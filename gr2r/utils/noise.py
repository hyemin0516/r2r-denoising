from __future__ import annotations
import torch
from torch.nn import functional as F


# -------------------------
# Noise: add_noise (x -> y) and R2R corruptor (y -> y1)
# (matches the author's pytorch_demo.py as closely as possible)
# -------------------------
def add_gaussian_noise(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_level
    return x + noise

def add_poisson_noise(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    # torch.poisson expects rate; demo uses: poisson(x / noise_level) * noise_level
    return torch.poisson(x / noise_level) * noise_level

def add_gamma_noise(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    # demo: Gamma(noise_level, 1) sample * (x / noise_level)
    gamma_dist = torch.distributions.Gamma(noise_level, 1.0)
    sample = gamma_dist.sample(x.shape).to(x.device)
    return sample * (x / noise_level)

def add_correlated_poisson_noise(x: torch.Tensor, noise_level: float, kernel_size: int = 3) -> torch.Tensor:
    """
    공간적 상관관계가 있는 푸아송 노이즈 생성
    (3차원 입력[C,H,W]과 4차원 입력[B,C,H,W] 모두 지원하도록 수정)
    """
    # 1. 입력 차원 확인 및 배치 차원 추가
    is_batch = x.ndim == 4
    if not is_batch:
        x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    b, c, h, w = x.shape  # 이제 4개로 언패킹 가능
    
    # 2. 독립적인(Independent) 푸아송 노이즈 생성
    y_indep = torch.poisson(x / noise_level) * noise_level
    
    # 3. 노이즈 성분만 추출 (Residual)
    noise_residual = y_indep - x
    
    # 4. 노이즈에 블러(Blur)를 적용하여 픽셀 간 상관관계 형성
    kernel = torch.ones((c, 1, kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
    
    # Padding을 주어 크기 유지
    noise_correlated = F.conv2d(
        noise_residual, kernel, padding=kernel_size//2, groups=c
    )
    
    # 5. 블러링으로 줄어든 노이즈의 강도(분산)를 다시 맞춰줌
    scale_factor = noise_residual.std() / (noise_correlated.std() + 1e-8)
    noise_correlated = noise_correlated * scale_factor
    
    # 6. 원본에 더함
    out = x + noise_correlated
    out = out.clamp(min=0.0)
    
    # 7. 원래 차원으로 복구
    if not is_batch:
        out = out.squeeze(0)
        
    return out

def add_correlated_gaussian_noise(x: torch.Tensor, noise_level: float, kernel_size: int = 3) -> torch.Tensor:
    """
    공간적 상관관계가 있는 가우시안 노이즈 생성
    """
    # 1. 입력 차원 확인 및 배치 차원 추가
    is_batch = x.ndim == 4
    if not is_batch:
        x = x.unsqueeze(0)
    
    b, c, h, w = x.shape
    
    # 2. 독립적인(Independent) 가우시안 노이즈 생성 (순수 노이즈)
    # [수정됨] 기존 코드의 y_indep - x 과정은 불필요합니다.
    noise_indep = torch.randn_like(x) * noise_level
    
    # 3. 노이즈에 블러(Blur)를 적용하여 픽셀 간 상관관계 형성
    # (커널 생성 부분은 동일)
    kernel = torch.ones((c, 1, kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
    
    # Padding을 주어 크기 유지
    noise_correlated = F.conv2d(
        noise_indep, kernel, padding=kernel_size//2, groups=c
    )
    
    # 4. 블러링으로 줄어든 노이즈의 강도(Std)를 다시 맞춰줌
    # 블러링을 하면 노이즈가 뭉개지면서 분산(Variance)이 줄어들기 때문에 보정해야 합니다.
    scale_factor = noise_indep.std() / (noise_correlated.std() + 1e-8)
    noise_correlated = noise_correlated * scale_factor
    
    # 5. 원본에 더함
    out = x + noise_correlated
    
    # [답변] 가우시안 노이즈는 음수 값이 나오므로 clamp가 필수입니다.
    # 일반적으로 이미지 범위인 0~1 사이로 맞춥니다.
    out = out.clamp(0.0, 1.0)
    
    # 6. 원래 차원으로 복구
    if not is_batch:
        out = out.squeeze(0)
        
    return out

def gaussian_corruptor(y: torch.Tensor, alpha: float, noise_level: float) -> torch.Tensor:
    noise = torch.randn_like(y) * noise_level
    return y + alpha * noise

def poisson_corruptor(y: torch.Tensor, alpha: float, noise_level: float) -> torch.Tensor:
    z = y / noise_level
    z_int = torch.round(z)
    binom = torch.distributions.Binomial(total_count=z_int, probs=alpha)
    w = binom.sample().to(y.dtype)
    return noise_level * (z - w) / (1 - alpha)

def gamma_corruptor(y: torch.Tensor, alpha: float, noise_level: float) -> torch.Tensor:
    concentration1 = noise_level * alpha
    concentration0 = noise_level * (1 - alpha)
    beta_dist = torch.distributions.Beta(concentration1, concentration0)
    w = beta_dist.sample(y.shape).to(y.device)
    return y * (1 - w) / (1 - alpha)


def get_noise_fns(distribution: str):
    distribution = distribution.lower()
    if distribution == "gaussian":
        return add_gaussian_noise, gaussian_corruptor
    if distribution == "poisson":
        return add_poisson_noise, poisson_corruptor
    if distribution == "gamma":
        return add_gamma_noise, gamma_corruptor
    if distribution == "correlated_gaussian":
        return add_gaussian_noise, gaussian_corruptor
    if distribution == "correlated_poisson":
        # 노이즈 생성은 '상관관계 푸아송'
        # Corruptor(분할)는 '일반 푸아송' 사용 (모델을 속이기 위해)
        return add_correlated_gaussian_noise, gaussian_corruptor
    raise ValueError(f"Unknown distribution: {distribution}")


