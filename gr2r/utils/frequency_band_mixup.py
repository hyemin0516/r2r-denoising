import torch 
import torch.fft

def apply_residual_frequency_band_mixup(
    residual,
    low_range=(0.2, 0.8),
    mid_range=(0.7, 1.3),
    high_range=(0.8, 1.5),
    low_cut=0.2,
    high_cut=0.6,
    renorm=True,
):
    B, C, H, W = residual.shape
    device = residual.device
    dtype = residual.dtype

    freq = torch.fft.fftshift(
        torch.fft.fft2(residual, norm="ortho"),
        dim=(-2, -1)
    )

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = torch.sqrt(torch.tensor(cy ** 2 + cx ** 2, device=device, dtype=dtype))
    r = radius / (max_radius + 1e-8)

    mask_low = (r < low_cut).to(dtype).view(1, 1, H, W)
    mask_mid = ((r >= low_cut) & (r < high_cut)).to(dtype).view(1, 1, H, W)
    mask_high = (r >= high_cut).to(dtype).view(1, 1, H, W)

    def rand_range(lo, hi):
        return lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=device, dtype=dtype)

    w_low = rand_range(*low_range)
    w_mid = rand_range(*mid_range)
    w_high = rand_range(*high_range)

    band_weight = mask_low * w_low + mask_mid * w_mid + mask_high * w_high

    freq_aug = freq * band_weight

    residual_aug = torch.fft.ifft2(
        torch.fft.ifftshift(freq_aug, dim=(-2, -1)),
        norm="ortho"
    ).real

    if renorm:
        orig_std = residual.std(dim=(2, 3), keepdim=True) + 1e-8
        new_std = residual_aug.std(dim=(2, 3), keepdim=True) + 1e-8
        residual_aug = residual_aug / new_std * orig_std

    return residual_aug