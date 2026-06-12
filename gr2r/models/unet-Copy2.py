import torch
import torch.nn as nn
from collections import OrderedDict

# -------------------------------------------------------------------------
# [수정된 부분] Standard ResUNet (Noise Map 입력 없음)
# -------------------------------------------------------------------------
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)
        
class UNet(nn.Module):
    r"""
    ResUNet architecture (DRUNet without Noise Map input).
    It takes only the image as input.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        device=None,
    ):
        super(UNet, self).__init__()
        
        # [변경 1] Noise Channel(+1) 제거
        self.m_head = conv(in_channels, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(f"downsample mode [{downsample_mode}] is not found")

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode="2"),
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode="2"),
        )
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode="2"),
        )

        self.m_body = sequential(
            *[ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode}] is not found")

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
        )

        self.m_tail = conv(nc[0], out_channels, bias=False, mode="C")

        # 가중치 초기화
        self.apply(weights_init_drunet)
        nn.init.zeros_(self.m_tail.weight)
        
        if device is not None:
            self.to(device)

    def forward(self, x):
        # [수정] Sigma Map 처리 로직 제거, 순수 U-Net Forward
        
        h, w = x.size()[-2:]
        identity = x
        # ------------------------------------------------------------------
        # [Fix] torch.ceil 에러 해결: 나머지 연산(%)으로 패딩 계산
        # h가 8의 배수면 0, 아니면 부족한 만큼(8 - 나머지) 계산
        # ------------------------------------------------------------------
        paddingBottom = (16 - h % 16) % 16
        paddingRight = (16 - w % 16) % 16

        # 입력 크기가 8의 배수가 아닐 경우 안전장치 패딩 (Reflect Pad)
        if paddingBottom > 0 or paddingRight > 0:
            x = nn.functional.pad(x, (0, paddingRight, 0, paddingBottom), mode='reflect')

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        
        x = self.m_body(x4)
        
        # Additive Skip Connection (ResUNet Style)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        
        x = self.m_tail(x + x1)
        
        # 패딩이 들어갔었다면 원상복구 (잘라내기)
        if paddingBottom > 0 or paddingRight > 0:
            x = x[..., :h, :w]
            
        return identity + x

# -------------------------------------------------------------------------
# 아래는 기존 DRUNet의 Utils (Conv, ResBlock 등) - 그대로 유지
# -------------------------------------------------------------------------

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CBR", negative_slope=0.2):
    L = []
    for t in mode:
        if t == "C":
            L.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "T":
            L.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "2":
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        # ... (필요 시 다른 모드 추가 가능)
    return sequential(*L)

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="CRC", negative_slope=0.2):
        super(ResBlock, self).__init__()
        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        return x + self.res(x)

def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode="2R", negative_slope=0.2):
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode="2R", negative_slope=0.2):
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)

def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode="2R", negative_slope=0.2):
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)

def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode="2R", negative_slope=0.2):
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode="C" + mode, negative_slope=negative_slope)
    return up1

def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode="2R", negative_slope=0.2):
    if mode[0] == "2": uc = "UC"
    elif mode[0] == "3": uc = "uC"
    elif mode[0] == "4": uc = "vC"
    mode = mode.replace(mode[0], uc)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)

def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode="2R", negative_slope=0.2):
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

def weights_init_drunet(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)