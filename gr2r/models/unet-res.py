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

def weights_init_drunet(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)

class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    ICLR 2020 논문 기반 Bias-Free BatchNorm.
    노이즈의 평균(Mean Shift)을 외우는 꼼수를 원천 차단하여 일반화 성능을 극대화합니다.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, use_bias=False, affine=True):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.use_bias = use_bias
        self.affine = affine

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** 0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** 0.5 + self.eps)
        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)
        
class UNet(nn.Module):
    r"""
    ResUNet architecture (DRUNet without Noise Map input).
    It takes only the image as input.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        bias=False
    ):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 재사용 가능한 블록 생성 함수들 (내부에 Bias-free BN 포함)
        def conv_block(ch_in, ch_out):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                BFBatchNorm2d(ch_out, use_bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                BFBatchNorm2d(ch_out, use_bias=bias),
                nn.ReLU(inplace=True),
            )

        def up_conv(ch_in, ch_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                BFBatchNorm2d(ch_out, use_bias=bias),
                nn.ReLU(inplace=True),
            )

        # Encoding Path
        self.Conv1 = conv_block(in_channels, 64)
        self.Conv2 = conv_block(64, 128)
        self.Conv3 = conv_block(128, 256)
        self.Conv4 = conv_block(256, 512)

        # Decoding Path
        self.Up4 = up_conv(512, 256)
        self.Up_conv4 = conv_block(512, 256) # 256+256이 Cat되므로 in_channels는 512

        self.Up3 = up_conv(256, 128)
        self.Up_conv3 = conv_block(256, 128)

        self.Up2 = up_conv(128, 64)
        self.Up_conv2 = conv_block(128, 64)

        # Final 1x1 Conv
        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

        # 가중치 초기화
        self.apply(weights_init_drunet)
        nn.init.zeros_(self.Conv_1x1.weight)
        

    def forward(self, x):
        identity = x
        h, w = x.size()[-2:]
        
        # 16의 배수(2^4)로 패딩 (박사님의 안전장치 코드 이식!)
        paddingBottom = (16 - h % 16) % 16
        paddingRight = (16 - w % 16) % 16
        if paddingBottom > 0 or paddingRight > 0:
            x = F.pad(x, (0, paddingRight, 0, paddingBottom), mode='reflect')

        # [1. Encoding]
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))

        # [2. Decoding + Concatenation (핵심!)]
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1) # 채널(dim=1) 방향으로 이어붙이기
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # [3. Final Projection]
        out = self.Conv_1x1(d2)

        # 패딩 원상복구
        if paddingBottom > 0 or paddingRight > 0:
            out = out[..., :h, :w]

        # [4. Global Residual Learning (핵심!)]
        # 노이즈(out)만 예측한 뒤 원본(identity)에 더해 정제된 이미지를 반환
        return identity + out

