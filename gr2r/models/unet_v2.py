import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
class UNet_v2(nn.Module):
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
        super(UNet_v2, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(3,32,5,padding = 2, bias = bias)
        self.conv2 = nn.Conv2d(32,32,3,padding = 1, bias = bias)
        self.conv3 = nn.Conv2d(32,64,3,stride=2, padding = 1, bias = bias)
        self.conv4 = nn.Conv2d(64,64,3,padding = 1, bias=bias)
        self.conv5 = nn.Conv2d(64,64,3,dilation=2, padding = 2, bias = bias)
        self.conv6 = nn.Conv2d(64,64,3,dilation = 4,padding = 4, bias = bias)
        self.conv7 = nn.ConvTranspose2d(64,64, 4,stride = 2, padding = 1, bias = bias)
        self.conv8 = nn.Conv2d(96,32,3,padding=1, bias = bias)
        self.conv9 = nn.Conv2d(32,3,5,padding = 2, bias = False)

        # 가중치 초기화
        self.apply(weights_init_drunet)

        nn.init.zeros_(self.conv9.weight)
        if bias:
            nn.init.zeros_(self.conv9.bias)
        

    def forward(self, x, return_features=None):
        h, w = x.size()[-2:]
        
        # 16의 배수(2^4)로 패딩 (박사님의 안전장치 코드 이식!)
        paddingBottom = x.shape[-2]%2
        paddingRight = x.shape[-1]%2
        
        x = F.pad(x, (0, paddingRight, 0, paddingBottom), mode='reflect')
        
        out = F.relu(self.conv1(x))
        
        out_saved = F.relu(self.conv2(out))
        
        out = F.relu(self.conv3(out_saved))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        
        out = torch.cat([out,out_saved],dim = 1)
        
        out = F.relu(self.conv8(out))
        out = self.conv9(out)
        
        # 패딩 원상복구
        if paddingBottom > 0 or paddingRight > 0:
            out = out[..., :h, :w]
        
        # [4. Global Residual Learning (핵심!)]
        return out

