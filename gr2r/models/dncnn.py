import math
import torch
import torch.nn as nn


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


class DnCNN(nn.Module):
    """
    Bias-Free DnCNN (Adapted for Patch-Craft Correlated Noise Removal)
    forward(x, sigma=None) kept to be compatible with deepinv-style signatures,
    but sigma is unused.
    """
    def __init__(self, in_channels=3, out_channels=3, depth=20, nf=64):
        super().__init__()
        self.depth = int(depth)

        # 1. 편향(Bias) 완전 제거: 모든 Conv에서 bias=False 강제
        self.in_conv = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=False)
        
        self.conv_list = nn.ModuleList([])
        self.running_sd = nn.ParameterList([])
        self.gammas = nn.ParameterList([])
        
        # 2. 중간 레이어 및 BF_BatchNorm용 파라미터 초기화
        for _ in range(self.depth - 2):
            self.conv_list.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
            # 추론 시 사용할 이동 평균 표준편차 (학습되지 않음)
            self.running_sd.append(nn.Parameter(torch.ones(1, nf, 1, 1), requires_grad=False))
            # 스케일 파라미터 감마 (학습됨, 극소값으로 초기화)
            g = (torch.randn((1, nf, 1, 1)) * (2. / 9. / 64.)).clamp_(-0.025, 0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True))
            
        self.out_conv = nn.Conv2d(nf, out_channels, 3, 1, 1, bias=False)
        
        # inplace=True로 메모리 효율성 확보
        self.nl_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(self.depth - 1)])

        # 기존처럼 Kaiming 초기화 유지 (외부 함수라 가정)
        self.apply(weights_init_kaiming)

    def forward(self, x, sigma=None):
        # 3. 노이즈 잔차 학습을 위해 원본 이미지 보존
        x_in = x.clone()
        
        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)
        
        for i in range(self.depth - 2):
            x1 = self.conv_list[i](x1)
            
            # 4. BF_BatchNorm 적용 (평균 차감 없음)
            sd_x = torch.sqrt(x1.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-05)
            
            if self.training:
                # 학습 시: 현재 배치의 표준편차로 나누고 EMA 업데이트
                x1 = x1 / sd_x.expand_as(x1)
                self.running_sd[i].data = (1 - 0.1) * self.running_sd[i].data + 0.1 * sd_x
                x1 = x1 * self.gammas[i].expand_as(x1)
            else:
                # 추론(Eval) 시: 누적된 이동 평균 표준편차 사용
                x1 = x1 / self.running_sd[i].expand_as(x1)
                x1 = x1 * self.gammas[i].expand_as(x1)
                
            x1 = self.nl_list[i + 1](x1)
            
        out = self.out_conv(x1)
        
        # 5. 순수 노이즈 예측 모드: 원본에서 예측된 노이즈를 뺌
        return out


# class DnCNN(nn.Module):
#     def __init__(self, channels=3, num_of_layers=17):
#         super(DnCNN, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         layers = []
#         layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(num_of_layers-2):
#             layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
#             layers.append(nn.BatchNorm2d(features))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
#         self.dncnn = nn.Sequential(*layers)
#     def forward(self, x):
#         out = self.dncnn(x)
#         return x-out