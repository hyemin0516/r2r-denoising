import os
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class SIDDTrainDataset(Dataset):
    def __init__(self, root_dir, crop_size=240):
        """
        Args:
            root_dir (str): 'dataset/prep/SIDD_s512_o128' 경로
            crop_size (int): 박사님의 황금 사이즈 240
        """
        self.crop_size = crop_size
        
        # 1. Clean(CL)과 Noisy(RN) 경로 읽어오기
        # glob으로 읽은 뒤 sort()를 해야 순서가 1:1로 매칭됩니다.
        self.clean_paths = sorted(glob(os.path.join(root_dir, 'CL', '*.png')))
        self.noisy_paths = sorted(glob(os.path.join(root_dir, 'RN', '*.png')))
        
        # 데이터 개수 확인 (안전장치)
        assert len(self.clean_paths) == len(self.noisy_paths), \
            f"개수 불일치! Clean: {len(self.clean_paths)}, Noisy: {len(self.noisy_paths)}"
        
        print(f"[SIDD Train] Loaded {len(self.noisy_paths)} pairs. Crop Size: {crop_size}")

    def __len__(self):
        return len(self.noisy_paths)

    def _sync_transform(self, clean, noisy):
        """
        [중요] Clean과 Noisy에 '동일한' 랜덤 변환을 적용하는 함수
        """
        # 1. Random Crop (위치 동기화)
        # 512 이미지에서 240 크기를 뜯어낼 좌표(i, j, h, w)를 구함
        i, j, h, w = T.RandomCrop.get_params(
            clean, output_size=(self.crop_size, self.crop_size)
        )
        clean = TF.crop(clean, i, j, h, w)
        noisy = TF.crop(noisy, i, j, h, w)

        # 2. Random Horizontal Flip (확률 50%)
        if random.random() > 0.5:
            clean = TF.hflip(clean)
            noisy = TF.hflip(noisy)

        # 3. Random Vertical Flip (확률 50%)
        if random.random() > 0.5:
            clean = TF.vflip(clean)
            noisy = TF.vflip(noisy)

        # 4. ToTensor (0~1 범위로 변환)
        clean = TF.to_tensor(clean)
        noisy = TF.to_tensor(noisy)
        
        return clean, noisy

    def __getitem__(self, idx):
        # 1. 이미지 로드 (PIL)
        # self.paths[idx] 대신 Clean/Noisy 각각 로드
        clean_img = Image.open(self.clean_paths[idx]).convert("RGB")
        noisy_img = Image.open(self.noisy_paths[idx]).convert("RGB")

        # 2. 동기화된 Transform 적용
        # add_noise 대신, 이미 있는 Real Noisy를 함께 변환
        x, y = self._sync_transform(clean_img, noisy_img)

        # 3. 박사님 코드의 batch["x"], batch["y"] 키값과 일치시킴
        # x: Clean (Ground Truth) -> Loss 계산이나 PSNR 측정용
        # y: Noisy (Input) -> 모델에 들어갈 입력
        return {"x": x, "y": y}

class SIDDValidationDataset(Dataset):
    def __init__(self, noisy_file_path, gt_file_path):
        """
        Args:
            noisy_file_path: 'ValidationNoisyBlocksSrgb.mat' 경로
            gt_file_path: 'ValidationGtBlocksSrgb.mat' 경로
        """
        super().__init__()
        
        # 1. .mat 파일 로드
        # 데이터 형태: (40, 32, 256, 256, 3) -> (Scene, Block, H, W, C)
        self.noisy_data = scipy.io.loadmat(noisy_file_path)['ValidationNoisyBlocksSrgb']
        self.gt_data = scipy.io.loadmat(gt_file_path)['ValidationGtBlocksSrgb']
        
        # 2. 평탄화 (Flatten)
        # (1280, 256, 256, 3) 형태로 쭉 폅니다.
        n_scenes, n_blocks, h, w, c = self.noisy_data.shape
        self.n_samples = n_scenes * n_blocks
        
        self.noisy_data = self.noisy_data.reshape(self.n_samples, h, w, c)
        self.gt_data = self.gt_data.reshape(self.n_samples, h, w, c)
        
        self.to_tensor = ToTensor()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 3. 인덱스에 맞는 이미지 가져오기 (0~255 uint8 상태)
        noisy_img = self.noisy_data[idx]
        gt_img = self.gt_data[idx]
        
        # 4. 텐서 변환 및 정규화 (0~1 float)
        # [중요] 여기서 노이즈를 더하면 안 됩니다! (이미 노이즈가 있음)
        noisy_tensor = self.to_tensor(noisy_img)
        gt_tensor = self.to_tensor(gt_img)
        
        # 딕셔너리 형태로 반환 (박사님 코드 스타일에 맞춤)
        return {"x": gt_tensor, "y": noisy_tensor}