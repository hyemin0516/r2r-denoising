from __future__ import annotations
import torchvision.transforms as T


def build_transforms(img_size: int):
    to_tensor = T.ToTensor()
    train_tf = T.Compose([
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        to_tensor,
    ])
    val_tf = T.Compose([
        T.CenterCrop((256, 256)),
        to_tensor,
    ])
    return train_tf, val_tf