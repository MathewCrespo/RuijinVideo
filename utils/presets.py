import torch
from torchvision.transforms import transforms
import torch.nn as nn
from PIL import Image
import numpy as np


class VideoClassificationPresetTrain:
    def __init__(
        self,
        resize_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        trans = [
            # transforms.ConvertImageDtype(torch.float32),
            # transforms.ToTensor(),
            ConvertBHWCtoCBHW(),
            transforms.Resize(resize_size),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([transforms.Normalize(mean=mean, std=std)])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        # x = Image.fromarray(np.transpose(x,(0,3,1,2)))
        # print(x.type)
        x=torch.Tensor(x)
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self, resize_size, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose(
            [
                # transforms.ToTensor(),
                ConvertBHWCtoCBHW(),
                # transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(resize_size),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x):
        x=torch.Tensor(x)
        # x = Image.fromarray(np.transpose(x,(0,3,1,2)))
        return self.transforms(x)


class ConvertBHWCtoCBHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(3,0,1,2)