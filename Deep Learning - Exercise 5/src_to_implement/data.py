from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor, Normalize, ColorJitter, RandomAffine

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self,data,mode: str):
        super().__init__()
        self.data = data
        self.mode = mode
        if self.mode == "train":
            self.transform = transforms.Compose([
                ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                ToTensor(),
                Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self.transform = transforms.Compose([
                ToPILImage(),
                ToTensor(),
                Normalize(mean=train_mean, std=train_std)
            ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        filename, crack, inactive = self.data.iloc[index]
        img = imread(Path(filename), as_gray=True)
        img = gray2rgb(img)
        img = self.transform(img)
        return img, torch.tensor([crack,inactive], dtype=torch.float32)

