import os
import random

import cv2
from torch.utils.data import Dataset


def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ScanImageDataset(Dataset):
    def __init__(self, noisy_image_dir_path, clean_image_dir_path, patch_size=128, transform=None):
        self.clean_image_file_paths = [os.path.join(noisy_image_dir_path, x) for x in os.listdir(noisy_image_dir_path)]
        self.noisy_image_file_paths = [os.path.join(clean_image_dir_path, x) for x in os.listdir(clean_image_dir_path)]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_file_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_file_paths[index])
        clean_image = load_img(self.clean_image_file_paths[index])

        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :]

        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image


class ScanImageTestDataset(Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])

        # Convert numpy array to PIL image
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path