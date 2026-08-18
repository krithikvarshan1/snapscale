import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconductorDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, filenames, augment=False):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.filenames = filenames
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        noisy = np.load(os.path.join(self.noisy_dir, filename)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, filename)).astype(np.float32)

        # Spatial data augmentations during training
        if self.augment:
            if random.random() < 0.5:
                noisy = np.fliplr(noisy).copy()
                gt = np.fliplr(gt).copy()

            if random.random() < 0.5:
                noisy = np.flipud(noisy).copy()
                gt = np.flipud(gt).copy()

            k = random.randint(0, 3)
            if k > 0:
                noisy = np.rot90(noisy, k).copy()
                gt = np.rot90(gt, k).copy()

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt
