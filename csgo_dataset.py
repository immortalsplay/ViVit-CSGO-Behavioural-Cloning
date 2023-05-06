import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


class CSGODataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['data'])

        self.new_size = (224, 224)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            data = f['data'][idx]
            label = f['label'][idx]

        data = data.reshape(1100, 125, 200, 3)  # 修改数据形状
        data = np.array([cv2.resize(frame, self.new_size) for frame in data])
        data = np.transpose(data, (0, 3, 1, 2))  # 调整通道维度的位置

        return torch.from_numpy(data).float(), torch.tensor(label).long()


class SmallCSGODataset(Dataset):
    def __init__(self, h5_file, num_samples, frames_per_sample):
        self.h5_file = h5_file
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample

        with h5py.File(self.h5_file, 'r') as f:
            self.data = f['data'][0:self.num_samples]
            self.label = f['label'][0:self.num_samples]

        self.new_size = (224, 224)

    def __len__(self):
        return self.num_samples * (1100 // self.frames_per_sample)

    def __getitem__(self, idx):
        data_idx = idx // (1100 // self.frames_per_sample)
        frame_idx = (idx % (1100 // self.frames_per_sample)) * self.frames_per_sample

        data = self.data[data_idx][frame_idx:frame_idx + self.frames_per_sample]
        label = self.label[data_idx][frame_idx:frame_idx + self.frames_per_sample]

        data = data.reshape(-1, 125, 200, 3)  # 修改数据形状
        data = np.array([cv2.resize(frame, self.new_size) for frame in data])
        data = np.transpose(data, (0, 3, 1, 2))  # 调整通道维度的位置
        return torch.from_numpy(data).float(), torch.tensor(label).float()


