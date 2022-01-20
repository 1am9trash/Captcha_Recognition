import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import utils

class CAPTCHA_DATASET(Dataset):
    def __init__(self, path):
        self.path = path
        self.filename_list = os.listdir(self.path)

    def __getitem__(self, index):
        filename = self.filename_list[index]
        img = Image.open(self.path + "/" + filename)
        img = utils.to_binary(img)
        img = utils.clean_noise(img, 3)
        img = np.array(img)
        
        label = np.zeros((144), dtype=np.float32)
        for i, letter in enumerate(filename[:4]):
            if letter >= "A" and letter <= "Z":
                label[i * 36 + 10 + ord(letter) - ord("A")] = 1.0
            else:
                label[i * 36 + ord(letter) - ord("0")] = 1.0
        return  torch.as_tensor(img),  torch.as_tensor(label)

    def __len__(self):
        return len(self.filename_list)
