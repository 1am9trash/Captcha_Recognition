import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchsummary import summary
import os

import utils
from dataset import CAPTCHA_DATASET
from model import CAPTCHA_MODEL

test_path = "../data/test"
batch_size = 1

model = torch.load("result/model_75.pt", map_location="cpu")

test_dataset = CAPTCHA_DATASET(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print("Model Summary")
summary(model, input_size=(1, 80, 26))
print()

print("Model Architecture")
print(model)
print()

filenames = os.listdir(test_path)
matrix = np.zeros((36, 36), dtype=int)

correct_letter = 0
correct_code= 0
for filename in filenames:
    img = Image.open(test_path + "/" + filename)
    img = utils.to_binary(img)
    img = utils.clean_noise(img, 3)
    img = np.array(img)
    x = torch.as_tensor(img).reshape(-1, 1, 80, 26).to(torch.float32)

    predict_letters = ""
    predict_hat = model(x.to("cpu"))
    for i in range(4):
        predict_num = predict_hat[0][i * 36:i * 36 + 36].argmax()
        if predict_num < 10:
            predict_letters += chr(ord("0") + predict_num)
        else:
            predict_letters += chr(ord("A") + predict_num - 10)

        if filename[i] == predict_letters[-1]:
            correct_letter += 1

        if filename[i] >= "0" and filename[i] <= "9":
            label_num = ord(filename[i]) - ord("0")
        else:
            label_num = ord(filename[i]) - ord("A") + 10
        matrix[label_num][predict_num] += 1

    if filename[:4] == predict_letters:
        correct_code += 1

    # print("Label:   ", filename[:4], sep="")
    # print("Predict: ", predict_letters, sep="")

print("Total Letter Acc: {:.6f}".format(float(correct_letter) / len(test_dataset) / 4))
print("Total Code Acc:   {:.6f}".format(float(correct_code) / len(test_dataset)))

for i in range(36):
    if i < 10:
        predict_letter = chr(ord("0") + i)
    else:
        predict_letter = chr(ord("A") + i - 10)
    print(predict_letter)
    print("  Accuracy: {:.4f}".format(matrix[i][i] / matrix[i].sum()))
    if matrix[i][i] == matrix[i].sum():
        continue
    print("  Error on letters:", end="") 
    for j in range(36):
        if i == j or matrix[i][j] == 0:
            continue
        if j < 10:
            error_letter = chr(ord("0") + j)
        else:
            error_letter = chr(ord("A") + j - 10)
        print(" {:s}({:.4f})".format(error_letter, matrix[i][j] / matrix[i].sum()), sep="", end="")
    print()


