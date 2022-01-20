import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchsummary import summary

import utils
from model import CAPTCHA_MODEL
from dataset import CAPTCHA_DATASET

train_path = "../data/train"
test_path = "../data/test"
n_epochs = 40
batch_size = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
print()

# model = CAPTCHA_MODEL().to(device)
# model.device = device
model = torch.load("result/model.pt", map_location=device)

train_dataset = CAPTCHA_DATASET(train_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataset = CAPTCHA_DATASET(test_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.45)
# optimizer_state = optimizer.state_dict()
# scheduler_state = lr_scheduler.state_dict()
# optimizer.load_state_dict(optimizer_state)
# lr_scheduler.load_state_dict(scheduler_state)

print("Model Summary")
summary(model, input_size=(1, 80, 26))
print()

print("Model Architecture")
print(model)
print()

best = 0.0
for epoch in range(n_epochs):
    model.eval()
    test_acc_letter = 0
    test_acc_code = 0
    for imgs, labels in test_dataloader:
        imgs = imgs.reshape(-1, 1, 80, 26).to(torch.float32)
        with torch.no_grad():
            labels_hat = model(imgs.to(device))
            for i in range(len(imgs)):
                code_correct = True
                for j in range(4):
                    correct = (labels[i][j * 36:j * 36 + 36].to(device).argmax() == labels_hat[i][j * 36:j * 36 + 36].argmax())
                    test_acc_letter += correct
                    if not correct:
                        code_correct = False
                test_acc_code += code_correct
        torch.cuda.empty_cache()
    print("Valid ", epoch, sep="")
    print("  Letter acc: {:.6f}".format(float(test_acc_letter) / len(test_dataset) / 4), sep="")
    print("    Code acc: {:.6f}".format(float(test_acc_code) / len(test_dataset)), sep="")
    if test_acc_code / len(test_dataset) > best:
        print("Save best model")
        best = test_acc_code / len(test_dataset)
        torch.save(model, "result/model.pt")
        
    model.train()
    train_acc_letter = 0
    train_acc_code = 0
    for imgs, labels in train_dataloader:
        optimizer.zero_grad()
        imgs = imgs.reshape(-1, 1, 80, 26).to(torch.float32)
        labels_hat = model(imgs.to(device))
        train_loss = criterion(labels_hat, labels.to(device))
        train_loss.backward()
        optimizer.step()
        for i in range(len(imgs)):
            code_correct = True
            for j in range(4):
                correct = (labels[i][j * 36:j * 36 + 36].to(device).argmax() == labels_hat[i][j * 36:j * 36 + 36].argmax())
                train_acc_letter += correct
                if not correct:
                    code_correct = False
            train_acc_code += code_correct
    # lr_scheduler.step()
    print("Train: ", epoch, sep="")
    print("  Letter acc: {:.6f}".format(float(train_acc_letter) / len(train_dataset) / 4), sep="")
    print("    Code acc: {:.6f}".format(float(train_acc_code) / len(train_dataset)), sep="")
    print()
