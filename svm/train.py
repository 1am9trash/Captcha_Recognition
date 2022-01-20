import string
import os
from PIL import Image
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import numpy as np

import utils

letters = string.digits + string.ascii_uppercase
train_path = "../data/train_split_letters"
train_data = []
for i, letter in enumerate(letters):
    print("\rData processing: {:6.2f}%".format(i / 35 * 100), sep="", end="")
    for filename in os.listdir(train_path + "/" + letter):
        features = []
        img = Image.open(train_path + "/" + letter + "/" + filename).resize((10, 26))
        train_data.append(np.concatenate((np.array(img).reshape(-1), np.array(i)), axis=None))
print()

train_df = pd.DataFrame(train_data)
train_df = shuffle(train_df)

svc = SVC(kernel="linear")

cv_num = 3
cross = cross_val_score(svc, train_df.drop(260, axis=1), train_df[260], cv=cv_num)
for i in range(cv_num):
    print("Valid: {:.4f}".format(cross[i]))
svc.fit(train_df.drop(260, axis=1), train_df[260])
print("Training Acc: {:.4f}".format(svc.score(train_df.drop(260, axis=1), train_df[260])))


test_data = []
test_path = "../data/test_split_letters"
for filename in os.listdir(test_path):
    img = Image.open(test_path + "/" + filename).resize((10, 26))
    test_data.append(np.array(img).reshape(-1))
test_df = pd.DataFrame(test_data)
predict = svc.predict(test_df)

filenames = os.listdir(test_path)
matrix = np.zeros((36, 36), dtype=int)
correct_letter = 0
correct_code= 0
correct = True
for i, filename in enumerate(filenames):
    if i % 4 == 0:
        correct = True
    matrix[utils.to_number(filename[5])][predict[i]] += 1
    if predict[i] == utils.to_number(filename[5]):
        correct_letter += 1
    else:
        correct = False
    if i % 4 == 3 and correct:
        correct_code += 1

print("Total Letter Acc: {:.6f}".format(correct_letter / len(filenames)))
print("Total Code Acc:   {:.6f}".format(correct_code / (len(filenames) / 4)))

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
