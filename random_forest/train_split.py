import numpy as np
import os
from PIL import Image

import utils

cnt = np.zeros((36), dtype=int)
train_path = "../data/train"
filenames = os.listdir(train_path)
for filename in filenames:
    img = Image.open(train_path + "/" + filename)
    img = utils.to_binary(img)
    img = utils.clean_noise(img, 3)

    splits = utils.letter_split(img)
    for i, split in enumerate(splits):
         n = utils.to_number(filename[i])
         path = "../data/train_split_letters/" + filename[i] + "/" + str(cnt[n]) + ".jpg"
         sub_img = img.crop((split[0], 0, split[1] + 1, 26))
         sub_img.save(path)
         cnt[n] += 1
