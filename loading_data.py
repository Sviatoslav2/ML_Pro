import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook

img_size_ori = 101
img_size_target = 128


train_df = pd.read_csv("../Data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../Data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]


train_df["images"] = [np.array(load_img("../Data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["masks"] = [np.array(load_img("../Data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

