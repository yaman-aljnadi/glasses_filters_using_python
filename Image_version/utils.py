import os 
import cv2
import numpy as np 
import matplotlib.pyplot as pyplot

from keras.models import load_model
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load_data(test=False):
    file_train = "data/training.csv"
    file_test = "data/test.csv"

    file_name = file_test if test else file_train

    df = read_csv(os.path.expanduser(file_name))

    df["Image"] = df["Image"].apply(lambda im: np.fromstring(im, sep=" "))

    df = df.dropna()

    X = np.vstack(df["Image"].values) / 255 
    X = X.astype(np.float32)
    X = X.reshape(-1,96,96, 1)

    if not test:
        Y = df[df.columns[:-1]].values
        Y = (Y - 48) / 48

        X, Y = shuffle(X, Y, random_state=42)
        Y = Y.astype(np.float32)
    else:
        Y = None

    return X, Y 