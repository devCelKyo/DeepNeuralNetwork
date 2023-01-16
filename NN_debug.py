from Networks.DNN import DNN
from tools import readTrainMNIST, show_m
import numpy as np

X, Y = readTrainMNIST()
img = X[0].flatten()

model = DNN((img.shape[0], 200, 200, 10))
sortie = model.entree_sortie(img)

Y = sortie[-1]
