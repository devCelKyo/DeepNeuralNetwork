from Networks.DNN import DNN
from tools import readTrainMNIST, show_m, label_array, save, load
import numpy as np
from time import time

X, Y = readTrainMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

tb = 500
epochs = 200

DIMS = [(784, 100, 100, 10), (784, 200, 200, 10), (784, 200, 200, 200, 10),
        (784, 200, 200, 200, 10), (784, 200, 200, 10),
        (784, 300, 300, 10), (784, 700, 700, 10)]

MODELS_P = []
MODELS_NP = []

for dim in DIMS:
    model_name = "DNN-"
    for N in dim:
        model_name += "{}x".format(N)
    model_name = model_name[:-1]
    model_name += "-pretrained"

    model_p = load(model_name)
    model_np = DNN(dim)

    MODELS_P.append(model_p)
    MODELS_NP.append(model_np)
    
model = MODELS_P[0]
