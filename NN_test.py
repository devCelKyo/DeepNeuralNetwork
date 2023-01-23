from Networks.DNN import DNN
from tools import readTrainMNIST, readTestMNIST, show_m, label_array
from tools import save, load, cross_entropie
import numpy as np
from time import time

X_test, Y_test = readTestMNIST()
X_test, Y_test = X_test.reshape((X_test.shape[0], 784)), label_array(Y_test)

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

model_p = MODELS_P[0]
sorties_p = model_p.entree_sortie(X_test)[-1]

model_np = MODELS_NP[0]
sorties_np = model_np.entree_sortie(X_test)[-1]
