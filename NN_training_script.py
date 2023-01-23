from Networks.DNN import DNN
from tools import readTrainMNIST, show_m, label_array, save, load, hms
import numpy as np
from time import time

X, Y = readTrainMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

tb = 5000
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

for i in range(len(MODELS_P)):
    model_p = MODELS_P[i]
    model_np = MODELS_NP[i]

    debut = time()
    print("Modèle préentrainé : {}".format(model_p.dims))
    model_p.retropropagation(X, Y, epochs, 0.1, tb, True)
    fin = time()
    print("tps d'exec : {}h {}m {}s".format(hms(fin - debut)))

    save(model_p, note="trained60k-200epochs")
