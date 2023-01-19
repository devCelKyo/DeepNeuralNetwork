from Networks.DNN import DNN
from tools import readTrainMNIST, show_m, label_array, save, load, hms
import numpy as np
from time import time

X, Y = readTrainMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

DIMS = [(784, 100, 100, 10), (784, 200, 200, 10), (784, 200, 200, 200, 10),
        (784, 200, 200, 200, 200, 200, 10),
        (784, 300, 300, 10), (784, 700, 700, 10)]

tb = 500
epochs = 100

debut = time()
for dim in DIMS:
    print(dim)
    check = time()
    model = DNN(dim)
    model.pretrain(epochs, 0.1, tb, X)
    save(model)
    duree = time() - check
    h, m, s = hms(duree)
    print("Tps d'execution : {}h {}m {}s".format(h, m, s))
fin = time()

h, m, s = hms(fin - debut)
print("Tps d'execution total pretraining : {}h {}m {}s".format(h, m, s))
