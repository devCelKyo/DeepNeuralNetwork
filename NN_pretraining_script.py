from Networks.DNN import DNN
from tools import readTrainMNIST, show_m, label_array, save
import numpy as np
from time import time

X, Y = readTrainMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

DIMS = [(784, 200, 200, 10), (784, 200, 200, 200, 10), (784, 200, 200, 200, 200, 200, 10),
        (784, 100, 100, 10), (784, 300, 300, 10), (784, 700, 700, 10)]
tb = 100
epochs = 100

debut = time()
for dim in DIMS:
    print(dim)
    check = debut
    model = DNN(dim)
    model.pretrain(epochs, 0.1, tb, X)
    save(model)
    print("Tps d'exec pour ce modele : {}s".format(time() - check))
    
duree = time() - debut
print("Tps d'exec total : {}s".format(duree))
