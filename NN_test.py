from Networks.DNN import DNN
from tools import readTrainMNIST, readTestMNIST, show_m, label_array
from tools import save, load, cross_entropie
import numpy as np
from time import time

X, Y = readTrainMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

model_t = load("DNN-784x100x100x10-pretrained-trained-200epochs-trained60k-200epochs")
#model_t.retropropagation(X, Y, 10, 0.1, 5000)

X_test, Y_test = readTestMNIST()
X_test = X_test.reshape((X_test.shape[0], 784))

model_nt = load("DNN-784x100x100x10-pretrained")

def accuracy(model, X, Y):
    Y_p = model.entree_sortie(X)[-1]
    n = X.shape[0]
    Y_pred = np.zeros(n)
    for i in range(n):
        Y_pred[i] = np.argmax(Y_p[i])

    return np.sum((Y_pred == Y))/n

print(accuracy(model_t, X_test, Y_test))
print(accuracy(model_nt, X_test, Y_test))
