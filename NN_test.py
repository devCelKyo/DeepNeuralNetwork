from tools import readTrainMNIST, readTestMNIST, show, label_array
from tools import lire_alpha_digit
from DNN import DNN
import numpy as np

X, Y = readTrainMNIST()
X_flat, Y_arr = np.reshape(X, (X.shape[0], -1)), label_array(Y)

N_dims = (X_flat.shape[1], 100, 100, 10)
n_iter_rbm = 100
n_iter_retro = 200
eps = 0.1
tb = 10

NN = DNN(N_dims)
#NN.pretrain(n_iter_rbm, eps, tb, X_flat)
#NN.retropropagation(X_flat, Y_arr, eps, tb)
