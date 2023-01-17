from Networks.DNN import DNN
from tools import readTrainMNIST, show_m, label_array
import numpy as np

N = 200
X, Y = readTrainMNIST()
X_train, Y_train = X[0:N].reshape((N, 784)), label_array(Y[0:N])

model = DNN((784, 200, 200, 10))
model.retropropagation(X_train, Y_train, 200, 0.1, 1)

img = X[0:N].reshape((N, 784))
sortie = model.entree_sortie(img)

Y_pred = np.array(sortie[-1])
Y_true = label_array(Y[0:N])
print(Y_pred)
print(Y_true)
