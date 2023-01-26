from Networks.DNN import DNN
from tools import readTrainMNIST, label_array, save, hms
from time import time
import numpy as np

X, Y = readTrainMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

epochs, eps, tb = 100, 0.1, 5000

model = DNN((784, 700, 700, 700, 10))

start_pretrain = time()
model.pretrain(epochs, eps, tb, X)
fin_pretrain = time()

print("Tps d'exec pretrain : {}h {}m {}s".format(*hms(fin_pretrain - start_pretrain)))

start_train = time()
model.retropropagation(X, Y, 200, eps, tb)
fin_train = time()

print("Tps d'exec train : {}h {}m {}s".format(*hms(fin_pretrain - start_pretrain)))

save(model, note="60k")
