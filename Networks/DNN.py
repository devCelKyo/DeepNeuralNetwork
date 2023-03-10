from Networks.DBN import DBN
from numpy.random import rand, binomial
from tools import shuffle_two, cross_entropie
import numpy as np
from copy import deepcopy

class DNN:
    def __init__(self, N_dims):
        self.DBN = DBN(N_dims)
        self.RBM = self.DBN.RBM
        self.W, self.b = [], []
        for rbm in self.RBM:
            self.W.append(rbm.W)
            self.b.append(rbm.b)
        self.dims = N_dims
        self.pretrained = False
        self.trained = False
        self.epochs = 0

    def getRBM(self):
        return self.DBN.RBM

    def getW(self, i):
        return self.DBN.RBM[i].W

    def setW(self, i, W):
        self.DBN.RBM[i].W = W
        return self

    def addW(self, i, added_W):
        return self.setW(i, self.getW(i) + added_W)
        
    def getb(self, i):
        return self.DBN.RBM[i].b

    def setb(self, i, b):
        self.DBN.RBM[i].b = b
        return self

    def addb(self, i, added_b):
        return self.setb(i, self.getb(i) + added_b)

    def pretrain(self, epochs, eps, tb, X):
        self.pretrained = True
        self.DBN.train(epochs, eps, tb, X, False)

    def entree_sortie(self, X):
        RBM = self.getRBM()
        N_couches = len(RBM)
        X_work = [np.copy(X)]
        for i in range(N_couches):
            if i != N_couches-1:
                X_work.append(RBM[i].entree_sortie(X_work[i]))
            else:
                X_work.append(RBM[i].calc_softmax(X_work[i]))
        return X_work

    def retropropagation(self, X, Y, epochs, eps, tb, verbose=False):
        n = X.shape[0]
        N_couches = len(self.DBN.RBM)
        self.trained = True
        new_DNN = deepcopy(self)
        for i in range(epochs):
            X, Y = shuffle_two(X, Y)
            for k in range(0, n, tb):
                X_b = X[k:min(n, k+tb),:]
                Y_b = Y[k:min(n, k+tb),:]
                sortie = self.entree_sortie(X_b)
                C = sortie[-1] - Y_b
                grad_W = np.transpose(sortie[-2])@C
                grad_b = np.sum(C, axis=0)
                new_DNN.addW(-1, -eps*grad_W/tb)
                new_DNN.addb(-1, -eps*grad_b/tb)
                for p in range(N_couches-2, -1, -1):
                    C = (C@np.transpose(self.getW(p+1)))*(sortie[p+1]*(1-sortie[p+1]))
                    grad_W = np.transpose(sortie[p])@C
                    grad_b = np.sum(C, axis=0)
                    new_DNN.addW(p, -eps*grad_W/tb)
                    new_DNN.addb(p, -eps*grad_b/tb)
                self.DBN = deepcopy(new_DNN.DBN)
            self.epochs += 1
            if verbose:
                sortie = self.entree_sortie(X)[-1]
                print(cross_entropie(Y, sortie))

    def test(self, X, Y):
        Y_est = self.entree_sortie(X)[-1]
        n = Y.shape[0]
        err = 0
        for i in range(n):
            if np.argmax(Y_est[i]) != np.argmax(Y[i]):
                err += 1
        return err/n
