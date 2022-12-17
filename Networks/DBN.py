from Networks.RBM import RBM
from numpy.random import rand, binomial
import numpy as np

class DBN:
    def __init__(self, N_dims):
        self.RBM = [RBM((N_dims[i-1], N_dims[i])) for i in range(1, len(N_dims))]
        self.dims = N_dims
        self.pretrained = False
        
    def train(self, epochs, eps, tb, X, verbose):
        self.pretrained = True
        X_train = np.copy(X)
        for rbm in self.RBM:
            rbm.train(epochs, eps, tb, X_train, verbose)
            X_train = rbm.entree_sortie(X_train)

    def generer_image(self, gibbs_iter, n_images):
        last_rbm = self.RBM[-1]
        H = last_rbm.generer_image(gibbs_iter, n_images)
        for i in range(2, len(self.RBM)+1):
            ph = self.RBM[-i].sortie_entree(H)
            q = len(self.RBM[-i].a)
            H = (rand(ph.shape[0], q) < ph)*1
        return H
