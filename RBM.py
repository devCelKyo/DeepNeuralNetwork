import numpy as np
from numpy.random import rand, binomial
from tools import err_quad

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax(z, k):
    somme = 0
    for zi in z:
        somme += np.exp(-zi)
    return z[k]/somme

sigmoid = np.vectorize(sigmoid)

class RBM:
    def __init__(self, N_dims):
        self.W = np.random.normal(0, 0.01, size=N_dims)
        self.a = np.zeros(N_dims[0])
        self.b = np.zeros(N_dims[1])

    def entree_sortie(self, X):
        X_tilde = np.matmul(X, self.W)
        X_tilde += self.b

        return sigmoid(X_tilde)

    def sortie_entree(self, Y):
        X = np.matmul(Y, np.transpose(self.W))
        X += self.a

        return sigmoid(X)

    def train(self, epochs, eps, tb, X, verbose):
        n, p, q = len(X), len(self.a), len(self.b)
        for i in range(epochs):
            np.random.shuffle(X)
            for k in range(0, n, tb):
                X_b = X[k:min(n, k+tb),:]
                ph_v0 = self.entree_sortie(X_b)
                h_0 = (rand(ph_v0.shape[0], q) < ph_v0)*1
                pv_h0 = self.sortie_entree(h_0)
                v_1 = (rand(pv_h0.shape[0], p) < pv_h0)*1
                ph_v1 = self.entree_sortie(v_1)
                grad_W = np.transpose(X_b)@ph_v0 - np.transpose(v_1)@ph_v1
                grad_b = np.sum(ph_v0 - ph_v1, axis=0)
                grad_a = np.sum(X_b - v_1, axis=0)

                self.W += eps*grad_W
                self.b += eps*grad_b
                self.a += eps*grad_a
                if verbose:
                    print(err_quad(X_b, v_1))

    def generer_image(self, gibbs_iter, n_images):
        p, q = len(self.a), len(self.b)
        V = np.zeros((n_images, p))
        for i, j in np.ndindex(V.shape):
            V[i, j] = binomial(1, 0.5)
        for i in range(gibbs_iter):
            ph = self.entree_sortie(V)
            H = (rand(ph.shape[0], q) < ph)*1
            pv = self.sortie_entree(H)
            V = (rand(pv.shape[0], p) < pv)*1
        return V

    def calc_softmax(self, X):
        p, q = len(self.a), len(self.b)
        X_tilde = np.matmul(X, self.W)
        X_tilde += self.b
        Y = np.zeros((X.shape[0], q))
        for i, j in np.ndindex(Y.shape):
            Y[i, j] = softmax(X[i], j)
        return Y
