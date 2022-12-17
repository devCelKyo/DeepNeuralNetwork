import scipy.io
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle

def lire_alpha_digit(carac):
    data = scipy.io.loadmat('data/binaryalphadigs.mat')['dat']
    X = []
    for c in carac:
        for i in range(len(data[c])):
            X.append(data[c, i].flatten())
    return np.array(X)

def readTrainMNIST():
    X = np.load('data/m_train_data.npy')
    Y = np.load('data/m_train_label.npy')

    return X, Y

def readTestMNIST():
    X = np.load('data/m_test_data.npy')
    Y = np.load('data/m_test_label.npy')

    return X, Y

def label_array(Y):
    new_Y = np.zeros((Y.shape[0], 10))
    for i in range(len(Y)):
        new_Y[i, Y[i]] = 1

    return new_Y

def err_quad(X, Y):
    somme = 0
    n = X.shape[0]*X.shape[1]
    for i, j in np.ndindex(X.shape):
        somme += (X[i, j] - Y[i, j])**2

    return somme/n

def show(img, titre="default", dim2=16):
    img *= 255
    img = np.reshape(img, (-1, dim2))
    img = img.astype(np.uint8)
    cv.imshow(titre, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def shuffle_two(A, B):
    permutation = np.random.permutation(A.shape[0])
    shuffled_A = np.zeros(A.shape)
    shuffled_B = np.zeros(B.shape)
    for i in range(A.shape[0]):
        shuffled_A[i] = A[permutation[i]]
        shuffled_B[i] = B[permutation[i]]

    return shuffled_A, shuffled_B

def save(NN, note=None):
    net_type = NN.__class__.__name__
    filename = "{}-"
    for dim in NN.dims:
        filename += "{}x"
    filename = filename[:-1]
    filename = filename.format(net_type, *NN.dims)
    if NN.pretrained:
        filename += "-pretrained"
    if note is not None:
        filename += "-{}".format(note)
    path = "Networks/saves/{}".format(filename)
    filehandler = open(path, "wb")
    pickle.dump(NN, filehandler)

def load(filename):
    path = "Networks/saves/{}".format(filename)
    filehandler = open(path, "rb")
    
    return pickle.load(filehandler)
