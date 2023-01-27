import scipy.io
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
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

    return X/255, Y

def readTestMNIST():
    X = np.load('data/m_test_data.npy')
    Y = np.load('data/m_test_label.npy')

    return X/255, Y

def label_array(Y):
    if type(Y) == np.uint8:
        new_Y = np.zeros(10)
        new_Y[Y] = 1
    else:
        new_Y = np.zeros((Y.shape[0], 10))
        for i in range(len(Y)):
            new_Y[i, Y[i]] = 1
    return np.array(new_Y)

def err_quad(X, Y):
    somme = 0
    n = X.shape[0]*X.shape[1]
    for i, j in np.ndindex(X.shape):
        somme += (X[i, j] - Y[i, j])**2

    return somme/n   

def cross_entropie(Y, Y_pred):
    somme = 0
    n = Y.shape[0]
    for i, j in np.ndindex(Y.shape):
        somme -= Y[i, j]*np.log(Y_pred[i, j])
    return somme/n
    
def show(img, titre="default", dim1=20, dim2=16, save=False, filename="foo", show=True):
    image = np.copy(img)
    image *= 255
    N_images = image.shape[0]
    N_grid = int(np.sqrt(N_images))
    M_grid = int(np.ceil(N_images/N_grid))
    N1 = N_grid*M_grid*dim1*dim2
    N2 = N_images*dim1*dim2
    image = np.reshape(image, -1)
    image = np.append(image, np.zeros(N1 - N2))
    image = np.reshape(image, (N_grid, M_grid, dim1, dim2))
    image = image.swapaxes(1, 2)
    image = np.reshape(image, (N_grid*dim1, -1))
    image = image.astype(np.uint8)
    if save:
        cv.imwrite(filename, image)
    if show:
        cv.imshow(titre, image)
        cv.waitKey(0)
        cv.destroyAllWindows()

def show_m(img):
    '''
    Pour afficher les images de la BDD MNIST
    '''
    dim1, dim2 = img.shape
    colors = ["white", "black"]
    cmap = mpl.colors.ListedColormap(colors)
    plt.axis('off')
    plt.imshow(img, cmap=cmap)
    plt.show()

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
    if net_type == "DNN" and NN.trained:
        filename += "-trained-{}epochs".format(NN.epochs)
    if note is not None:
        filename += "-{}".format(note)
    
    path = "Networks/saves/{}".format(filename)
    filehandler = open(path, "wb")
    pickle.dump(NN, filehandler)
    filehandler.close()

def load(filename):
    path = "Networks/saves/{}".format(filename)
    filehandler = open(path, "rb")
    
    return pickle.load(filehandler)

def hms(duree):
    h = int(duree/3600)
    m = int((duree%3600)/60)
    s = int(duree%60)

    return h, m, s

def reformat(X, Y):
    return X.reshape((X.shape[0], 784)), label_array(Y)
