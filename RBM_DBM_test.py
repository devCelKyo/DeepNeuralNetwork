from tools import lire_alpha_digit, show, save, load
from Networks.RBM import RBM
from Networks.DBN import DBN

caracs = [10, 11, 13, 14, 15]
X = lire_alpha_digit(caracs)
epochs = 100
eps = 0.1 # Learning rate
tb = 1 # Taille batch
Q_RBM = 100 # Nombre de neurones sur la couche de sortie (RBM)
Q_DBN = (200, 200, 200)

#model = RBM((X.shape[1], Q_RBM))
model = DBN((X.shape[1], *Q_DBN))

model.train(epochs, eps, tb, X, verbose=False) # verbose=True pour voir les EQM à chaque tour
img = model.generer_image(2, 64)

show(img)
#save(model, "{}_chars".format(len(caracs)))
