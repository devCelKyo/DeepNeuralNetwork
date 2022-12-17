from tools import lire_alpha_digit, show, save, load
from Networks.RBM import RBM
from Networks.DBN import DBN

CARACS = [[10], [10, 11], [10, 11, 12], [10, 11, 12, 13]]
Q_RBMS = [100, 200, 300, 400]
Q_DBNS = [(100, 100), (200, 200), (100, 100, 100), (200, 200, 200), (300, 300, 300)]

epochs = 100
eps = 0.1 # Learning rate
tb = 1 # Taille batch

for carac in CARACS:
    X = lire_alpha_digit(carac)
    print(carac)
    for Q_RBM in Q_RBMS:
        print(Q_RBM)
        model = RBM((X.shape[1], Q_RBM))
        model.train(epochs, eps, tb, X, verbose=False)
        save(model, "{}_chars".format(len(carac)))
    for Q_DBN in Q_DBNS:
        print(Q_DBN)
        model = DBN((X.shape[1], *Q_DBN))
        model.train(epochs, eps, tb, X, verbose=False)
        save(model, "{}_chars".format(len(carac)))

print("ok")
