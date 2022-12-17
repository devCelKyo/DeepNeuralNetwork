from tools import lire_alpha_digit, show, save, load
from Networks.RBM import RBM
from Networks.DBN import DBN

X = lire_alpha_digit([10, 11, 12])
#dbn = DBN((X.shape[1], 100, 100, 100))
#dbn.train(100, 0.1, 3, X, False)
#img = dbn.generer_image(30, 10)

rbm = RBM((X.shape[1], 100))
rbm.train(100, 0.05, 3, X, False)
img = rbm.generer_image(30, 10)

show(img)
