from tools import readTrainMNIST, readTestMNIST
from tools import save, load, show_m, label_array, reformat

# On commence par importer les images d'entrainement et de test
images_train, labels_train = readTrainMNIST()
images_test, labels_test = readTestMNIST()

# On reformate les images 28x28 en vecteurs 784, on reformate les labels
# en vecteurs [0, 0, ..., 1, ..., 0]

X_train, Y_train = reformat(images_train, labels_train)
X_test, Y_test = reformat(images_test, labels_test)

# On affiche une image
show_m(images_train[0])

# Importons la classe DNN et initialisons un modèle de taille 784 x 200 x 200 x 10
from Networks.DNN import DNN

model = DNN((784, 200, 200, 10))
# Pré-entrainons le rapidement (devrait durer 30-40 secondes)
epochs, eps, tb = 5, 0.01, 5000
model.pretrain(epochs, eps, tb, X_train)

# Désormais, entraînons le de manière supervisée avec verbose=True pour voir l'évolution de
# la cross-entropie (devrait durer 1 minute aussi)
model.retropropagation(X_train, Y_train, epochs, eps, tb, verbose=True)

# Enfin, calculons la sortie d'une image de test
img, label = images_test[0], labels_test[0]
show_m(img)
sortie = model.entree_sortie(X_test[0])[-1]
print(sortie)
print("Vrai label : {}".format(label))

# Désormais, importons un modèle déjà bien entraîné et testons le sur une image, puis sur
# l'ensemble de la base de test
model = load("DNN-784x700x700x700x10-pretrained-trained-3500epochs-60k", special=True)
sortie = model.entree_sortie(X_test[0])[-1]
print(sortie)
print("Vrai label : {}".format(label))

accuracy = 1 - model.test(X_test, Y_test) # model.test renvoie le taux d'erreur sur X_test
print("Précision sur les 10000 images : {}%".format(100*accuracy))
