from tools import load, label_array, readTestMNIST
import numpy as np
import matplotlib.pyplot as plt

X, Y = readTestMNIST()
X, Y = X.reshape((X.shape[0], 784)), label_array(Y)

def predict(models_p, models_np):
    err_p = []
    err_np = []

    for model in models_p:
        err_p.append(model.test(X, Y))
    for model in models_np:
        err_np.append(model.test(X, Y))

    return err_p, err_np

def fig1():
    index = [2, 3, 5]
    models_p = [load("DNN-784x200x200x10-pretrained-trained-200epochs-60k"),
                load("DNN-784x200x200x200x10-pretrained-trained-200epochs-60k"),
                load("DNN-784x200x200x200x200x200x10-pretrained-trained-200epochs-60k")]

    models_np = [load("DNN-784x200x200x10-trained-200epochs-60k"),
                load("DNN-784x200x200x200x10-trained-200epochs-60k"),
                load("DNN-784x200x200x200x200x200x10-trained-200epochs-60k")]

    err_p, err_np = predict_show(models_p, models_np)

    plt.plot(index, err_p, label="pretrained")
    plt.plot(index, err_np, label="not pretrained")
    plt.xticks(index)
    plt.legend()
    plt.xlabel('Nombre de couches (200 neurones)')
    plt.ylabel("Taux d'erreurs de classification")

    plt.title("Evolution du taux d'erreur sur les 10 000 images en fonction du nombre de couches")
    plt.show()

def fig2():
    index = [100, 200, 300, 700]
    models_p = [load("DNN-784x100x100x10-pretrained-trained-200epochs-60k"),
                load("DNN-784x200x200x10-pretrained-trained-200epochs-60k"),
                load("DNN-784x300x300x10-pretrained-trained-200epochs-60k"),
                load("DNN-784x700x700x10-pretrained-trained-200epochs-60k")]

    models_np = [load("DNN-784x100x100x10-trained-200epochs-60k"),
                load("DNN-784x200x200x10-trained-200epochs-60k"),
                load("DNN-784x300x300x10-trained-200epochs-60k"),
                load("DNN-784x700x700x10-trained-200epochs-60k")]

    err_p, err_np = predict(models_p, models_np)

    plt.plot(index, err_p, label="pretrained")
    plt.plot(index, err_np, label="not pretrained")
    plt.xticks(index)
    plt.legend()
    plt.xlabel('Nombre de neurones (2 couches)')
    plt.ylabel("Taux d'erreurs de classification")

    plt.title("Evolution du taux d'erreur sur les 10 000 images en fonction du nombre de neurones sur 2 couches")
    plt.show()

def fig3():
    index = ["1k", "3k", "7k", "10k", "30k", "60k"]
    models_p = [load("DNN-784x200x200x10-pretrained-trained-200epochs-1k"),
                load("DNN-784x200x200x10-pretrained-trained-200epochs-3k"),
                load("DNN-784x200x200x10-pretrained-trained-200epochs-7k"),
                load("DNN-784x200x200x10-pretrained-trained-200epochs-10k"),
                load("DNN-784x200x200x10-pretrained-trained-200epochs-30k"),
                load("DNN-784x200x200x10-pretrained-trained-200epochs-60k")]

    models_np = [load("DNN-784x200x200x10-trained-200epochs-1k"),
                load("DNN-784x200x200x10-trained-200epochs-3k"),
                load("DNN-784x200x200x10-trained-200epochs-7k"),
                load("DNN-784x200x200x10-trained-200epochs-10k"),
                load("DNN-784x200x200x10-trained-200epochs-30k"),
                load("DNN-784x200x200x10-trained-200epochs-60k")]

    err_p, err_np = predict(models_p, models_np)

    plt.plot(index, err_p, label="pretrained")
    plt.plot(index, err_np, label="not pretrained")
    plt.xticks(index)
    plt.legend()
    plt.xlabel("Nombre de données d'entrainement")
    plt.ylabel("Taux d'erreurs de classification")

    plt.title("Evolution du taux d'erreur sur les 10 000 images en fonction du nombre de données d'entrainement utilisées")
    plt.show()

fig3()
