import numpy as np
from data_MNIST import x, y
from funciones_MNIST import test_metrics
pesos = np.load("pesos_modelo.npz")

w1 = pesos["w1"]
b1 = pesos["b1"]
w2 = pesos["w2"]
b2 = pesos["b2"]
w3 = pesos["w3"]
b3 = pesos["b3"]
wo = pesos["wo"]
bo = pesos["bo"]

x = x[10000:70000]
y = y[10000:70000]

accuracy, precision, recall, f1, roc_auc = test_metrics(w1, b1, w2, b2, w3, b3, wo, bo, y, x)
print(
    f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nroc auc: {roc_auc}"
    )

#terminal: 
#Accuracy: 0.9495166666666667
#Precision: 0.9492372056806838
#Recall: 0.9489489243338192
#F1 Score: 0.9490445479347451
#roc auc: 0.9976377988875319