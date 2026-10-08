import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#funciones de activacion y sus gradientes
#Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def grad_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
#ReLU
def relu(z):
    return np.maximum(0, z)
def grad_relu(z):
    return (z > 0).astype(float)
#Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    p = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return p
#La derivada de Softmax sola es su Jacobiano 10*10, pero para simplificar usaremos:
def grad_softmax(p, y):
    return p - y

#para la prediccion de testeo
def predict(x,w,b):
    pred = np.dot(x, w) + b
    return pred

#Cross-entropy
def cross_entropy(p, y):
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y * np.log(p), axis=1))



#NEURONAS
#implementacion para las neuronas hidden
def hidden_neuron_matrices(w, x, b, funcion_de_activacion=relu):
  #XW+B
    z = np.dot(x, w) + b
    h = funcion_de_activacion(z)
    return z, h
#implementacion para las neuronas output
def output_neuron(w, x, b):
  #XW+B
    z = np.dot(x, w) + b
    h = softmax(z)
    return z, h



#GRADIENTES DE W & B
def gradiente_hidden_w_b(gradz, w, z, h, x, grad_funcion_de_activacion=grad_relu):
    grad_z = gradz @ w.T * grad_funcion_de_activacion(z)
    grad_b = np.mean(grad_z, axis=0)
    grad_w = h.T @ grad_z / x.shape[0]
    return grad_z, grad_b, grad_w

#actualizacion
def actualizar(lr, w_b, grad):
    new = w_b - (lr*grad)
    return new



def metrics(y_test, pred, average_type='binary'):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=average_type, zero_division=0)
    recall = recall_score(y_test, pred, average=average_type, zero_division=0)
    f1 = f1_score(y_test, pred, average=average_type, zero_division=0)
    return accuracy, precision, recall, f1

def test_metrics(w1, b1, w2, b2, w3, b3, wo, bo, y_test, x_test):
    z1, h1 = hidden_neuron_matrices(w1, x_test, b1)
    z2, h2 = hidden_neuron_matrices(w2, h1, b2)
    z3, h3 = hidden_neuron_matrices(w3, h2, b3)
    zo, output = output_neuron(wo, h3, bo)
    pred = np.argmax(output, axis=1)
    accuracy, precision, recall, f1 = metrics(y_test, pred, average_type='macro')
    roc_auc = roc_auc_score(y_test, output, multi_class='ovr', average='macro')
    return accuracy, precision, recall, f1, roc_auc
