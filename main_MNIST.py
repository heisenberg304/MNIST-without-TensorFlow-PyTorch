#Para MNIST usamos una distribución de probabilidades, por lo que he reemplazado MSE, por cross entropy en este caso
#Cross-Entropy está diseñada precisamente para comparar distribuciones de probabilidad
#
import numpy as np
from data_MNIST import x, y
from funciones_MNIST import relu, grad_relu, cross_entropy, grad_softmax, hidden_neuron_matrices, output_neuron, gradiente_hidden_w_b, actualizar, test_metrics

y_one_hot = np.zeros((y.size, 10))
y_one_hot[np.arange(y.size), y] = 1
y = y_one_hot

x_train = x[:10000] #(10000, 784)
y_train = y[:10000] #(10000, 10)

# Pesos de las neuronas de las capas Hidden y output
input_size=784
hidden1=100
hidden2=100
hidden3=50
output_size=10
learning_rate = 0.4

def activacion_de_pesos(input_size, hidden):
    w = np.random.randn(input_size, hidden).astype(np.float32) * np.sqrt(2.0 / input_size)
    b = np.zeros((1, hidden), dtype=np.float32)
    return w,b
w1, b1 = activacion_de_pesos(input_size, hidden1)
w2, b2 = activacion_de_pesos(hidden1, hidden2)
w3, b3 = activacion_de_pesos(hidden2, hidden3)
wo, bo = activacion_de_pesos(hidden3, output_size)


def main(x, wo, bo, w1, b1, w2, b2, w3, b3, y, funcion_de_activacion=relu, grad_funcion_de_activacion=grad_relu):
    z1, h1 = hidden_neuron_matrices(w1, x, b1, funcion_de_activacion)
    z2, h2 = hidden_neuron_matrices(w2, h1, b2, funcion_de_activacion)
    z3, h3 = hidden_neuron_matrices(w3, h2, b3, funcion_de_activacion)
    zo, output = output_neuron(wo, h3, bo)

    funcion_loss = cross_entropy(output, y)

    grad_zo = grad_softmax(output, y)
    grad_bo = np.mean(grad_zo, axis=0)
    grad_wo = h3.T @ grad_zo / x.shape[0]

    grad_z3, grad_b3, grad_w3 = gradiente_hidden_w_b(grad_zo, wo, z3, h2, x, grad_funcion_de_activacion)
    grad_z2, grad_b2, grad_w2 = gradiente_hidden_w_b(grad_z3, w3, z2, h1, x, grad_funcion_de_activacion)
    grad_z1, grad_b1, grad_w1 = gradiente_hidden_w_b(grad_z2, w2, z1, x, x, grad_funcion_de_activacion)

    wo_new = actualizar(learning_rate, wo, grad_wo)
    bo_new = actualizar(learning_rate, bo, grad_bo)
    w3_new = actualizar(learning_rate, w3, grad_w3)
    b3_new = actualizar(learning_rate, b3, grad_b3)
    w2_new = actualizar(learning_rate, w2, grad_w2)
    b2_new = actualizar(learning_rate, b2, grad_b2)
    w1_new = actualizar(learning_rate, w1, grad_w1)
    b1_new = actualizar(learning_rate, b1, grad_b1)
    return grad_wo, grad_bo, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, output, funcion_loss, wo_new, bo_new, w3_new, b3_new, w2_new, b2_new, w1_new, b1_new

grad_wo, grad_bo, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, output, funcion_loss, wo_new, bo_new, w3_new, b3_new, w2_new, b2_new, w1_new, b1_new = main(x_train, wo, bo, w1, b1, w2, b2, w3, b3, y_train, relu, grad_relu)


for epoch in range(1001):
    grad_wo, grad_bo, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, output, funcion_loss, wo, bo, w3, b3, w2, b2, w1, b1 = main(x_train, wo, bo, w1, b1, w2, b2, w3, b3, y_train, relu, grad_relu)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}")
        print(f"Loss: {funcion_loss}")
        print(f"Predicción: {np.argmax(output, axis=1)}")

    if epoch % 1000 == 0:
        np.savez(
            "pesos_modelo.npz",
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            w3=w3,
            b3=b3,
            wo=wo,
            bo=bo
        )


