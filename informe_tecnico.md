# Implementación de una Red Neuronal Multicapa desde Cero para la Clasificación de MNIST

## 1. Introducción

El objetivo de este proyecto fue implementar desde cero una red neuronal multicapa capaz de clasificar imágenes del conjunto de datos MNIST, utilizando Python y NumPy, sin utilizar frameworks de Deep Learning como TensorFlow o PyTorch.

El proyecto busca comprender de forma práctica los principales componentes matemáticos y computacionales de una red neuronal: propagación hacia adelante, funciones de activación, función de pérdida, retropropagación y descenso del gradiente.

## 2. Dataset

### 2.1 MNIST

Descripción del dataset:

- 70 000 imágenes de dígitos manuscritos.
- Imágenes de 28 × 28 píxeles.
- 10 clases: 0–9.
- Cada imagen se transforma en un vector de 784 características.

### 2.2 Preparación de los datos

Los valores de intensidad de los píxeles de MNIST se encuentran originalmente en el intervalo $[0,255]$. Para normalizar las entradas cada valor fue convertido a `float32` y dividido entre 255, obteniendo valores en el intervalo $[0,1]$.

Las etiquetas fueron posteriormente transformadas mediante One-Hot Encoding. Cada clase se representa mediante un vector de 10 posiciones, donde la posición correspondiente a la clase contiene el valor 1 y las restantes contienen 0.

En este proyecto se utilizaron 10 000 muestras para entrenamiento y 60 000 para evaluación.


## 3. Arquitectura del modelo

### 3.1 Capas ocultas

Las tres capas ocultas utilizan la función de activación ReLU.
 - Primera capa: $X \rightarrow Z_{1} = X W_{1} + b_{1} \rightarrow \mathrm{ReLU}(Z_{1}) = H_{1}$
 - Segunda capa: $H_{1} \rightarrow Z_{2} = H_{1} W_{2} + b_{2} \rightarrow \mathrm{ReLU}(Z_{2}) = H_{2}$
 - Tercera capa: $H_{2} \rightarrow Z_{3} = H_{2} W_{3} + b_{3} \rightarrow \mathrm{ReLU}(Z_{3}) = H_{3}$

### 3.2 Capa de salida

La capa de salida contiene 10 neuronas, una por cada clase de MNIST,
y utiliza Softmax para obtener una distribución de probabilidades.
- Capa de salida: $H_{3} \rightarrow Z_{o} = H_{3}  W_{o} + b_{o} \rightarrow \mathrm{Softmax}(Z_{o}) = Output$


## 4. Entrenamiento

### 4.1 Propagación hacia adelante 

En cada capa se calcula:

$$
Z = XW + b
$$

y posteriormente se aplica la función de activación:

$$
H = f(Z)
$$

El proceso se repite hasta obtener las probabilidades de la capa de
salida.

### 4.2 Función de pérdida
Se utilizó Cross-Entropy como función de pérdida:

$$
L = -\frac{1}{N}\sum_{n=1}^{N}\sum_{i=1}^{10}y_{n,i}\log(p_{n,i})
$$

### 4.3 Retropropagación
Los gradientes se calcularon mediante la regla de la cadena.
Para la combinación Softmax + Cross-Entropy:

$$
\frac{\partial L}{\partial Z_o} = P-Y
$$

A partir de este gradiente se calcularon los gradientes de pesos y
sesgos de cada capa.

La propagracion hacia atras se desglosa de la siguiente manera:

$$
\frac{\partial L}{\partial b_o}
=
\frac{1}{N}\sum_{n=1}^{N}
\frac{\partial L}{\partial Z_{o,n}}
$$

$$
\frac{\partial L}{\partial W_o}
=
\frac{1}{N}
H_3^T
\frac{\partial L}{\partial Z_o}
$$

$$
\frac{\partial L}{\partial H_3}
=
\frac{\partial L}{\partial Z_o}W_o^T
$$

Para atravesar la función de activación ReLU:

$$
\frac{\partial L}{\partial Z_3}
=
\frac{\partial L}{\partial H_3}
\odot
\mathrm{ReLU}'(Z_3)
$$

A partir de este gradiente se calculan los parámetros de la tercera
capa oculta:

$$
\frac{\partial L}{\partial b_3}
=
\frac{1}{N}
\sum_{n=1}^{N}
\frac{\partial L}{\partial Z_{3,n}}
$$

$$
\frac{\partial L}{\partial W_3}
=
\frac{1}{N}
H_2^T
\frac{\partial L}{\partial Z_3}
$$

$$
\frac{\partial L}{\partial H_2}
=
\frac{\partial L}{\partial Z_3}W_3^T
$$

El mismo procedimiento se repite para las capas anteriores:

$$
\frac{\partial L}{\partial Z_2}
=
\frac{\partial L}{\partial H_2}
\odot
\mathrm{ReLU}'(Z_2)
$$

$$
\frac{\partial L}{\partial b_2}
=
\frac{1}{N}
\sum_{n=1}^{N}
\frac{\partial L}{\partial Z_{2,n}}
$$

$$
\frac{\partial L}{\partial W_2}
=
\frac{1}{N}
H_1^T
\frac{\partial L}{\partial Z_2}
$$

$$
\frac{\partial L}{\partial H_1}
=
\frac{\partial L}{\partial Z_2}W_2^T
$$

Finalmente:

$$
\frac{\partial L}{\partial Z_1}
=
\frac{\partial L}{\partial H_1}
\odot
\mathrm{ReLU}'(Z_1)
$$

$$
\frac{\partial L}{\partial b_1}
=
\frac{1}{N}
\sum_{n=1}^{N}
\frac{\partial L}{\partial Z_{1,n}}
$$

$$
\frac{\partial L}{\partial W_1}
=
\frac{1}{N}
X^T
\frac{\partial L}{\partial Z_1}
$$


Este proceso se repite hasta alcanzar la primera capa oculta. Una vez calculado $\frac{\partial L}{\partial Z_1}$ , se obtienen los gradientes correspondientes a $W_1$ y $b_1$. No es necesario continuar el cálculo para actualizar los parámetros, ya que $X$ corresponde a la entrada de la red y no es un parámetro entrenable.


### 4.4 Actualización de parámetros
Los parámetros fueron actualizados mediante descenso del gradiente:

$$
W = W - \eta \nabla W
$$

$$
b = b - \eta \nabla b
$$

Donde $\eta$ representa el learning rate.


## 5. Implementación

El proyecto fue dividido en diferentes archivos para separar las responsabilidades:
```
data_MNIST.py
    ↓
Carga y preparación de los datos

funciones_MNIST.py
    ↓
Funciones matemáticas y operaciones de la red

MNIST_main.py
    ↓
Entrenamiento del modelo

test_MNIST.py
    ↓
Evaluación del modelo

pesos_modelo.npz
    ↓
Parámetros entrenados
```
### 5.1 Funciones principales

Entre las funciones implementadas se encuentran:
- relu()
- grad_relu()
- sigmoid()
- grad_sigmoid()
- softmax()
- cross_entropy()
- hidden_neuron_matrices()
- output_neuron()
- gradiente_hidden_w_b()
- actualizar()

Entre las funciones implementadas se encuentran las funciones de activación ReLU y Sigmoid, aunque la arquitectura final utilizada para MNIST emplea ReLU en las capas ocultas.

Todas las operaciones principales de la red fueron implementadas utilizando NumPy.


## 6. Evaluación
El modelo fue evaluado utilizando:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 6.1 Resultados

|  Métrica  | Resultado |
| --------- | --------- |
| Accuracy  | 94.95 %   |
| Precision | 94.92 %   |
| Recall    | 94.89 %   |
| F1-Score  | 94.90 %   |
| ROC-AUC   | 99.76 %   |


### 6.2 Interpretación

El modelo obtuvo una exactitud (Accuracy) de aproximadamente 95 % sobre las muestras utilizadas para evaluación.

El ROC-AUC alcanzó un valor de 0.9976, indicando una elevada capacidad del modelo para separar las diferentes clases utilizando las probabilidades producidas por la capa Softmax.


## 7. Resultados y observaciones

La implementación permitió comprobar que una red neuronal multicapa puede construirse utilizando únicamente operaciones matriciales y funciones matemáticas implementadas con NumPy.
Uno de los aspectos principales del proyecto fue comprender la relación entre las dimensiones de las matrices, la propagación de los datos hacia adelante y el cálculo de los gradientes durante la retropropagación.
También se comprobó experimentalmente la importancia de utilizar una implementación estable de Softmax y de calcular correctamente los gradientes promedio sobre el batch.


## 8. Limitaciones

El modelo presenta algunas limitaciones:
- Se utilizó una arquitectura MLP en lugar de una CNN.
- El entrenamiento se realizó mediante Full-Batch Gradient Descent.
- Solo se utilizaron 10 000 imágenes para el entrenamiento.
- No se utilizaron optimizadores como Momentum o Adam.
- No se implementó regularización.
- El modelo no utiliza aceleración mediante GPU.


## 9. Trabajo futuro

Como posibles extensiones del proyecto se plantea:
- Implementar Mini-Batch Gradient Descent.
- Implementar Momentum desde cero.
- Implementar Adam desde cero.
- Incorporar una matriz de confusión.
- Realizar Gradient Checking.
- Implementar una CNN desde cero utilizando NumPy.
- Comparar el rendimiento del MLP con una CNN.
- Comparar la implementación propia con frameworks como PyTorch.


## 10. Conclusiones

Se implementó exitosamente una red neuronal multicapa para la clasificación de imágenes MNIST utilizando Python y NumPy, sin recurrir a frameworks especializados de Deep Learning.
La implementación permitió estudiar de manera práctica los principales componentes de una red neuronal, incluyendo propagación hacia adelante, funciones de activación, Cross-Entropy, retropropagación y descenso del gradiente.
El modelo alcanzó un Accuracy de 94.95 % y un ROC-AUC de 99.76 % en el conjunto utilizado para evaluación.
El proyecto demuestra que los mecanismos fundamentales de una red neuronal pueden implementarse directamente mediante operaciones matriciales, proporcionando una comprensión más profunda de su funcionamiento interno.

