import numpy as np
from sklearn.datasets import fetch_openml
X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)
x = X.astype(np.float32) / 255.0
y = y.astype(np.int64)