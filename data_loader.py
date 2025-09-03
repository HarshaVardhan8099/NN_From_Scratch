import cupy as cp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(use_gpu=True):
    mnist = fetch_openml('mnist_784', version = 1, as_frame = False)
    X, y = mnist["data"], mnist["target"].astype(cp.int32 if use_gpu else int)

    X = X / 255.0
    X = X.reshape(-1, 784)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 1/7, random_state = 42
    )

    if use_gpu:
        X_train, X_test = cp.asarray(X_train), cp.asarray(X_test)
        y_train, y_test = cp.asarray(y_train), cp.asarray(y_test)

    return X_train, X_test, y_train, y_test