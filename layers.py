import cupy as cp

class Dense:
    def __init__(self, in_features, out_features, lr=0.01):
        self.W = cp.random.randn(in_features, out_features) * 0.01
        self.b = cp.zeros((1, out_features))
        self.lr = lr

    def forward(self, X):
        self.X = X
        return X @self.W + self.b

    def backward(self, dOut):
        dW = self.X.T @ dOut
        db = cp.sum(dOut, axis = 0, keepdims = True)
        dX = dOut @ self.W.T

        # Update weights
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dX