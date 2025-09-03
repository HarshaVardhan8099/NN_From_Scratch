import cupy as cp

class ReLU:

    def forward(self, X):
        self.mask = (X > 0)
        return cp.maximum(0, X)

    def backward(self, dOut):
        return dOut * self.mask


class Softmax:

    def forward(self, X):
        exps = cp.exp(X - cp.max(X, axis = 1, keepdims = True))
        self.out = exps / cp.sum(exps, axis = 1, keepdims = True)
        return self.out

    def backward(self, dOut):
        return dOut