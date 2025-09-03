
class SimpleNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, grad):
        for layer in reversed(self. layers):
            grad = layer.backward(grad)