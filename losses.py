import cupy as cp

class CrossEntropyLoss:
    def forward(self, preds, targets):
        m = preds.shape[0]
        log_likelihood = -cp.log(preds[cp.arange(m), targets] + 1e-9)
        loss = cp.sum(log_likelihood) / m
        return loss

    def backward(self, preds, targets):
        m = preds.shape[0]
        grad = preds.copy()
        grad[cp.arange(m), targets] -= 1
        grad /= m
        return grad
