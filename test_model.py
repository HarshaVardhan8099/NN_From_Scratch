import pickle
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from data_loader import load_mnist


# Load trained model
with open("nn_mnist.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
X_train, X_test, y_train, y_test = load_mnist(use_gpu = True)

# Prediction
def predict(model, X):
    out = model.forward(X)
    return cp.argmax(out, axis = 1)

num_samples = 10
indices = np.random.choice(len(X_test), num_samples, replace = False)

X_samples = X_test[indices]
y_true = y_test[indices]
y_pred = predict(model, X_samples)

# plot results
plt.figure(figsize=(12,4))
for i in range(num_samples):
    plt.subplot(2, 5, i+1)
    plt.imshow(cp.asnumpy(X_samples[i].reshape(28, 28)), cmap="gray")
    plt.title(f"True:{int(y_true[i])}, Pred:{int(y_pred[i])}")
    plt.axis("off")

plt.tight_layout()
plt.show()