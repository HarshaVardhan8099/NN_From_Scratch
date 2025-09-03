import cupy as cp
import pickle
from data_loader import load_mnist
from layers import Dense
from activations import ReLU, Softmax
from losses import CrossEntropyLoss
from model import SimpleNN
from utils import accuracy



# Load Data
X_train, X_test, y_train, y_test = load_mnist(use_gpu=True)

# Define Model
model = SimpleNN([
    Dense(784, 128,lr = 0.01),
    ReLU(),
    Dense(128, 10, lr=0.01),
    Softmax()
])

loss_fn = CrossEntropyLoss()

# Training Loop
epochs = 5
batch_size = 64

for epoch in range(epochs):
    prem = cp.random.permutation(len(X_train))
    X_train, y_train = X_train[prem], y_train[prem]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward
        out = model.forward(X_batch)
        loss = loss_fn.forward(out, y_batch)

        # backward
        grad = loss_fn.backward(out, y_batch)
        model.backward(grad)

    # Eval
    preds = model.forward(X_test)
    acc = accuracy(preds, y_test)
    print(f"Epoch {epoch+1}/{epochs}, Loss={loss: 4f}, Acc = {acc: 4f}")

# Save model
with open("nn_mnist.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as nn_mnist.pkl")
