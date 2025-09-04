import pickle
from data_loader import load_mnist
from layers import Dense
from activations import ReLU, Softmax
from losses import CrossEntropyLoss
from model import SimpleNN
from utils import accuracy
from backend import xp


# Load the MNIST data using the custom data loader.
X_train, X_test, y_train, y_test = load_mnist()

# Define the neural network architecture.
# This model has two Dense layers with a ReLU activation in between.
# The final layer uses Softmax for classification.
model = SimpleNN([
    Dense(784, 128, lr=0.01),  # Input layer (784 features) to hidden layer (128 neurons)
    ReLU(),                    # Activation function for the hidden layer
    Dense(128, 10, lr=0.01),   # Hidden layer (128 neurons) to output layer (10 classes)
    Softmax()                  # Softmax activation for the final output
])

# Define the loss function to be used during training.
loss_fn = CrossEntropyLoss()

# Set up training parameters.
epochs = 10
batch_size = 64

# Start the main training loop.
for epoch in range(epochs):
    # Shuffle the training data at the beginning of each epoch to prevent the model
    # from learning the order of the data.
    prem = xp.random.permutation(len(X_train))
    X_train, y_train = X_train[prem], y_train[prem]

    # Iterate through the data in batches.
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward Pass:
        # Pass the input batch through the network to get predictions.
        out = model.forward(X_batch)
        
        # Calculate the loss based on the predictions and true labels.
        loss = loss_fn.forward(out, y_batch)

        # Backward Pass:
        # Calculate the initial gradient from the loss function.
        grad = loss_fn.backward(out, y_batch)
        
        # Propagate the gradient back through the network to update weights.
        model.backward(grad)

    # Evaluation at the end of each epoch:
    # Use the test set to evaluate the model's performance on unseen data.
    preds = model.forward(X_test)
    acc = accuracy(preds, y_test)
    
    # Print the current loss and accuracy.
    print(f"Epoch {epoch+1}/{epochs}, Loss={loss:.4f}, Acc = {acc:.4f}")

# Save the trained model to a file using the pickle library.
# This allows the model to be re-used later without retraining.
with open("nn_mnist.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as nn_mnist.pkl")