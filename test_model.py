import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from backend import xp, GPU_AVAILABLE
from data_loader import load_mnist


# Load the pre-trained model from the saved pickle file.
# The model is loaded into memory, ready for inference.
with open("nn_mnist.pkl", "rb") as f:
    model = pickle.load(f)

# Load the test data to evaluate the model.
# Note: The data loader handles moving the data to GPU if available.
X_train, X_test, y_train, y_test = load_mnist()

# Define a function to make predictions.
def predict(model, X):
    """
    Makes predictions on a given dataset using the trained model.
    """
    # Perform a forward pass to get the output probabilities.
    out = model.forward(X)
    
    # Return the index of the highest probability for each sample.
    return xp.argmax(out, axis=1)

# Select a random sample of images from the test set for visualization.
num_samples = 10
indices = np.random.choice(len(X_test), num_samples, replace=False)

# Get the samples and their corresponding true labels.
X_samples = X_test[indices]
y_true = y_test[indices]

# Make a prediction for each of the selected samples.
y_pred = predict(model, X_samples)

# Create a figure to plot the images.
plt.figure(figsize=(12, 4))
for i in range(num_samples):
    # Check if a GPU is available to decide whether to use xp.asnumpy().
    # Matplotlib does not support plotting CuPy arrays directly.
    if GPU_AVAILABLE:
        # If GPU is available (xp is cupy), convert to numpy for plotting.
        image_data = xp.asnumpy(X_samples[i].reshape(28, 28))
        true_label = int(xp.asnumpy(y_true[i]))
        pred_label = int(xp.asnumpy(y_pred[i]))
    else:
        # If no GPU is available (xp is numpy), the data is already in numpy format.
        image_data = X_samples[i].reshape(28, 28)
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])

    # Plot the image.
    plt.subplot(2, 5, i + 1)
    plt.imshow(image_data, cmap="gray")
    
    # Add a title with the true and predicted labels.
    plt.title(f"True:{true_label}, Pred:{pred_label}")
    
    # Hide the axes.
    plt.axis("off")

# Adjust the plot layout to prevent titles from overlapping.
plt.tight_layout()

# Corrected logic:
# In a Codespaces environment, we cannot show the plot in a new window, so we save it as a file.
# In a local environment, we can show the plot directly.
# The check `os.environ.get('CODESPACES') == 'true'` is the reliable way to detect Codespaces.
if os.environ.get('CODESPACES') == 'true':
    # Save the plot to a file in Codespaces.
    plt.savefig('test_results.png')
    print("Output saved in 'test_results.png'")
else:
    # Show the plot window in a local environment.
    plt.show()
