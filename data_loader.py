from backend import xp, GPU_AVAILABLE
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist():
    """
    Loads and preprocesses the MNIST dataset.
    
    Returns:
        tuple: A tuple containing four xp.ndarrays: (X_train, X_test, y_train, y_test).
    """
    # Fetch the MNIST dataset from OpenML.
    # 'mnist_784' is a flattened version of the 28x28 images.
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    # Get the data (X) and labels (y). Convert labels to the correct data type.
    X, y = mnist["data"], mnist["target"].astype(xp.int32)

    # Normalize the pixel values from [0, 255] to [0, 1] for better training stability.
    X = X / 255.0
    
    # Reshape the data to ensure it's a 2D array (num_samples, num_features).
    X = X.reshape(-1, 784)

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/7, random_state=42
    )

    # If a GPU is available, convert the NumPy arrays to CuPy arrays for GPU processing.
    if GPU_AVAILABLE:
        X_train, X_test = xp.asarray(X_train), xp.asarray(X_test)
        y_train, y_test = xp.asarray(y_train), xp.asarray(y_test)

    return X_train, X_test, y_train, y_test