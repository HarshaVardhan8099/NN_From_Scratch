# NN_From_Scratch
Neural network implemented from scratch in Python
# Neural Network on MNIST (from Scratch with CuPy)

This repository contains a simple, from-scratch implementation of a feedforward neural network. The key feature of this project is its dynamic backend, which automatically uses **CuPy for GPU acceleration** if a CUDA-enabled device is available, and falls back to **NumPy for CPU-based computation** if not.

The network is designed to be **modular and easy to understand**, with a clear separation of concerns for different components like layers, activation functions, and loss functions. It is trained and evaluated on the classic **MNIST handwritten digits** dataset.

⚡ No deep learning frameworks (TensorFlow, PyTorch, Keras, etc.) were used.

---

## 🚀 Key Features

- **Adaptive Backend**: Seamlessly switches between CuPy (for GPU) and NumPy (for CPU) to optimize performance based on your system.
- **From-Scratch Implementation**: All core components, including dense layers, activations, and backpropagation, are built from the ground up.
- **Modular Design**: The codebase is organized into distinct files, making it easy to understand and extend.
- **MNIST Dataset**: Includes a data loader to handle fetching, preprocessing, and splitting the MNIST dataset.
- **Model Persistence**: The trained model can be saved to a file and loaded later for inference.

## 📁 Project Structure

```text
.
├── main.py           # Main script for training the model
├── model.py          # Defines the SimpleNN class and model flow
├── layers.py         # Implements Dense (fully connected) layers
├── activations.py    # ReLU, Softmax, and other activations
├── losses.py         # CrossEntropy loss function
├── data_loader.py    # MNIST data loading and preprocessing
├── backend.py        # Dynamic backend selector (NumPy or CuPy)
├── utils.py          # Utility functions (e.g., accuracy calculation)
├── test_model.py     # Load trained model and visualize predictions
└── nn_mnist.pkl      # Saved model file (after training)

```
---

## ✅ Prerequisites

- Python 3.7 or higher
- A CUDA-enabled GPU (optional, for GPU acceleration with CuPy)

## 🔧 Installation

1. **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```

2. **Install dependencies**:
    ```bash
    pip install numpy scikit-learn matplotlib
    ```

3. **(Optional) For GPU acceleration with CuPy**:
    Only if you have a CUDA-enabled GPU:
    ```bash
    pip install cupy-cuda11x  # Replace 11x with your CUDA version
    ```

> ⚠️ Note: Avoid installing CuPy in environments like GitHub Codespaces or Google Colab unless you're sure a compatible GPU is available.


## ⚙️ How the CPU/GPU Handling Works (`backend.py`)

The `backend.py` file handles automatic selection of the backend:

- **Main Variable (`xp`)**: Acts as an alias for either `numpy` or `cupy`.
- **Logic**:
  - Detects if running in a restricted cloud environment (e.g., Codespace).
  - Tries to import CuPy.
  - Checks for CUDA-compatible GPU.

**Result**:
- If GPU is available, `xp = cupy`.
- Otherwise, `xp = numpy`.

> This abstraction means the rest of the code doesn’t care whether it's running on a CPU or GPU — just use `xp` for all numerical operations.

## 🧠 Usage

### 1. Train and test the Model

    ```bash
    python main.py

    python test_model.py
    ```

---

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- NumPy, CuPy, Matplotlib, scikit-learn

---

