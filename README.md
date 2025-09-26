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
    
    python main.py

    python test_model.py

---

## 📤 Ouput - While Training

    [INFO] CuPy not found, falling back to CPU with NumPy
    Epoch 1/20, Loss=0.9523, Acc = 0.7667
    Epoch 2/20, Loss=0.4042, Acc = 0.8608
    Epoch 3/20, Loss=0.3339, Acc = 0.8838
    Epoch 4/20, Loss=0.3339, Acc = 0.8911
    Epoch 5/20, Loss=0.2471, Acc = 0.8994
    Epoch 6/20, Loss=0.2367, Acc = 0.9048
    Epoch 7/20, Loss=0.4560, Acc = 0.9099
    Epoch 8/20, Loss=0.3319, Acc = 0.9125
    Epoch 9/20, Loss=0.2036, Acc = 0.9153
    Epoch 10/20, Loss=0.3364, Acc = 0.9184
    Epoch 11/20, Loss=0.2244, Acc = 0.9221
    Epoch 12/20, Loss=0.3769, Acc = 0.9230
    Epoch 13/20, Loss=0.0955, Acc = 0.9255
    Epoch 14/20, Loss=0.1636, Acc = 0.9267
    Epoch 15/20, Loss=0.1315, Acc = 0.9304
    Epoch 16/20, Loss=0.3282, Acc = 0.9330
    Epoch 17/20, Loss=0.1046, Acc = 0.9340
    Epoch 18/20, Loss=0.2426, Acc = 0.9355
    Epoch 19/20, Loss=0.1741, Acc = 0.9375
    Epoch 20/20, Loss=0.0561, Acc = 0.9400
    Model saved as nn_mnist.pkl

## 📤Output - While Testing the Model

<img width="1200" height="400" alt="test_results" src="https://github.com/user-attachments/assets/7e7a867e-6c51-4806-92af-4343268f8124" />

---

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- NumPy, CuPy, Matplotlib, scikit-learn

---

