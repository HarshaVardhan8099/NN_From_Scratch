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

    [INFO] Using GPU with CuPy (found 1 device(s))
    Epoch 1/10, Loss=0.9467, Acc = 0.7852
    Epoch 2/10, Loss=0.3711, Acc = 0.8672
    Epoch 3/10, Loss=0.5757, Acc = 0.8867
    Epoch 4/10, Loss=0.2820, Acc = 0.8966
    Epoch 5/10, Loss=0.2373, Acc = 0.9035
    Epoch 6/10, Loss=0.2670, Acc = 0.9067
    Epoch 7/10, Loss=0.3872, Acc = 0.9102
    Epoch 8/10, Loss=0.2803, Acc = 0.9161
    Epoch 9/10, Loss=0.5226, Acc = 0.9181
    Epoch 10/10, Loss=0.6029, Acc = 0.9213
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

