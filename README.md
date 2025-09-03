# NN_From_Scratch
Neural network implemented from scratch in Python
# Neural Network on MNIST (from Scratch with CuPy)

This project implements a **fully-connected neural network** trained on the MNIST dataset, built **from scratch** using only:
- [CuPy](https://cupy.dev/) (for GPU acceleration, NumPy-like syntax)
- NumPy (for CPU fallback, optional)
- Matplotlib (for visualization)

⚡ No deep learning frameworks (TensorFlow, PyTorch, Keras, etc.) were used.

---

## ✨ Features
- Forward and backward propagation implemented manually.
- Vectorized matrix operations for efficiency.
- Training and testing pipeline.
- Save & load trained model with `pickle`.
- GPU acceleration with CuPy (CPU fallback planned).

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
