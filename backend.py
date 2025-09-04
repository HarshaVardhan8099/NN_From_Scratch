
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print("[INFO] Using GPU with CuPy")

except ImportError:
    import numpy as np
    xp = np
    GPU_AVAILABLE = False
    print("[INFO] CuPy not found, falling back to CPU with NumPy")
