import os
import sys

# Check if the code is running in a known cloud environment like GitHub Codespaces or Google Colab.
# These environments often require specific setup or don't have a GPU, so we default to NumPy.
if 'CODESPACES' in sys.modules or os.environ.get('CODESPACES') == 'True' or 'google.colab' in sys.modules:
    import numpy as np
    xp = np # 'xp' now stands for numpy
    GPU_AVAILABLE = False
    print("[INFO] Running in GitHub Codespaces, forcing CPU with NumPy")

# If not in a known cloud environment, try to use a GPU with CuPy.
else:
    try:
        import cupy as cp # Try to import the CuPy library.
        try:
            # Check if there is at least one CUDA-enabled GPU available.
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                xp = cp # 'xp' now stands for cupy
                GPU_AVAILABLE = True
                print(f"[INFO] Using GPU with CuPy (found {device_count} device(s))")
            else:
                # If CuPy is installed but no GPU is found, raise an error to trigger the fallback.
                raise RuntimeError("No CUDA devices found")

        except Exception as e:
            # This block handles the case where CuPy is installed, but a GPU is not usable.
            # It catches the RuntimeError and any other potential issues.
            import numpy as np
            xp = np # Fall back to NumPy.
            GPU_AVAILABLE = False
            print(f"[WARN] CuPy imported but GPU not usable ({e}), falling back to CPU with NumPy")

    except ImportError:
        # This block runs if CuPy is not installed at all.
        import numpy as np
        xp = np # Fall back to NumPy.
        GPU_AVAILABLE = False
        print("[INFO] CuPy not found, falling back to CPU with NumPy")