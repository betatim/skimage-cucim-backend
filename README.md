# A CuCIM backend for scikit-image

An experimental backend that dispatches scikit-image calls to [CuCIM](https://github.com/rapidsai/cucim), so you can accelerate your existing scikit-image code on the GPU without changing your API. Use the same `skimage` functions; pass [CuPy](https://cupy.dev/) arrays instead of NumPy, and supported calls run on the GPU automatically.

## Installation

Use conda (rapidsai and conda-forge) for the CUDA-related stack; it is simpler than mixing pip and conda. Install the backend with pip.

1. Install **scikit-image** and **cucim** from conda (this brings in CuPy and a CUDA-compatible environment). Use the rapidsai channel first, then conda-forge:
   ```bash
   conda install -c rapidsai -c conda-forge scikit-image cucim
   ```
2. Install **skimage-cucim-backend** with pip: `pip install skimage-cucim-backend` (when published), or from a local clone: `pip install -e /path/to/skimage-cucim-backend`

Install scikit-image and cucim from conda first (rapidsai and conda-forge) so you get a consistent CUDA stack; then install the backend with pip.

## How it works

You use the normal scikit-image API. When you call a dispatchable function with **CuPy** array arguments and this backend supports that function, the call is dispatched to CuCIM and runs on the GPU. If you pass NumPy arrays, or the function is not supported by the backend, scikit-image’s usual implementation runs (on the CPU). No code changes are needed beyond using CuPy arrays for the inputs you want accelerated. To disable dispatching entirely, set the environment variable `SKIMAGE_NO_DISPATCHING=1`.

## Example

```python
import cupy as cp
from skimage import data, filters, exposure

# Load an image (NumPy array)
img = data.chelsea()

# Transfer to GPU: CuPy arrays trigger dispatch to CuCIM when the backend is installed
img_gpu = cp.asarray(img)

# Same scikit-image API — these run on the GPU
blurred = filters.gaussian(img_gpu, sigma=2)
equalized = exposure.equalize_hist(img_gpu)

# Results are CuPy arrays; copy back to CPU if needed
blurred_cpu = cp.asnumpy(blurred)
```

You may see a `DispatchNotification` warning when a call is dispatched to CuCIM; that is expected. For metrics, you get CuPy scalars (e.g. 0-dimensional arrays): `skimage.metrics.mean_squared_error(im1_gpu, im2_gpu)` returns a CuPy scalar when both inputs are CuPy.

## Supported functions

Many functions in `skimage.metrics`, `skimage.filters`, `skimage.exposure`, `skimage.transform`, `skimage.feature`, and `skimage.morphology` are supported. The full list is in `skimage_cucim_backend.information.SUPPORTED_FUNCTIONS`.

## Development

Install with dev dependencies (includes ruff for formatting and linting):

```bash
pip install -e ".[dev]"
```

Format code:

```bash
ruff format .
```

Lint (optional):

```bash
ruff check .
```
