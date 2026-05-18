import sys

print("python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("torch: ERROR -", e)

try:
    import rasterio
    print("rasterio:", rasterio.__version__)
except Exception as e:
    print("rasterio: ERROR -", e)
