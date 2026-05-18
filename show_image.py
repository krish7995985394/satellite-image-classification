from PIL import Image
import numpy as np
import sys, os

# choose an image that exists in the repo
img_path = os.path.join("images", "crops.jpg")

try:
    img = Image.open(img_path)
except Exception as e:
    print("ERROR: failed to open", img_path, "->", e)
    sys.exit(1)

arr = np.array(img)
print("path:", img_path)
print("format:", img.format)
print("mode:", img.mode)
print("size (w,h):", img.size)
print("numpy dtype:", arr.dtype)
print("numpy shape:", arr.shape)
