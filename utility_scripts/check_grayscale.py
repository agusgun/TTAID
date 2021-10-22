from PIL import Image
from glob import glob

import os

def is_grayscale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

imgs_path = glob(os.path.join('./', '*.tif'))

num_of_color = 0
for img_path in imgs_path:
    if not is_grayscale(img_path):
        print(img_path)
        num_of_color += 1
print(num_of_color)