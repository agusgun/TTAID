import os
import numpy as np

from PIL import Image
from glob import glob

def compare_dirs(dir1: str, dir2: str) -> bool:
    # Compare the content of directory whether two directories contain the same images or not
    exts = ['*.jpg', '*.png', '*.jpeg']
    imgs_path_1, imgs_path_2 = [], []
    for ext in exts:
        imgs_path_1.extend(glob(os.path.join(dir1, ext)))
        imgs_path_2.extend(glob(os.path.join(dir2, ext)))
    
    imgs_path_1 = sorted(imgs_path_1)
    imgs_path_2 = sorted(imgs_path_2)    
    if imgs_path_1 != imgs_path_2:
        return False
    else:
        # check pixels
        error = 0
        for path_1, path_2 in zip(imgs_path_1, imgs_path_2):
            img_1 = np.array(Image.open(path_1)).astype(np.float32)
            img_2 = np.array(Image.open(path_2)).astype(np.float32)

            error += np.sum(img_1 - img_2)

        if error > 0:
            return False
        else:
            return True

if __name__ == "__main__":
    print(compare_dirs('noisy1', 'noisy2'))
    print(compare_dirs('noisy2', 'noisy3'))
