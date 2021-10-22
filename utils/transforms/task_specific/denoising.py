import numpy as np
import random
import torch
import torchvision.transforms.functional as F_t

from PIL import Image
from torchvision.utils import save_image


def synthesize_gaussian(img: torch.Tensor, std_l: float, std_r: float) -> torch.Tensor:
    """
    img: torch.Tensor range [0, 1]
    std_l, std_r: float in range [0, 255]
    """
    std = random.uniform(std_l / 255., std_r / 255.)
    noise = torch.randn_like(img) * std
    img = img + noise
    img = torch.clamp(img, 0., 1.)
    return img


def synthesize_speckle(img: torch.Tensor, std_l: float, std_r: float) -> torch.Tensor:
    """
    img: torch.Tensor range [0, 1]
    std_l, std_r: float in range [0, 255]
    """
    std = random.uniform(std_l / 255., std_r / 255.)
    noise = torch.randn_like(img) * std
    img = img + img * noise
    img = torch.clamp(img, 0., 1.)
    return img

def synthesize_salt_pepper(img: torch.Tensor, amount: float, salt_vs_pepper: float):
    p = amount
    q = salt_vs_pepper

    out = img.clone()
    
    # Salt
    num_salt = np.ceil(p * img.nelement() * q)
    coords = [torch.randint(0, i - 1, size=(int(num_salt),)) for i in img.size()]
    out[coords] = 1.

    # Pepper
    num_pepper = np.ceil(p * img.nelement() * (1. - q))
    coords = [torch.randint(0, i - 1, size=(int(num_pepper),)) for i in img.size()]
    out[coords] = 0.
    return out

def synthesize_noise(img: torch.Tensor):
    tasks = list(range(3))
    random.shuffle(tasks)

    for task in tasks:
        if task == 0:
            img = synthesize_gaussian(img, 5, 50)
        elif task == 1:
            img = synthesize_speckle(img, 5, 50)
        elif task == 2:
            img = synthesize_salt_pepper(img, random.uniform(0, 0.01), random.uniform(0.3, 0.8))
    
    return img

def main():
    img_path = 'data/DIV2K/train/0002.png'
    img = Image.open(img_path)
    img = F_t.to_tensor(img)
    img = synthesize_noise(img)
    save_image(img, './out_noisy.png')


if __name__ == "__main__":
    main()