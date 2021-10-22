import random
import torchvision.transforms as v_transforms
import torchvision.transforms.functional as F_t
import torch

def get_transform_params(img_size, crop_size):
    _, h, w = img_size # in Tensor Format
    x = random.randint(0, max(0, w - crop_size))
    y = random.randint(0, max(0, h - crop_size))

    is_horizontal_flip = random.uniform(0, 1) > 0.5
    params = {'crop_position': (x, y), 'crop_size': crop_size, 'is_horizontal_flip': is_horizontal_flip}
    return params

def get_transform(transform_params, is_train=True, normalization=False):
    """
    
    """
    transform_list = []

    if is_train:
        transform_list.append(
            v_transforms.Lambda(
                lambda img: crop(img, transform_params['crop_position'], transform_params['crop_size'])
            )
        )
        transform_list.append(
            v_transforms.Lambda(
                lambda img: horizontal_flip(img, transform_params['is_horizontal_flip'])
            )
        )
    else:
        transform_list.append(v_transforms.CenterCrop(transform_params['crop_size']))
    # transform_list.append(v_transforms.ToTensor())
    if normalization:
        transform_list.append(v_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return v_transforms.Compose(transform_list)

def crop(img, pos, crop_size):
    """
    img: torch.Tensor
    Note: can crop outside boundary, but it can be handled by choosing top-left point that is safe to crop
    """
    _, h, w = img.size() # pil format w, h
    x, y = pos
    tw = th = crop_size
    if (w > tw or h > th):
        return img[:, y:y+th, x:x+tw]
    return img

def horizontal_flip(img, is_flip):
    if is_flip:
        return F_t.hflip(img)
    return img

def augment(img: torch.Tensor, augment_mode):
    if augment_mode == 0:
        # original
        img = img 
    elif augment_mode == 1:
        # flip up and down
        img = F_t.vflip(img)
    elif augment_mode == 2:
        # rotate counterwise 90 degree
        img = torch.rot90(img, k=1, dims=(1,2))
    elif augment_mode == 3:
        # rotate 90 degree and flip up and down
        img = torch.rot90(img, k=1, dims=(1,2))
        img = F_t.vflip(img)
    elif augment_mode == 4:
        # rotate 180 degree
        img = torch.rot90(img, k=2, dims=(1,2))
    elif augment_mode == 5:
        # rotate 180 degree and flip
        img = torch.rot90(img, k=2, dims=(1,2))
        img = F_t.vflip(img)
    elif augment_mode == 6:
        # rotate 270 degree
        img = torch.rot90(img, k=3, dims=(1,2))
    elif augment_mode == 7:
        # rotate 270 degree and flip
        img = torch.rot90(img, k=3, dims=(1,2))
        img = F_t.vflip(img)
    return img


def main():
    from PIL import Image
    from torchvision.utils import save_image

    img_path = 'data/DIV2K/train/0003.png'
    img = Image.open(img_path)
    img = F_t.to_tensor(img)

    img = augment(img, 7)
    save_image(img, './augment.png')

if __name__ == "__main__":
    main()