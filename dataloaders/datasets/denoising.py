import numpy as np
import os
import random
import scipy.io
import torchvision.transforms as v_transforms
import torchvision.transforms.functional as F_t


from utils.transforms.transforms import get_transform_params, get_transform
from utils.transforms.task_specific.denoising import synthesize_noise
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from pathlib import Path


class DIV2KSyntheticNoise(Dataset):
    def __init__(self, args, is_train=True, with_transform=True):
        self.img_dir = args.data_dir
        self.data_exts = args.data_exts
        self.crop_size = args.crop_size
        self.normalization = args.normalization

        self.is_train = is_train
        self.with_transform = with_transform

        if self.is_train:
            self.img_dir = os.path.join(self.img_dir, "train")
        else:
            self.img_dir = os.path.join(self.img_dir, "valid")

        self.imgs_path = []
        for ext in args.data_exts:
            self.imgs_path.extend(glob(os.path.join(self.img_dir, ext)))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]

        clean = Image.open(img_path).convert("RGB")
        clean = F_t.to_tensor(clean)

        noisy = clean.clone()
        noisy = synthesize_noise(noisy)

        if self.with_transform:
            transform_params = get_transform_params(noisy.size(), self.crop_size)
            noisy_transform = get_transform(
                transform_params,
                is_train=self.is_train,
                normalization=self.normalization,
            )
            clean_transform = get_transform(
                transform_params,
                is_train=self.is_train,
                normalization=self.normalization,
            )

            noisy = noisy_transform(noisy)
            clean = clean_transform(clean)

        return {"noisy": noisy, "clean": clean, "img_path": img_path}


class NAMRealNoise(Dataset):
    def __init__(self, args, is_train=False, with_transform=False):
        self.img_dir = args.data_dir
        self.data_exts = args.data_exts
        self.crop_size = args.crop_size
        self.normalization = args.normalization

        self.is_train = is_train
        assert not (self.is_train), "NAMRealNoise only support test dataset"

        self.with_transform = with_transform

        self.imgs_path = []
        for ext in args.data_exts:
            for path in Path(self.img_dir).rglob(ext):
                self.imgs_path.append(path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]

        mat = scipy.io.loadmat(img_path)

        noisy = mat["img_noisy"]
        clean = mat["img_mean"]

        noisy = F_t.to_tensor(noisy)
        clean = F_t.to_tensor(clean)

        if self.with_transform:
            transform = v_transforms.CenterCrop(self.crop_size)

            noisy = transform(noisy)
            clean = transform(clean)

        return {"noisy": noisy, "clean": clean, "img_path": str(img_path)}


class PolyRealNoise(Dataset):
    def __init__(self, args, is_train=True):
        self.img_dir = args.data_dir_poly
        self.data_exts = args.data_exts
        self.crop_size = args.crop_size
        self.normalization = args.normalization

        self.is_train = is_train

        # if self.is_train:
        #    self.img_dir = os.path.join(self.img_dir, "train")
        # else:
        #    self.img_dir = os.path.join(self.img_dir, "valid")

        self.imgs_path = []
        for ext in args.data_exts:
            self.imgs_path.extend(glob(os.path.join(self.img_dir, ext)))

        self.imgs_path = list(
            set(["_".join(i.split("_")[:-1]) for i in self.imgs_path])
        )

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]

        img_clean_path = img_path + "_mean.JPG"
        img_noisy_path = img_path + "_real.JPG"

        clean = Image.open(img_clean_path).convert("RGB")
        clean = F_t.to_tensor(clean)

        noisy = Image.open(img_noisy_path).convert("RGB")
        noisy = F_t.to_tensor(noisy)

        label_dataset = "poly"

        return {
            "noisy": noisy,
            "clean": clean,
            "img_path": img_path,
            "ds_label": label_dataset,
        }


class SIDDRealNoise(Dataset):
    def __init__(self, args, is_train=True, with_transform=True):
        self.img_dir = args.data_dir_sidd
        self.data_exts = args.data_exts
        self.crop_size = args.crop_size
        self.normalization = args.normalization

        self.is_train = is_train
        self.with_transform = with_transform

        self.imgs_path = list(open(self.img_dir + "/Scene_Instances.txt", "r"))
        self.imgs_path = [
            os.path.join(self.img_dir, "Data", i.strip("\n")) for i in self.imgs_path
        ]

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        img_path_clean = glob(os.path.join(img_path, "GT*"))[0]
        img_path_noisy = glob(os.path.join(img_path, "NOISY*"))[0]

        clean = Image.open(img_path_clean).convert("RGB")
        clean = F_t.to_tensor(clean)

        noisy = Image.open(img_path_noisy).convert("RGB")
        noisy = F_t.to_tensor(noisy)

        return {"noisy": noisy, "clean": clean, "img_path": img_path}


if __name__ == "__main__":
    from utils.config import parse_json
    import argparse

    arg_parser = argparse.ArgumentParser(description="Pytorch Project")
    args = arg_parser.parse_args()
    json_args = parse_json("configs/denoising_baseline_exp0.json")
    args = arg_parser.parse_args(namespace=json_args)

    dataset = DIV2KSyntheticNoise(args, is_train=True, with_transform=False)
    transform_to_img = v_transforms.Compose([v_transforms.ToPILImage()])

    print(len(dataset))
    chosen_img = dataset[0]
    print(chosen_img["img_path"])
    transform_to_img(chosen_img["noisy"]).show()
    transform_to_img(chosen_img["clean"]).show()
