import torch
import random
import os
import numpy as np

from torch.utils.data import DataLoader

from .datasets.denoising import *


class DIV2KSynthesisDegradationLoader:
    def __init__(self, args):
        self.args = args

        if args.data_mode == "imgs":
            if args.mode == "training":
                train_dataset = DIV2KSyntheticNoise(
                    self.args, is_train=True, with_transform=True
                )
                val_dataset = DIV2KSyntheticNoise(
                    self.args, is_train=False, with_transform=True
                )

                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.train_batch_size,
                    shuffle=True,
                    num_workers=args.data_loader_workers,
                )
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            elif args.mode == "validation":
                self.train_loader = None

                val_dataset = DIV2KSyntheticNoise(
                    self.args, is_train=False, with_transform=False
                )
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            else:
                self.train_loader = None

                val_dataset = NAMRealNoise(args, is_train=False, with_transform=False)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
        elif args.data_mode == "numpy":
            raise NotImplementedError("This is not yet implemented")
        elif args.data_mode == "h5":
            raise NotImplementedError("This is not yet implemented")
        else:
            raise Exception("Please specify data mode in json config")

    def get_loader(self):
        return self.train_loader, self.val_loader


class PolyDatasetLoader:
    def __init__(self, args):
        self.args = args

        if args.data_mode == "imgs":
            if args.mode == "training":
                train_dataset = PolyRealNoise(self.args, is_train=True)
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.train_batch_size,
                    shuffle=True,
                    num_workers=args.data_loader_workers,
                )
                self.val_loader = DataLoader(
                    train_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            elif args.mode == "validation":
                self.train_loader = None

                val_dataset = PolyRealNoise(self.args, is_train=True)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            else:
                self.train_loader = None

                val_dataset = PolyRealNoise(self.args, is_train=True)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
        elif args.data_mode == "numpy":
            raise NotImplementedError("This is not yet implemented")
        elif args.data_mode == "h5":
            raise NotImplementedError("This is not yet implemented")
        else:
            raise Exception("Please specify data mode in json config")

    def get_loader(self):
        return self.train_loader, self.val_loader


class SIDDDatasetLoader:
    def __init__(self, args):
        self.args = args

        if args.data_mode == "imgs":
            if args.mode == "training":
                train_dataset = SIDDRealNoise(self.args, is_train=True)
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.train_batch_size,
                    shuffle=True,
                    num_workers=args.data_loader_workers,
                )
                self.val_loader = DataLoader(
                    train_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            elif args.mode == "validation":
                self.train_loader = None

                val_dataset = SIDDRealNoise(self.args, is_train=True)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            else:
                self.train_loader = None

                val_dataset = SIDDRealNoise(self.args, is_train=True)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
        elif args.data_mode == "numpy":
            raise NotImplementedError("This is not yet implemented")
        elif args.data_mode == "h5":
            raise NotImplementedError("This is not yet implemented")
        else:
            raise Exception("Please specify data mode in json config")

    def get_loader(self):
        return self.train_loader, self.val_loader


class RealNoiseDataLoader:
    def __init__(self, args):
        self.args = args

        if args.data_mode == "imgs":
            if args.mode == "training":
                train_dataset = RealNoiseTrainingDataset(self.args, is_train=True, with_transform=True)
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.train_batch_size,
                    shuffle=True,
                    num_workers=args.data_loader_workers,
                )

                val_dataset = RealNoiseTrainingDataset(self.args, is_train=False, with_transform=True)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            elif args.mode == "validation":
                self.train_loader = None

                if args.meta_transfer:
                    with_transform = True
                else:
                    with_transform = False
                val_dataset = RealNoiseTrainingDataset(self.args, is_train=False, with_transform=with_transform)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
            else:
                self.train_loader = None

                val_dataset = NAMRealNoise(args, is_train=False, with_transform=True)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.data_loader_workers,
                )
        elif args.data_mode == "numpy":
            raise NotImplementedError("This is not yet implemented")
        elif args.data_mode == "h5":
            raise NotImplementedError("This is not yet implemented")
        else:
            raise Exception("Please specify data mode in json config")

    def get_loader(self):
        return self.train_loader, self.val_loader


