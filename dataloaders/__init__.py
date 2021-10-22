import importlib

def build_data_loader(args):
    cls = getattr(importlib.import_module("dataloaders.{}".format(args.data_loader_module)), '{}'.format(args.data_loader_name))
    dataset = cls(args)
    return dataset.get_loader()