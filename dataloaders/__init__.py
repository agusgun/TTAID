import importlib


def build_data_loader(args):
    if args.meta_transfer == 1:
        datasets = []
        for i in range(len(args.data_loader_name)):
            cls = getattr(
                importlib.import_module(
                    "dataloaders.{}".format(args.data_loader_module)
                ),
                "{}".format(args.data_loader_name[i]),
            )
            dataset = cls(args)
            datasets.append(dataset.get_loader())
        return datasets
    else:
        cls = getattr(
            importlib.import_module("dataloaders.{}".format(args.data_loader_module)),
            "{}".format(args.data_loader_name),
        )
        dataset = cls(args)
        return dataset.get_loader()