import importlib

def build_model(args, device):
    cls = getattr(importlib.import_module('modules.models.{}'.format(args.model_module)), '{}'.format(args.model_name))
    model = cls(args, device)
    return model
