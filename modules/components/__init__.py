import importlib

def build_components(args):
    module = importlib.import_module('modules.components.{}'.format(args.model_module))
    return module.build_components(args)