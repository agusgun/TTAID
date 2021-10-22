"""
Config related stuffs
"""

import json
import argparse

def parse_json(file_path):
    """
    in:
        file_path: json file path
    out:
        return argparse name parser
    objective:
        parse json file
    """
    with open(file_path, 'r') as file:
        config_dict = json.load(file)
        args = argparse.Namespace()
        args.__dict__.update(config_dict)

        return args