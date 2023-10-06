import yaml
import os
import pathlib
import warnings

config_file_name = "bw_config.yaml"
fallback_path = pathlib.Path(__file__).resolve().parent


def get_raw_config():
    pwd = pathlib.Path(os.curdir).resolve()
    for path in [pwd] + list(pwd.parents) + [fallback_path]:
        file_candidate = path / config_file_name
        if os.path.exists(file_candidate) and os.path.isfile(file_candidate):
            with open(file_candidate, 'r') as fhan:
                return yaml.safe_load(fhan), path
    raise Exception("No config file found, even at the fallback location.")


def make_paths_absolute(config, path):
    # todo: this is hacky, but it works for now
    if path == fallback_path and os.getlogin() != 'jgould':
        warnings.warn("Loading the fallback config file.")
        path = pathlib.Path(os.curdir).resolve()

    new_config = {}
    for key, value in config.items():
        if type(value) == str:
            p = pathlib.Path(value)
            if not p.is_absolute():
                p = path/p

            new_config[key] = p
        else:
            new_config[key] = value
    return new_config



CONFIG = make_paths_absolute(*get_raw_config())