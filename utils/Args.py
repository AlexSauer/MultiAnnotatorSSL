import torch
import yaml
import os
import warnings
from typing import Optional


class Args:
    def __init__(self, base_yaml_file: str, extension_yaml_file: Optional[str] = None):
        # Load the yaml file and update the corresponding attributes
        if os.path.isfile(base_yaml_file):
            with open(base_yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        # The file can also be given as a direct string and then is read in directly (useful for debugging)
        else:
            config_dict = yaml.safe_load(base_yaml_file)
        self.__dict__.update(**config_dict)

        # If an extension yaml file is given, update the parameters according to this extension file:
        # This is useful if I run several experiments with one base configuration and want to see the
        # effects of isolated changes in the extension_yaml_file
        if extension_yaml_file is not None:
            if os.path.isfile(extension_yaml_file):
                with open(extension_yaml_file, 'r') as f:
                    extension_dict = yaml.safe_load(f)
            else:
                extension_dict = yaml.safe_load(extension_yaml_file)

            for key, value in extension_dict.items():
                if key in vars(self):
                    setattr(self, key, value)
                else:
                    warnings.warn(f'Key {key} is in the extension yaml file but not in base configuration!')

        # Transform some data
        self.device = torch.device("cuda:" + str(self.DEVICE_ID))
        self.transforms = self.transforms if self.transforms is not None else []
        for var in ['lr', 'l2_reg', 'debug_n_samples', 'sched_step']:
            self.convert_scientific_notation_to_float(var)
        for var in ['debug_n_samples', 'input_channels']:
            self.convert_to_int(var)

        # Set some defaults
        if 'log_stdout' not in vars(self):
            self.log_stdout = True

        # Check path
        for attr in vars(self):
            if attr.startswith('path'):
                assert getattr(self, attr).endswith('/'), \
                       f"Path has to end with / character! But is {getattr(self, attr)}"

    def to_string(self) -> str:
        doc = "Configurations: \n"
        doc +=' \n'.join(k + ': ' + str(v) for k, v in self.__dict__.items())
        return doc

    def convert_scientific_notation_to_float(self, name: str) -> None:
        try:
            setattr(self, name, float(getattr(self, name)))
        except AttributeError as e:
            print(f'While converting args to float: ', e, sep='')

    def convert_to_int(self, name: str) -> None:
        try:
            setattr(self, name, int(getattr(self, name)))
        except AttributeError:
            pass

    # def __getattr__(self, item):
    #     '''If an attribute is not found, then return None'''
    #     return None