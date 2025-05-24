# deepsensor_greatlakes/model.py

import os
import json
import torch
import numpy as np
import re
import pprint # For debug_info printing

from deepsensor.model import ConvNP
from deepsensor.model.nps import construct_neural_process # Although not directly called in load_convnp_model,
                                                        # ConvNP's init relies on it, so useful to have
from deepsensor.data import DataProcessor, TaskLoader
import torch.nn as nn # For deserialization of activation functions

# --- Custom save function ---
def save_model(model, model_ID: str):
    os.makedirs(model_ID, exist_ok=True)
    torch.save(model.model.state_dict(), os.path.join(model_ID, "model.pt"))
    config_fpath = os.path.join(model_ID, "model_config.json")
    with open(config_fpath, "w") as f:
        json.dump(model.config, f, indent=4, sort_keys=False, default=str)

# --- Helper functions for deserialization ---
def _convert_string_to_numeric_if_possible(value):
    if isinstance(value, str):
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value):
            try:
                return float(value)
            except ValueError:
                pass
    return value

def _deserialize_config(config_data):
    if isinstance(config_data, dict):
        deserialized = {}
        for key, value in config_data.items():
            deserialized[key] = _deserialize_config(value)
        return deserialized
    elif isinstance(config_data, list):
        return [_deserialize_config(item) for item in config_data]
    else:
        converted_val = _convert_string_to_numeric_if_possible(config_data)
        if isinstance(converted_val, str):
            # import torch.nn as nn # Moved to top of file
            if converted_val == "<class 'torch.nn.modules.activation.ReLU'>":
                return nn.ReLU
            elif converted_val == "<class 'torch.nn.modules.activation.LeakyReLU'>":
                return nn.LeakyReLU
            else:
                return converted_val
        else:
            return converted_val

# --- Main custom load function ---
def load_convnp_model(model_ID: str, data_processor: DataProcessor, task_loader: TaskLoader):
    config_fpath = os.path.join(model_ID, "model_config.json")
    with open(config_fpath, "r") as f:
        config_raw = json.load(f)

    deserialized_config = _deserialize_config(config_raw)

    # Prepare config for constructing the underlying neural process.
    config_for_nps_constructor = deserialized_config.copy()

    if 'family' in config_for_nps_constructor:
        del config_for_nps_constructor['family']
    if 'neural_process_type' in config_for_nps_constructor:
        del config_for_nps_constructor['neural_process_type']
    if 'data_processor' in config_for_nps_constructor:
        del config_for_nps_constructor['data_processor']
    if 'task_loader' in config_for_nps_constructor:
        del config_for_nps_constructor['task_loader']


    print("Attempting to instantiate ConvNP model (randomly initialized initially):")
    print("Architectural config for construct_neural_process (passed as **kwargs):", config_for_nps_constructor)

    try:
        loaded_convnp_model = ConvNP(
            data_processor,
            task_loader,
            **config_for_nps_constructor
        )
    except Exception as e:
        print(f"Error when instantiating ConvNP: {e}")
        debug_info = {
            'data_processor_arg': data_processor,
            'task_loader_arg': task_loader,
            'architectural_kwargs': config_for_nps_constructor
        }
        pprint.pprint(debug_info)
        raise

    model_weights_fpath = os.path.join(model_ID, "model.pt")
    loaded_convnp_model.model.load_state_dict(
        torch.load(model_weights_fpath, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True)
    )
    loaded_convnp_model.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    loaded_convnp_model.config = deserialized_config

    return loaded_convnp_model