import logging
logging.captureWarnings(True)

import deepsensor.torch
from deepsensor.train import set_gpu_default_device

# Run on GPU if available by setting GPU as default device
set_gpu_default_device()

# If nothing happens, this is a good sign