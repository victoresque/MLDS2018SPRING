import os
from copy import copy
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

