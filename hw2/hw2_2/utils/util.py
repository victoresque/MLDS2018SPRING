import os


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
