import logging
import numpy as np
from sklearn.model_selection import train_test_split
import sys

LOG = logging.getLogger("modnet")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)


def get_hash_of_file(fname, algo="sha512"):
    """Returns the hexdigest of the SHA512 checksum of the
    file found at fname.

    """
    import hashlib

    block_size = 65536
    if algo.lower() == "md5":
        _hash = hashlib.md5()
    else:
        _hash = hashlib.sha512()
    with open(fname, "rb") as f:
        fb = f.read(block_size)
        while fb:
            _hash.update(fb)
            fb = f.read(block_size)

    return _hash.hexdigest()


def generate_shuffled_and_stratified_val_split(
    y: np.ndarray, val_fraction: float, classification: bool
):
    """
    Generate train validation split that is shuffled, reproducible and, if classification, stratified.
    Please note for classification tasks that stratification is performed on first target.
    y: np.ndarray (n_samples, n_targets) or (n_samples, n_targets, n_classes)
    """
    if classification:
        if isinstance(y[0][0], list) or isinstance(y[0][0], np.ndarray):
            ycv = np.argmax(y[:, 0], axis=1)
        else:
            ycv = y[:, 0]
        return train_test_split(
            range(len(y)),
            test_size=val_fraction,
            random_state=42,
            shuffle=True,
            stratify=ycv,
        )
    else:
        return train_test_split(
            range(len(y)), test_size=val_fraction, random_state=42, shuffle=True
        )
