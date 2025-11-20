"""Dataset registry mapping dataset names to filenames.

This module provides a simple registry of available benchmark datasets.
Each dataset is stored as a .npz file with features (X) and labels (y).
"""

# Simple mapping of dataset names to their .npz filenames
DATASET_REGISTRY: dict[str, str] = {
    "annthyroid": "annthyroid.npz",
    "backdoor": "backdoor.npz",
    "breast": "breast_w.npz",
    "cardio": "cardio.npz",
    "cover": "cover.npz",
    "donors": "donors.npz",
    "fraud": "fraud.npz",
    "glass": "glass.npz",
    "hepatitis": "hepatitis.npz",
    "http": "http.npz",
    "ionosphere": "ionosphere.npz",
    "letter": "letter.npz",
    "lymphography": "lymphography.npz",
    "magic_gamma": "magic_gamma.npz",
    "mammography": "mammography.npz",
    "mnist": "mnist.npz",
    "musk": "musk.npz",
    "optdigits": "optdigits.npz",
    "pageblocks": "pageBlocks.npz",
    "pendigits": "pendigits.npz",
    "satimage2": "satimage2.npz",
    "shuttle": "shuttle.npz",
    "smtp": "smtp.npz",
    "stamps": "stamps.npz",
    "thyroid": "thyroid.npz",
    "vowels": "vowels.npz",
    "wbc": "wbc.npz",
    "wine": "wine.npz",
    "yeast": "yeast.npz",
}
