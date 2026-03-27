import os
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

from SequenceDatasets import dataset

signal_size = 1024
stride = 1024

CLASS_NAMES = [
    "KA01", "KA03", "KA04", "KA05", "KA07", "KA08",
    "KI01", "KI03", "KI04", "KI07", "KI16", "KI18"
]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DEFAULT_SRC_DOMAIN = "condition_0"
DEFAULT_TAR_DOMAIN = "condition_1"


def _resolve_domain_name(domain_name: str) -> str:
    if domain_name == "Source domain":
        return DEFAULT_SRC_DOMAIN
    if domain_name == "Target domain":
        return DEFAULT_TAR_DOMAIN
    return domain_name


def load_training(root_path, dir, batch_size, kwargs):
    domain_name = _resolve_domain_name(dir)
    list_data = get_files_pu(root_path, domain_name)
    data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
    source_train = dataset(list_data=data_pd)
    train_loader = torch.utils.data.DataLoader(
        source_train,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    domain_name = _resolve_domain_name(dir)
    list_data = get_files_pu(root_path, domain_name)
    data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
    test_train = dataset(list_data=data_pd)
    test_loader = torch.utils.data.DataLoader(
        test_train,
        batch_size=int(batch_size),
        shuffle=False,
        **kwargs,
    )
    return test_loader


def get_files_pu(root, domain_name):
    domain_path = os.path.join(root, domain_name)
    if not os.path.isdir(domain_path):
        raise FileNotFoundError(
            f"Domain folder not found: {domain_path}\n"
            f"Please check --root_path and --src/--tar settings."
        )

    data = []
    lab = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(domain_path, class_name)
        if not os.path.isdir(class_dir):
            warnings.warn(f"Class folder not found, skipped: {class_dir}")
            continue

        mat_files = sorted(
            [f for f in os.listdir(class_dir) if f.lower().endswith('.mat')]
        )
        if len(mat_files) == 0:
            warnings.warn(f"No .mat files found in: {class_dir}")
            continue

        label = CLASS_TO_LABEL[class_name]
        for mat_name in mat_files:
            mat_path = os.path.join(class_dir, mat_name)
            data_i, lab_i = data_load_pu(mat_path, label=label)
            data += data_i
            lab += lab_i

    if len(data) == 0:
        raise RuntimeError(
            f"No valid samples were loaded from {domain_path}.\n"
            f"Please check folder structure, .mat content, and variable names."
        )

    return [data, lab]


def data_load_pu(filename, label):
    signal = _read_pu_signal(filename)
    signal = _normalize_to_minus1_1(signal)

    data = []
    lab = []
    start, end = 0, signal_size

    while end <= len(signal):
        window = signal[start:end].reshape(-1, 1).astype(np.float32)
        data.append(window)
        lab.append(label)
        start += stride
        end += stride

    return data, lab


def _collect_numeric_candidates(obj, out_list):
    """Recursively collect numeric 1D candidates from MATLAB-loaded objects."""
    if obj is None:
        return

    if isinstance(obj, mat_struct):
        for field in getattr(obj, '_fieldnames', []):
            _collect_numeric_candidates(getattr(obj, field), out_list)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith('__'):
                continue
            _collect_numeric_candidates(v, out_list)
        return

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for item in obj.flat:
                _collect_numeric_candidates(item, out_list)
            return
        arr = np.asarray(obj).squeeze()
        if np.issubdtype(arr.dtype, np.number) and arr.size >= signal_size:
            out_list.append(arr.reshape(-1))
        return

    if np.isscalar(obj):
        return

    # Generic Python objects with attributes (fallback)
    if hasattr(obj, '__dict__'):
        for v in vars(obj).values():
            _collect_numeric_candidates(v, out_list)



def _read_pu_signal(filename):
    mat = loadmat(filename, squeeze_me=True, struct_as_record=False)

    # 1) Common keys first
    preferred_keys = [
        'data', 'signal', 'DE_time', 'FE_time', 'BA_time',
        'X', 'Y', 'Channel_1', 'channel_1'
    ]
    for key in preferred_keys:
        if key in mat:
            candidates = []
            _collect_numeric_candidates(mat[key], candidates)
            if candidates:
                candidates.sort(key=lambda x: x.size, reverse=True)
                return candidates[0]

    # 2) Search every non-metadata key recursively
    all_candidates = []
    for key, value in mat.items():
        if key.startswith('__'):
            continue
        _collect_numeric_candidates(value, all_candidates)

    if not all_candidates:
        raise KeyError(
            f"No usable numeric signal was found in file: {filename}\n"
            f"Available keys: {list(mat.keys())}"
        )

    all_candidates.sort(key=lambda x: x.size, reverse=True)
    return all_candidates[0]



def _normalize_to_minus1_1(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0
