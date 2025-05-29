"""Utility functions for loading, saving, metrics."""
import json
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def compute_metrics(true, pred, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    return dict(zip(labels, zip(precision, recall, f1)))

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
