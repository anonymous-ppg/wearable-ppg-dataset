# encoding=utf-8
"""
data_prep.py
------------
Dispatcher that routes args.dataset to the correct data loader.
"""

import os
from data_preprocess.data_preprocess_ppg import prep_ppg
from data_preprocess.data_preprocess_multisite import prep_multisite
from data_preprocess.data_preprocess_multisite_accel import prep_domains_subject_sp as prep_multisite_accel
from data_preprocess.data_preprocess_multisite_ir import prep_domains_subject_sp as prep_multisite_ir


# create output directories once at import time
os.makedirs('results', exist_ok=True)


def setup_dataloaders(args):
    """
    Returns (train_loaders, val_loader, test_loader) for the requested dataset.

    train_loaders is a list with one DataLoader (kept as a list for compatibility
    with the training loop which iterates over train_loaders[0]).
    """
    if args.dataset == 'ppg':
        return prep_ppg(args)
    elif args.dataset == 'multisite':
        return prep_multisite(args)
    elif args.dataset == 'multisite_accel':
        return prep_multisite_accel(args)
    elif args.dataset == 'multisite_ir':
        return prep_multisite_ir(args)
    else:
        raise ValueError(
            f"Unknown dataset: '{args.dataset}'. "
            f"Currently supported: 'ppg', 'multisite', "
            f"'multisite_accel', 'multisite_ir'."
        )
