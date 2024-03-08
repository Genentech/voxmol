import torch

from voxmol.dataset.dataset import DatasetVoxMol


def create_loader(config: dict):
    """
    Create data loaders for training and validation sets.

    Args:
        config (dict): Configuration parameters for the data loaders.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    dset_train = DatasetVoxMol(
        dset_name=config["dset_name"],
        data_dir=config["data_dir"],
        elements=config["elements"],
        split="train",
        small=config["debug"],
    )
    loader_train = torch.utils.data.DataLoader(
        dset_train,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    dset_val = DatasetVoxMol(
        dset_name=config["dset_name"],
        data_dir=config["data_dir"],
        elements=config["elements"],
        split="val",
        small=config["debug"],
    )
    loader_val = torch.utils.data.DataLoader(
        dset_val,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    print(f">> training/val set size: {len(dset_train)}/{len(dset_val)}")
    return loader_train, loader_val
