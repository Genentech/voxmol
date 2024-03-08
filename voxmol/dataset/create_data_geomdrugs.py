import argparse
import gc
import os
import pickle
from pyuul import utils
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm import tqdm
import urllib.request

from voxmol.utils import makedir
from voxmol.constants import ELEMENTS_HASH

RDLogger.DisableLog("rdApp.*")

RAW_URL_TRAIN = "https://drive.switch.ch/index.php/s/UauSNgSMUPQdZ9v/download"
RAW_URL_VAL = "https://drive.switch.ch/index.php/s/YNW5UriYEeVCDnL/download"
RAW_URL_TEST = "https://drive.switch.ch/index.php/s/GQW9ok7mPInPcIo/download"


def download_data(raw_data_dir: str):
    """
    Download the raw data files from the specified URLs and save them in the given directory.

    Args:
        raw_data_dir (str): The directory where the raw data files will be saved.
    """
    urllib.request.urlretrieve(RAW_URL_TRAIN, os.path.join(raw_data_dir, "train_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_VAL, os.path.join(raw_data_dir, "val_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_TEST, os.path.join(raw_data_dir, "test_data.pickle"))


def preprocess_geom_drugs_dataset(raw_data_dir: str, data_dir: str, split: str = "train"):
    """
    Preprocesses the Geom Drugs dataset.

    Args:
        raw_data_dir (str): The directory path where the raw data is stored.
        data_dir (str): The directory path where the preprocessed data will be saved.
        split (str, optional): The split of the dataset to preprocess. Defaults to "train".

    Returns:
        list: The preprocessed dataset.

    """
    print("  >> load data raw from ", os.path.join(raw_data_dir, f"{split}_data.pickle"))
    with open(os.path.join(raw_data_dir, f"{split}_data.pickle"), 'rb') as f:
        all_data = pickle.load(f)

    # get all conformations of all molecules
    mols_confs = []
    for i, data in enumerate(all_data):
        _, all_conformers = data
        for j, conformer in enumerate(all_conformers):
            if j >= 5:
                break
            mols_confs.append(conformer)

    # write sdf / load with PyUUL
    print("  >> write .sdf of all conformations and extract coords/types with PyUUL")
    sdf_path = os.path.join(data_dir, f"{split}.sdf")
    with Chem.SDWriter(sdf_path) as w:
        for m in mols_confs:
            w.write(m)
    coords, atname = utils.parseSDF(sdf_path)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)

    # create the dataset
    print("  >> create the dataset for this split")
    data = []
    num_errors = 0
    for i, mol in enumerate(tqdm(mols_confs)):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if smiles is None:
            num_errors += 1
        datum = {
            "mol": mol,
            "smiles": smiles,
            "coords": coords[i].clone(),
            "atoms_channel": atoms_channel[i].int(),
        }

        data.append(datum)

    print(f"  >> split size: {len(data)} ({num_errors} errors)")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="data/drugs/raw/")
    parser.add_argument("--data_dir", type=str, default="data/drugs/")
    args = parser.parse_args()

    if not os.path.isdir(args.raw_data_dir):
        makedir(args.raw_data_dir)
        download_data(args.raw_data_dir)

    makedir(args.data_dir)

    data = {}
    for split in ["train", "val", "test"]:
        print(f">> preprocessing {split}...")
        dset = preprocess_geom_drugs_dataset(args.raw_data_dir, args.data_dir, split)
        torch.save(dset, os.path.join(args.data_dir, f"{split}_data.pth"),)

        del dset
        gc.collect()
