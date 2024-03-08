import os
import math
import random
import torch
from torch.utils.data import Dataset

from voxmol.constants import ELEMENTS_HASH


class DatasetVoxMol(Dataset):
    """
    A custom dataset class for VoxMol dataset.

    Args:
        dset_name (str): The name of the dataset. Can be "qm9" or "drugs".
        data_dir (str): The directory where the dataset is stored.
        elements (list): The list of elements to include in the dataset.
        split (str): The split of the dataset. Can be "train", "val", or "test".
        small (bool): Whether to use a small subset of the dataset.
        atomic_radius (float): The atomic radius to assign to each atom.
        max_n_atoms (int): The maximum number of atoms allowed in a sample.
    """

    def __init__(
        self,
        dset_name: str = "qm9",
        data_dir: str = "dataset/data/",
        elements: list = None,
        split: str = "train",
        small: bool = False,
        atomic_radius: float = .5,
        max_n_atoms: int = 80,
    ):
        if elements is None:
            elements = ELEMENTS_HASH
        assert dset_name in ["qm9", "drugs"], "dset_name must be qm9 or drugs"
        assert split in ["train", "val", "test"], "split must be train, val or test"
        self.dset_name = dset_name
        self.data_dir = data_dir
        self.split = split
        self.atomic_radius = atomic_radius
        self.max_n_atoms = max_n_atoms

        self.data = torch.load(os.path.join(data_dir, dset_name, f"{split}_data.pth"))
        if small:
            self.data = self.data[:5000]

        # Add any extra data preprocessing if needed
        if max_n_atoms > 0:
            self._filter_by_n_atoms()
        self._filter_by_elements(elements)

    def _filter_by_elements(self, elements: list):
        """
        Filters the data by elements.

        Args:
            elements (list): The list of elements to include in the dataset.
        """
        filtered_data = []
        elements_ids = [ELEMENTS_HASH[element] for element in elements]

        for datum in self.data:
            atoms = datum['atoms_channel'][datum['atoms_channel'] != 999]
            include = True
            for atom_id in atoms.unique():
                if int(atom_id.item()) not in elements_ids:
                    include = False
                    break
            if include:
                filtered_data.append(datum)
        if len(self.data) != len(filtered_data):
            print(f"  | filter data (elements): data reduced from {len(self.data)} to {len(filtered_data)}")
            self.data = filtered_data

    def _filter_by_n_atoms(self):
        """
        Filters the data by the maximum number of atoms.
        """
        filtered_data = []
        for datum in self.data:
            n_atoms = len(datum["atoms_channel"][datum["atoms_channel"] != 999])
            if n_atoms < self.max_n_atoms:
                filtered_data.append(datum)
        if len(self.data) != len(self.data):
            print(f"  | filter data (n atoms): data reduced from {len(self.data)} to {len(filtered_data)}")
            self.data = filtered_data

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Returns the sample at the given index.

        Args:
            index (int): The index of the sample.

        Returns:
            dict: The sample data.
        """
        sample_raw = self.data[index]
        sample = {
            "coords": sample_raw["coords"],
            "atoms_channel": sample_raw["atoms_channel"],
            "radius": torch.Tensor(sample_raw["atoms_channel"].shape).fill_(self.atomic_radius),
        }

        # Add noise/rotation on the coords, or any other augmentation
        sample = _center_coords(sample)
        sample = _rotate_coords(sample)
        sample = _shift_coords(sample, delta=.5)

        return sample


########################################################################################
# Data augmentation
def _center_coords(sample: dict):
    """
    Centers the coordinates of the atoms in the sample around the center of mass.

    Args:
        sample (dict): A dictionary containing the sample data.

    Returns:
        dict: The modified sample with centered coordinates.
    """
    coords = sample["coords"]
    mask = sample['atoms_channel'] != 999
    coords_masked = coords[mask]  # ignore value 999

    # go to center of mass
    center_coords = torch.mean(coords_masked, dim=0)
    center_coords = center_coords.unsqueeze(0).repeat(coords_masked.shape[0], 1)
    coords_masked = coords_masked - center_coords

    sample["coords"][mask] = coords_masked
    return sample


def _shift_coords(sample: dict, delta: float = .5):
    """
    Shifts the coordinates of atoms in the sample dictionary by adding random noise.

    Args:
        sample (dict): The sample dictionary containing the atoms and coordinates.
        delta (float, optional): The maximum magnitude of the random noise to be added. Defaults to 1.

    Returns:
        dict: The modified sample dictionary with shifted coordinates.
    """
    mask = sample['atoms_channel'] != 999
    noise = (torch.rand((1, 3)) - 1/2)*2*delta
    sample["coords"][mask] += noise.repeat(sample["coords"][mask].shape[0], 1)
    return sample


def _rotate_coords(sample: dict):
    """
    Rotate the coordinates of a sample using a random rotation matrix.

    Args:
        sample (dict): A dictionary containing the sample data.

    Returns:
        dict: The modified sample dictionary with rotated coordinates.
    """
    rot_matrix = _random_rot_matrix()

    coords = sample["coords"]

    idx = sample['atoms_channel'] != 999
    coords_masked = coords[idx]  # ignore value 999
    coords_masked = torch.reshape(coords_masked, (-1, 3))

    # go to center of mass
    center_coords = torch.mean(coords_masked, dim=0)
    center_coords = center_coords.unsqueeze(0).tile((coords_masked.shape[0], 1))
    coords_masked = coords_masked - center_coords

    coords_rot = torch.einsum("ij, kj -> ki", rot_matrix, coords_masked)
    coords[: coords_rot.shape[0], :] = coords_rot
    sample["coords"] = coords
    return sample


def _random_rot_matrix():
    """
    Generate a random rotation matrix.

    Returns:
        torch.Tensor: The random rotation matrix.
    """
    theta_x = random.uniform(0, 2) * math.pi
    rot_x = torch.Tensor([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)],
    ])
    theta_y = random.uniform(0, 2) * math.pi
    rot_y = torch.Tensor([
        [math.cos(theta_y), 0, -math.sin(theta_y)],
        [0, 1, 0],
        [math.sin(theta_y), 0, math.cos(theta_y)],
    ])
    theta_z = random.uniform(0, 2) * math.pi
    rot_z = torch.Tensor([
        [math.cos(theta_z), -math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1],
    ])

    return rot_z @ rot_y @ rot_x


########################################################################################
# Test
if __name__ == "__main__":
    from voxmol.voxelizer import Voxelizer
    from voxmol.utils import makedir, visualize_voxel_grid
    dset = DatasetVoxMol(dset_name="qm9", data_dir="data/")
    voxelizer = Voxelizer(grid_dim=32)
    for i in range(len(dset)):
        sample = dset[i]
        sample["coords"] = sample["coords"].unsqueeze(0)
        sample["atoms_channel"] = sample["atoms_channel"].unsqueeze(0)
        sample["radius"] = sample["radius"].unsqueeze(0)

        voxel = voxelizer(sample)
        dirname = "figs/"
        makedir(dirname)
        visualize_voxel_grid(voxel, fname=f"{dirname}/{i}.png", to_png=True, to_html=False)
        if i == 2:
            break
