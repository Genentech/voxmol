import numpy as np
from copy import deepcopy
from functools import partial
from scipy import ndimage as ndi
import torch

from pyuul import VolumeMaker
from torch import nn


class Voxelizer(nn.Module):
    def __init__(
            self,
            grid_dim: int = 32,
            num_channels: int = 5,
            resolution: float = .25,
            cubes_around: int = 5,
            radius: float = .5,
            device: str = "cuda"
    ):
        """Convert back and forth from molecule to voxel grid

        Args:
            grid_dim (int, optional): lenght of each dimension of the voxel grid.
                Defaults to 32.
            density_fn (str, optional): density function for atoms ("gaussian"
            or "sigmoid"). Defaults to "gaussian".
            resolution (float, optional): grid resolution. Default .5.
            cubes_around (int, optional): cubes around parameter for pyuul. Default 5.
            radius (int, optional): radius of each atom. Default .5.
            num_channels (int, optional): number of channels on voxel grid. Default 5.
            device (str, optional): cpu or cuda. Defaults to "cpu".
        """
        super(Voxelizer, self).__init__()
        self.grid_dim = grid_dim
        self.device = device
        self.resolution = resolution
        self.cubes_around = cubes_around
        self.radius = radius
        self.num_channels = num_channels

        self.vol_maker = VolumeMaker.Voxels(
            device=device,
            sparse=False,
        )

    def forward(self, batch):
        return self.mol2vox(batch)

    def mol2vox(self, batch):
        """
        Convert a batch of molecules to voxel representation.

        Args:
            batch (dict): A dictionary containing the molecule information.

        Returns:
            torch.Tensor: Voxel representation of the molecules.
        """
        # dumb coordinates to center molecule
        batch = self._add_dumb_coords(batch)

        voxels = self.vol_maker(
                batch["coords"].to(self.device),
                batch["radius"].to(self.device),
                batch["atoms_channel"].to(self.device),
                resolution=self.resolution,
                cubes_around_atoms_dim=self.cubes_around,
                function="gaussian",
                numberchannels=self.num_channels,
            )
        # get center box (remove dumb coordinates)
        c = voxels.shape[-1] // 2
        box_min, box_max = c - self.grid_dim // 2, c + self.grid_dim // 2
        voxels = voxels[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
        return voxels

    def vox2mol(self, voxels: torch.Tensor, refine: bool = True):
        """
        Converts voxel data to molecular coordinates.

        Args:
            voxels (torch.Tensor): Voxel data with shape (batch_size, channels, depth, height, width).
            refine (bool, optional): Whether to refine the coordinates. Defaults to True.

        Returns:
            Union[None, List[Dict[str, torch.Tensor]]]: Refined molecular coordinates if `refine` is True,
            otherwise initial molecular coordinates. Returns None if no valid molecular coordinates are found.
        """
        assert len(voxels.shape) == 5

        # intialize coods with simple peak detection
        mol_inits = []
        voxel_inits = []
        for voxel in voxels:
            peaks = find_peaks(voxel.cpu())
            mol_init = get_atom_coords(peaks, rad=self.radius, resolution=self.resolution)
            if mol_init is not None and mol_init["coords"].shape[1] < 200:
                mol_inits.append(mol_init)
                voxel_inits.append(voxel.unsqueeze(0))

        if len(mol_inits) == 0:
            return None

        if not refine:
            return mol_inits

        voxel_inits = torch.cat(voxel_inits, axis=0)

        # refine coords
        optim_factory = partial(
            torch.optim.LBFGS, history_size=10, max_iter=4, line_search_fn="strong_wolfe",
        )

        mols = self._refine_coords(mol_inits, voxel_inits, optim_factory, maxiter=10)
        del voxels, mol_inits, voxel_inits
        torch.cuda.empty_cache()

        return mols

    def _refine_coords(
        self,
        mol_inits: list,
        voxels: torch.Tensor,
        optim_factory: callable,
        tol: float = 1e-6,
        maxiter: int = 10,
    ):
        """
        Refines the coordinates of molecules based on voxel data.

        Args:
            mol_inits (list): List of initial molecule configurations.
            voxels (torch.Tensor): Voxel data with shape (batch_size, channels, depth, height, width).
            optim_factory (callable): Factory function to create an optimizer.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
            maxiter (int, optional): Maximum number of iterations. Defaults to 10.
            callback (callable, optional): Callback function called after each iteration. Defaults to None.

        Returns:
            list: List of refined molecule configurations, each containing the refined
                coordinates, atoms channel, and radius.
        """

        assert len(voxels.shape) == 5, "voxels need to have dimension 5 (including the batch dim.)"

        mols = []
        for i in range(voxels.shape[0]):
            mol_init = mol_inits[i]
            voxel = voxels[i].unsqueeze(0)

            mol = deepcopy(mol_init)
            mol["coords"].requires_grad = True

            optimizer = optim_factory([mol["coords"]])

            def closure():
                optimizer.zero_grad()
                voxel_fit = self.forward(mol)
                loss = torch.nn.functional.mse_loss(voxel, voxel_fit)
                loss.backward()
                return loss

            loss = 1e10
            for _ in range(maxiter):
                try:
                    prev_loss = loss
                    loss = optimizer.step(closure)
                except Exception:
                    print(
                        "refine coords diverges, so use initial cordinates...",
                        f"(coords min: {mol['coords'].min().item()}, max: {mol['coords'].max().item()})"
                    )
                    mol = deepcopy(mol_init)
                    break

                if abs(loss.item() - prev_loss) < tol:
                    break

            mols.append({
                "coords": mol["coords"].detach().cpu(),
                "atoms_channel": mol["atoms_channel"].detach().cpu(),
                "radius": mol["radius"].detach().cpu(),
            })

        return mols

    def _add_dumb_coords(self, batch: dict):
        bsz = batch['coords'].shape[0]
        return {
            "coords": torch.cat(
                (batch['coords'], torch.Tensor(bsz, 1, 3).fill_(-15), torch.Tensor(bsz, 1, 3).fill_(15)), 1
            ),
            "atoms_channel": torch.cat(
                (batch['atoms_channel'], torch.Tensor(bsz, 2).fill_(0)), 1
            ),
            "radius": torch.cat(
                (batch['radius'], torch.Tensor(bsz, 2).fill_(.5), ), 1
            )
        }


########################################################################################
# aux functions
def local_maxima(data: np.ndarray, order: int = 1):
    """
    Find local maxima in a 3D array.

    Args:
        data (ndarray): The input 3D array.
        order (int, optional): The order of the local maxima filter. Defaults to 1.

    Returns:
        ndarray: The modified 3D array with local maxima set to non-zero values.
    """
    data = data.numpy()
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    data[data <= filtered] = 0
    return data


def find_peaks(voxel: torch.Tensor):
    """
    Find peaks in a voxel tensor.

    Args:
        voxel (torch.Tensor): Input voxel tensor.

    Returns:
        torch.Tensor: Tensor containing the coordinates of the peaks.
    """
    voxel[voxel < .25] = 0
    voxel = voxel.squeeze().clone()
    peaks = []
    for channel_idx in range(voxel.shape[0]):
        vox_in = voxel[channel_idx]
        peaks_ = local_maxima(vox_in, 1)
        peaks_ = torch.Tensor(peaks_).unsqueeze(0)
        peaks.append(peaks_)
    peaks = torch.concat(peaks, axis=0)
    return peaks


def get_atom_coords(grid: torch.Tensor, rad: float = 0.5, resolution: float = 0.25):
    """
    Get the coordinates of atoms in a grid.

    Args:
        grid (torch.Tensor): The input grid.
        rad (float, optional): The radius of the atoms. Defaults to 0.5.
        resolution (float, optional): The resolution of the grid. Defaults to 0.25.

    Returns:
        dict: A dictionary containing the coordinates, atoms_channel, and radius of the atoms.
    """
    coords = []
    atoms_channel = []
    radius = []

    for channel_idx in range(grid.shape[0]):
        px, py, pz = torch.where(grid[channel_idx] > 0)
        px, py, pz = px.float(), py.float(), pz.float()
        coords.append(torch.cat([px.unsqueeze(1), py.unsqueeze(1), pz.unsqueeze(1)], axis=1))
        atoms_channel.append(torch.Tensor(px.shape[0]).fill_(channel_idx))
        radius.append(torch.Tensor(px.shape[0]).fill_(rad))

    coords = (torch.cat(coords, 0).unsqueeze(0) - grid.shape[-1] / 2) * resolution
    if coords.shape[1] == 0:
        return None
    coords = coords - torch.mean(coords, 1)

    return {
        "coords": coords,
        "atoms_channel": torch.cat(atoms_channel, 0).unsqueeze(0),
        "radius": torch.cat(radius, 0).unsqueeze(0),
    }
