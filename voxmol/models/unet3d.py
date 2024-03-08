import math
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, Union, List

from voxmol.voxelizer import Voxelizer
from voxmol.utils import mol2xyz


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.use_norm = n_groups > 0
        # first norm + conv layer
        if self.use_norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        # second norm + conv layer
        if self.use_norm:
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        else:
            self.shortcut = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if self.use_norm:
            h = self.norm1(x)
            h = self.act1(h)
        else:
            h = self.act1(x)
        h = self.conv1(h)

        if self.use_norm:
            h = self.norm2(h)
        h = self.act2(h)
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 16
    ):

        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.use_norm = n_groups > 0
        self.n_heads = n_heads
        self.d_k = d_k

        if self.use_norm:
            self.norm = nn.GroupNorm(n_groups, n_channels)

        # Probably need to change this as we are in 3D
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)

        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width, depth = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum("bihd, bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)

        res = torch.einsum("bijh, bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)

        res = self.output(res)
        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width, depth)
        return res


class DownBlock(nn.Module):
    """
    This combines ResidualBlock and AttentionBlock .
    These are used in the first half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool, dropout: float):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    This combines ResidualBlock and AttentionBlock.
    These are used in the second half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool, dropout: float):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, n_groups: int, dropout: float):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)
        self.attn = AttentionBlock(n_channels, n_groups=n_groups)
        self.res2 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))
        # TODO: Upsample + Conv3d instead

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class UNet3D(nn.Module):
    # inspired by https://nn.labml.ai/diffusion/ddpm/unet.html
    def __init__(
        self,
        n_elements: int = 5,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0
    ):
        super().__init__()

        self.smooth_sigma = smooth_sigma
        self.n_elements = n_elements
        n_resolutions = len(ch_mults)

        # projection
        self.grid_projection = nn.Conv3d(n_elements, n_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # encoder
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        # bottleneck
        self.middle = MiddleBlock(out_channels, n_groups, dropout)

        # decoder
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        if n_groups > 0:
            self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(in_channels, n_elements, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # n params
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f">> model has {(n_params/1e6):.02f}M parameters")

    def forward(self, x: torch.Tensor):
        x = self.grid_projection(x)

        # encoder
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        # bottleneck
        x = self.middle(x)

        # decoder
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        if hasattr(self, "norm"):
            x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        return x

    def initialize_y_v(
        self,
        grid_dim: int = 32,
        n_chains: int = 10,
    ):
        """Initialize y and v for walk-jump sampling.

        Args:
            grid_dim (int, optional): dimension of the grid. Defaults to 32.
            n_chains (int, optional): number of chains (ie, batch size). Defaults to 10.

        Returns:
            torch.Tensor: y, v
        """
        # gaussian noise
        y = torch.cuda.FloatTensor(n_chains, self.n_elements, grid_dim, grid_dim, grid_dim)
        y.normal_(0, self.smooth_sigma)

        # uniform noise
        y += torch.cuda.FloatTensor(y.shape).uniform_(0, 1)

        return y, torch.zeros_like(y)

    def score(self, y: torch.Tensor):
        """Compute the score function of voxelized molecule y

        Args:
            y (torch.Tensor): voxelimzed molecule (BxCxLxLxL)

        Returns:
            torch.Tensor: score function of voxelized molecule y
        """
        xhat = self.forward(y)
        return (xhat - y) / (self.smooth_sigma ** 2)

    @torch.no_grad()
    def wjs_walk_steps(
        self,
        y: torch.Tensor,
        v: torch.Tensor,
        n_steps: int,
        delta: float = .5,
        friction: float = 1.,
        lipschitz: float = 1.,
    ):
        """"Walk steps of walk-jump sampling.
        Do config["steps_wjs"] Langevin MCMC steps on p(y).
        We Use Sachs et al. discretization of the underdamped Langevin MCMC.
        See the paper and its references on walk jump sampling.

        Args:
            y (torch.Tensor): sample y from mcmc chain
            v (torch.Tensor): velocity tensor (same dimensions as y)
            n_steps (int): number of steps to be performed
            delta (float, optional): WJS discretizing step. Defaults to .5.
            friction (float, optional): Langevin MCMC friction paraemter. Defaults to 1.
            lipschitz (float, optional): Langevin MCMC paraemter. Defaults to 1.

        Returns:
            torch.Tensor: y, v
        """
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(-friction)
        zeta2 = math.exp(-2 * friction)
        for _ in range(n_steps):
            y += delta * v / 2  # y_{t+1}
            psi = self.score(y)
            v += u * delta * psi / 2  # v_{t+1}
            v = zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * torch.randn_like(y)  # v_{t+1}
            y += delta * v / 2  # y_{t+1}
        torch.cuda.empty_cache()
        return y, v

    @torch.no_grad()
    def wjs_jump_step(self, y: torch.Tensor):
        """Jump step of walk-jump sampling.
        Recover clean sample x from noisy sample y.
        It is a simple forward of the network.

        Args:
            y (torch.Tensor): samples y from mcmc chain


        Returns:
            torch.Tensor: estimated ``clean'' samples xhats
        """
        return self.forward(y)

    def sample(
        self,
        grid_dim: int = 32,
        n_batch_chains: int = 100,
        n_repeats: int = 10,
        n_steps: int = 1000,
        max_steps: int = 1000,
        warmup_steps: int = 0,
        refine: bool = True,
        verbose: bool = True,
    ):
        """
        Generates molecular structures using the trained UNet3D model.

        Args:
            save_dir (str): The directory to save the generated molecules.
            grid_dim (int): The dimension of the voxel grid.
            n_batch_chains (int): The number of molecular chains to sample in each batch.
            n_repeats (int): The number of times to repeat the sampling process.
            n_steps (int): The number of steps to take in each iteration of the sampling process.
            max_steps (int): The maximum number of steps to take in total.
            warmup_steps (int): The number of warmup steps to take before starting the sampling process.
            refine (bool): Whether to refine the generated structures using voxel refinement.
            verbose (bool): Whether to print progress information during sampling.

        Returns:
            List[str]: A list of molecular structures in XYZ format.
        """
        self.eval()

        # create voxelizer to retrieve atomic coordinates
        voxelizer = Voxelizer(
            grid_dim=grid_dim,
            num_channels=self.n_elements,
            device="cuda:0",
        )

        # start sampling
        molecules_xyz = []
        for rep in range(n_repeats):
            if verbose:
                print("| repeat", rep)

            # initialize y and v
            if warmup_steps > 0:
                y, v = self.initialize_y_v(grid_dim=grid_dim, n_chains=1)
                y, v = self.wjs_walk_steps(y, v, warmup_steps)
                y = y.repeat(n_batch_chains, 1, 1, 1, 1)
                v = v.repeat(n_batch_chains, 1, 1, 1, 1)
            else:
                y, v = self.initialize_y_v(grid_dim=grid_dim, n_chains=n_batch_chains)

            # walk and jump
            for step in tqdm(range(0, max_steps, n_steps)):
                with torch.no_grad():
                    y, v = self.wjs_walk_steps(y, v, n_steps)  # walk
                    xhats = self.wjs_jump_step(y)  # jump

                xhats[xhats < .2] = 0
                mols = voxelizer.vox2mol(xhats, refine=refine)
                for i in range(xhats.shape[0]):
                    try:
                        mol = mols[i]
                        xyz_str = mol2xyz(mol)
                        molecules_xyz.append(xyz_str)
                    except Exception:
                        print(">> molecule not valid")
                        continue
        if verbose:
            print(f">> n valid molecules: {len(molecules_xyz)}")

        return molecules_xyz
