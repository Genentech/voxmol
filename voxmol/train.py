import os
import time
import getpass as gt
import wandb
import torch

from voxmol.options import parse_args
from voxmol.models import create_model
from voxmol.dataset import create_loader
from voxmol.metrics import create_metrics, MetricsDenoise
from voxmol.models.ema import ModelEma
from voxmol.models.adamw import AdamW
from voxmol.voxelizer import Voxelizer
from voxmol.utils import seed_everything, save_checkpoint, load_checkpoint, makedir, save_molecules_xyz


def main():
    # ----------------------
    # basic inits
    config = parse_args()
    print(">> n gpus available:", torch.cuda.device_count())
    torch.set_default_dtype(torch.float32)
    seed_everything(config["seed"])
    if config["wandb"] > 0:
        wandb.init(
            project="voxmol",
            entity=gt.getuser(),
            config=config,
            name=config["exp_name"],
            dir=config["output_dir"],
            settings=wandb.Settings(code_dir=".")
        )

    # ----------------------
    # data loaders
    start_epoch = 0
    loader_train, loader_val = create_loader(config)

    # ----------------------
    # voxelizer, model, criterion, optimizer, scheduler
    device = torch.device("cuda")
    voxelizer = Voxelizer(
        grid_dim=config["grid_dim"],
        num_channels=len(config["elements"]),
        device=device,
    )
    model = create_model(config)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss(reduction="sum").to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["wd"],
        betas=[0.99, 0.999],
    )
    optimizer.zero_grad()

    # optionally resume training
    if config["resume"] is not None:
        model, optimizer, start_epoch = load_checkpoint(
            model, config["output_dir"], optimizer, best_model=False
        )
        os.system(f"cp {os.path.join(config['output_dir'], 'checkpoint.pth.tar')} " +
                  f"{os.path.join(config['output_dir'], f'checkpoint_{start_epoch}.pth.tar')}")

    # ema (exponential moving average)
    model_ema = ModelEma(model, decay=.999)

    # ----------------------
    # metrics
    metrics = create_metrics()

    # ----------------------
    # start training
    print(">> start training...")
    for epoch in range(start_epoch, start_epoch + config["num_epochs"]):
        t0 = time.time()

        # train
        train_metrics = train(
            loader_train, voxelizer, model, model_ema, criterion, optimizer, metrics, config
        )

        # val
        val_metrics = val(
            loader_val, voxelizer, model_ema.module, criterion, metrics, config
        )

        # sample
        if epoch > 0 and epoch % 50 == 0:
            print(f"| sampling at epoch {epoch}")
            sample(model_ema.module, config, epoch)

        # print metrics, log wandb
        print_metrics(epoch, time.time()-t0, train_metrics, val_metrics)
        if config["wandb"] > 0:
            wandb.log({"train": train_metrics, "val": val_metrics, "sampling": None})

        # save model
        save_checkpoint({
            "epoch": epoch + 1,
            "config": config,
            "state_dict_ema": model_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, config=config)


def train(
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn.Module,
    model_ema: ModelEma,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: MetricsDenoise,
    config: dict,
):
    """
    Trains the model using the given data loader, voxelizer, model, criterion,
    optimizer, and metrics.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for loading the training data.
        voxelizer (Voxelizer): The voxelizer for converting input data into voxels.
        model (torch.nn.Module): The model to be trained.
        model_ema (ModelEma): The exponential moving average of the model parameters.
        criterion (torch.nn.Module): The loss function for calculating the training loss.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        metrics (MetricsDenoise): The metrics object for tracking the training metrics.
        config (dict): The configuration dictionary containing training settings.

    Returns:
        dict: The computed metrics for the training process.
    """
    metrics.reset()
    model.train()

    for i, batch in enumerate(loader):
        # voxelize
        voxels = voxelizer.forward(batch)
        smooth_voxels = add_noise_voxels(voxels, config["smooth_sigma"])

        # forward/backward
        pred = model(smooth_voxels)
        loss = criterion(pred, voxels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model_ema.update(model)

        # update metrics
        metrics.update(loss, pred, voxels)

        if i*config["batch_size"] >= 100_000:
            break
        if config["debug"] and i == 10:
            break

    return metrics.compute()


def val(
    loader: torch.utils.data.DataLoader,
    voxelizer: Voxelizer,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metrics: MetricsDenoise,
    config: dict,
):
    """
    Perform validation on the given data loader using the provided model and criterion.

    Args:
        loader (torch.utils.data.DataLoader): Data loader for validation data.
        voxelizer (Voxelizer): Voxelizer object for converting input data to voxels.
        model (torch.nn.Module): Model to be used for prediction.
        criterion (torch.nn.Module): Loss criterion for calculating the loss.
        metrics (MetricsDenoise): Metrics object for tracking evaluation metrics.
        config (dict): Configuration dictionary containing various settings.

    Returns:
        float: Computed metrics for the validation data.
    """
    metrics.reset()
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # voxelize
            voxels = voxelizer(batch)
            smooth_voxels = add_noise_voxels(voxels, config["smooth_sigma"])

            # forward
            pred = model(smooth_voxels)
            loss = criterion(pred, voxels)

            # update metrics
            metrics.update(loss, pred, voxels)

            if config["debug"] and i == 10:
                break
    return metrics.compute()


def sample(
    model: torch.nn.Module,
    config: dict,
    epoch: int = -1
):
    """
    Generate samples using the given model.

    Args:
        model (torch.nn.Module): The model used for sampling.
        config (dict): Configuration parameters for sampling.
        epoch (int, optional): The epoch number. Defaults to -1.
    """
    if torch.cuda.device_count() > 1:
        model = model.module
    model.eval()

    # sample
    molecules_xyz = model.sample(
        grid_dim=config["grid_dim"],
        n_batch_chains=config["n_chains"],
        n_repeats=config["repeats_wjs"],
        n_steps=config["steps_wjs"],
        max_steps=config["max_steps_wjs"],
        warmup_steps=config["warmup_wjs"],
        refine=True,
    )

    # save molecules on xyz format
    dirname_out = os.path.join(config["output_dir"], "samples/", f"epoch={epoch}/")
    print(f">> saving samples in {dirname_out}")
    makedir(dirname_out)
    save_molecules_xyz(molecules_xyz, dirname_out)


def print_metrics(
    epoch: int,
    time: float,
    train_metrics: list,
    val_metrics: list,
    sampling_metrics: dict = None,
):
    """
    Print the metrics for each epoch.

    Args:
        epoch (int): The current epoch number.
        time (float): The time taken for the epoch.
        train_metrics (list): The metrics for the training set.
        val_metrics (list): The metrics for the validation set.
        sampling_metrics (dict, optional):The metrics for the sampling.Defaults to None.
    """
    all_metrics = [train_metrics, val_metrics, sampling_metrics]
    metrics_names = ["train", "val", "sampling"]

    str_ = f">> epoch: {epoch} ({time:.2f}s)"
    for (split, metric) in zip(metrics_names, all_metrics):
        if metric is None:
            continue
        str_ += "\n"
        str_ += f"[{split}]"
        for k, v in metric.items():
            if k == "loss":
                str_ += f" | {k}: {v:.4f}"
            else:
                str_ += f" | {k}: {v:.4f}"
    print(str_)


def add_noise_voxels(voxels: torch.Tensor, sigma: float):
    """
    Adds Gaussian noise to the input voxels.

    Args:
        voxels (torch.Tensor): Input tensor representing the voxels.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Tensor with Gaussian noise added to the input voxels.
    """
    if sigma > 0:
        return voxels + torch.cuda.FloatTensor(voxels.shape).normal_(0, sigma)
    return voxels


if __name__ == "__main__":
    main()
