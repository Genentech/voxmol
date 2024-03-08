import argparse
import os
import torch
import yaml

from voxmol.utils import create_exp_dir


def parse_args():
    parser = argparse.ArgumentParser("voxmol", add_help=False)
    parser.add_argument(
        "--debug", action="store_true", help="debug mode"
    )
    parser.add_argument(
        "--resume", default=None, help="directory containing the trained model to resume training"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, help="num workers"
    )
    parser.add_argument(
        "--exp_dir", default="exps/", type=str, help="experiment dir"
    )
    parser.add_argument(
        "--exp_name", default=None, type=str, help="experiment name"
    )
    parser.add_argument(
        "--wandb", default=0, type=int, help="use wandb if > 0"
    )

    # data args
    parser.add_argument(
        "--data_dir", default="dataset/data/", type=str, help="data dir"
    )
    parser.add_argument(
        "--dset_name", default="qm9", type=str, help="dataset name (qm9 | drugs)"
    )
    parser.add_argument(
        "--smooth_sigma", default=0.9, type=float, help="noise level of smooth density"
    )
    parser.add_argument(
        "--grid_dim", default=None, type=int, help="value for each dimension of voxel grid"
    )

    # training args
    parser.add_argument(
        "--num_epochs", default=500, type=int, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size for training"
    )

    # optim args
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="absolute learning rate"
    )
    parser.add_argument(
        "--wd", type=float, default=1e-2, help="weight decay coeff."
    )

    # model args
    parser.add_argument(
        "--model_config", default="models/configs/unet3d_config.yml", help="model path"
    )

    # wjs args
    parser.add_argument(
        "--n_chains", default=50, type=int, help="n of chains to be gen in parallel"
    )
    parser.add_argument(
        "--steps_wjs", default=500, type=int, help="number of walk steps for the wjs"
    )
    parser.add_argument(
        "--max_steps_wjs", default=1000, type=int, help="max step for wjs chain"
    )
    parser.add_argument(
        "--warmup_wjs", default=0, type=int, help="warm-up steps for walk-jump sampling"
    )
    parser.add_argument(
        "--repeats_wjs", default=5, type=int, help="number of (batched) wjs chains"
    )

    args = parser.parse_args()
    config = args.__dict__
    assert config["dset_name"] in ["qm9", "drugs"], "dataset not supported"
    assert torch.cuda.is_available(), "you need GPUs to sample, otherwise it will take an eternity..."

    # update global config with model config options
    with open(config["model_config"], "r") as f:
        config_model = yaml.safe_load(f)
    for k, v in config_model.items():
        if k not in config or config[k] is None:
            config[k] = v

    # set default 'elements', 'grid_dim' and 'num_channels' according to the dataset
    if config["dset_name"] == "qm9":
        config["elements"] = ["C", "H", "O", "N", "F"]
    elif config["dset_name"] == "drugs":
        config["elements"] = ["C", "H", "O", "N", "F", "S", "Cl", "Br"]
    config["grid_dim"] = 32 if config["dset_name"] == "qm9" else 64
    config["num_channels"] = len(config["elements"])

    # create experiment name and dir to save outputs of experiment
    if config["exp_name"] is None:
        config["exp_name"] = f"exp_{config['dset_name']}_sig{config['smooth_sigma']}_lr{config['lr']}"
    create_exp_dir(config)

    if config["resume"] is not None:
        with open(os.path.join(args.resume, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        config["resume"] = args.resume
        config["output_dir"] = args.resume

    print(">> config:")
    for k, v in config.items():
        print(f"  | {k}: {v}")

    return config
