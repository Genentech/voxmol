import argparse
import os
import torch
import yaml

from voxmol.models import create_model
from voxmol.utils import load_checkpoint, makedir, seed_everything, save_molecules_xyz


def main(config):
    # basic inits
    print(">> n gpus available:", torch.cuda.device_count())
    torch.set_default_dtype(torch.float32)
    seed_everything(args.seed)

    # load model and voxelizer
    model = create_model(config)
    model.to("cuda")
    model, _ = load_checkpoint(model, config["pretrained_path"], best_model=False)

    # start sampling
    print(">> start sampling...")
    molecules_xyz = model.sample(
        grid_dim=config["grid_dim"],
        n_batch_chains=config["n_chains"],
        n_repeats=config["n_repeats"],
        n_steps=config["steps_wjs"],
        max_steps=config["max_steps_wjs"],
        warmup_steps=config["warmup_wjs"],
        refine=True,
    )
    print(" >> done sampling")

    # save xyz molecules
    dirname_out = os.path.join(config["pretrained_path"], config["out_dir"])
    print(f">> saving samples in {dirname_out}")
    makedir(dirname_out)
    save_molecules_xyz(molecules_xyz, dirname_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sample wjs", add_help=False)
    parser.add_argument("--pretrained_path", default=None, help="path to pretrained model")
    parser.add_argument("--seed", default=1234, type=int, help="seed")
    parser.add_argument("--n_chains", default=50, type=int, help="n chains to run in parallel (eg batch size)")
    parser.add_argument("--out_dir", default="xyzs/", type=str, help="output dir")
    parser.add_argument("--warmup_wjs", default=0, type=int, help="warm-up steps for walk-jump sampling")
    parser.add_argument("--steps_wjs", default=500, type=int, help="n walk steps btw each jump in walk-jump sampling")
    parser.add_argument("--max_steps_wjs", default=1000, type=int, help="max number of steps for hte walk-jump chains")
    parser.add_argument("--n_repeats", default=10, type=int, help="number of repeats of chain generation")

    args = parser.parse_args()
    assert torch.cuda.is_available(), "you need GPUs to sample, otherwise it will take an eternity..."

    with open(os.path.join(args.pretrained_path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    for k, v in args.__dict__.items():
        config[k] = v
    config["steps_wjs"] = args.steps_wjs
    config["max_steps_wjs"] = args.max_steps_wjs
    config["n_repeats"] = args.n_repeats
    config["out_dir"] += f"_s{config['steps_wjs']}_ms{config['max_steps_wjs']}"

    # start sampling
    main(config)
