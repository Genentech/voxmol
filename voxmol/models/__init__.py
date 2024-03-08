from voxmol.models.unet3d import UNet3D


def create_model(config):
    if config["model_name"] == "unet3d":
        model = UNet3D(
            n_elements=len(config["elements"]),
            n_channels=config["n_channels"],
            ch_mults=config["ch_mults"],
            is_attn=config["is_attn"],
            n_blocks=config["n_blocks"],
            n_groups=config["n_groups"],
            dropout=config["dropout"],
            smooth_sigma=config["smooth_sigma"]
        )
    else:
        NotImplementedError(f"{config['model_type']} Not implemented yet")

    return model
