import ml_collections


def get_config():
    cfg = ml_collections.ConfigDict()

    cfg.model = "me_vae"
    cfg.model_params = {
        "img_size": 128,
        "in_channels": 1,
        "latent_dim": 128,
        "hidden_dims": [32, 64, 128, 256, 512],
        "n_downsample": 3,
        "inter_dim": 512,
        "max_capacity": 25,
        "Capacity_max_iter": 1e5,
        "likelihood_dist": "gauss",  # Decoder modeling a 'gauss'ian or 'bern'ouli distribution
        "loss_type": "H",  # 'B' or 'H' -- see https://openreview.net/forum?id=Sy2fzU9gl and https://arxiv.org/pdf/1804.03599.pdf
    }

    return cfg
