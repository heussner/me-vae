import ml_collections
import os
import torch
from torchvision.transforms import (
    Compose,
    Resize,
)


def get_config():
    cfg = ml_collections.ConfigDict()

    cfg.deterministic = False
    cfg.max_epochs = 50
    cfg.batch_size = 64
    cfg.manual_seed = 100
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.accelerator = "gpu" if cfg.device == "cuda" else "cpu"
    # cfg.num_gpus = torch.cuda.device_count() if cfg.device == "cuda" else 0
    cfg.num_gpus = 1 if cfg.device == "cuda" else 0
    cfg.accel_strategy = "ddp" if cfg.num_gpus >= 2 else None
    cfg.detect_anomaly = False
    cfg.sample_step = 10  # every sample_step batches model logs

    # data
    cfg.datapath1 = "/home/groups/ChangLab/heussner/crc-data/pbmc-crops/cell_train1/cell_train"
    cfg.datapath2 = "/home/groups/ChangLab/heussner/crc-data/pbmc-crops/cell_train1/cell_train"
    cfg.img_size = (128, 128)
    cfg.data_params = {
        "dataset": {"transform": Resize(cfg.img_size),},
        "loader": {
            "batch_size": cfg.batch_size,
            "shuffle": False if cfg.deterministic else True,
            "pin_memory": False,
            "num_workers": os.cpu_count(),
        },
    }

    # optimizer
    cfg.optim = ml_collections.ConfigDict()
    cfg.optim.type = "adam"
    cfg.optim.lr = 0.0005
    cfg.optim.weight_decay = 0.0
    cfg.optim.scheduler = None
    cfg.optim.scheduler_params = None

    # logging details
    cfg.logging = ml_collections.ConfigDict()
    cfg.logging.deterministic = True
    cfg.logging.name = "5channel_pbmc_capan2sw480_mar29_2022"
    cfg.logging.debug = True
    cfg.logging.save_dir = (
        "/home/groups/ChangLab/heussner/single-cell-vae/single_cell_vae/logs"
    )
    cfg.logging.fix_version = False

    # early stopping
    cfg.early_stopping = ml_collections.ConfigDict()
    cfg.early_stopping.do = False
    cfg.early_stopping_params = {
        "metric": "train_loss",
        "mode": "min",
        "min_delta": 0.0,
        "patience": 20,
        "check_finite": True,
        "verbose": True,
    }

    # auto checkpointing
    cfg.auto_checkpoint = True
    cfg.checkpoint_dir = "checkpoints"
    cfg.checkpoint_monitor = "loss"
    cfg.checkpoint_mode = "min"
    cfg.load_checkpoint = False
    cfg.model_path = None

    cfg.eval_mode = False
    cfg.eval_model_path = "/home/users/strgar/strgar/single-cell-vae/single_cell_vae/logs/pbmc_capan2sw480_mar17_2022/version_0/checkpoints/last.ckpt"
    cfg.results_dir = (
        "/home/users/strgar/strgar/single-cell-vae/single_cell_vae/logs/pbmc_capan2sw480_mar17_2022/version_0/results"
    )
    cfg.vae_embed_file = "vae.csv"
    cfg.tsne_embed_file = "tsne.csv"
    cfg.umap_embed_file = "umap.csv"

    return cfg
