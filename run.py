from logging import log
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch, os
from absl import app, flags
from ml_collections.config_flags import config_flags
from experiment import MultiEncoderVAE
import pickle
from data import load_data
from models import vae_models
from torchsummary import summary


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("train_config")
config_flags.DEFINE_config_file("model_config")


def main(_):
    config = FLAGS.train_config
    model_config = FLAGS.model_config

    # https://pytorch.org/docs/stable/notes/randomness.html
    if config.deterministic:
        seed_everything(config.manual_seed, workers=True)
        torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    v = 0 if config.logging.fix_version else None
    logger = TensorBoardLogger(
        save_dir=config.logging.save_dir, name=config.logging.name, version=v, **{"flush_secs": 5},
    )

    logger.log_hyperparams({})
    log_dir = f"{logger.save_dir}/{logger.name}/version_{logger.version}/"
    with open(os.path.join(log_dir, "hparams.pkl"), "wb") as f:
        pickle.dump({"main": config, "model": model_config}, f)

    model_class = vae_models[model_config.model]
    if config.load_checkpoint:
        print(f"Loading model from checkpoint: {config.model_path}")
        model = model_class.load_from_checkpoint(
            config.model_path, **model_config.model_params
        )
    else:
        print("Instantiating new model.")
        model = model_class(**model_config.model_params)

    summary(
        model, 
        [(
            model_config.model_params.in_channels, 
            model_config.model_params.img_size, 
            model_config.model_params.img_size
        ),
            (
            model_config.model_params.in_channels, 
            model_config.model_params.img_size, 
            model_config.model_params.img_size
        ),
            (
            model_config.model_params.in_channels, 
            model_config.model_params.img_size, 
            model_config.model_params.img_size
        )]
    )

    dataset, trainloader = load_data(config.datapath1, config.datapath2, config.datapath3, **config.data_params)
    n_samples = len(dataset)

    experiment = MultiEncoderVAE(model, config.optim, n_samples, config.sample_step)

    callbacks = []
    if config.early_stopping.do:
        early_stopping = EarlyStopping(**config.early_stopping_params)
        callbacks.append(early_stopping)
    if config.auto_checkpoint:
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(log_dir, config.checkpoint_dir),
            monitor=config.checkpoint_monitor,
            save_last=True,
            mode=config.checkpoint_mode,
            every_n_train_steps=50,
        )
        callbacks.append(checkpoint)

    trainer = Trainer(
        logger=logger,
        accelerator=config.accelerator,
        strategy=config.accel_strategy,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        gpus=config.num_gpus,
        log_every_n_steps=2,
        deterministic=config.deterministic,
        sync_batchnorm=True,
    )

    trainer.fit(model=experiment, train_dataloaders=trainloader)


if __name__ == "__main__":
    app.run(main)
