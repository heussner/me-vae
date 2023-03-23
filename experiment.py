import os
from cv2 import normalize
from matplotlib import pyplot as plt
import torch
from torch import optim
from models.me_vae import MEVAE
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from ml_collections import ConfigDict


class MultiEncoderVAE(pl.LightningModule):
    def __init__(
        self,
        model: MEVAE,
        optim_config: ConfigDict,
        n_samples: int,
        sample_step: int = 10,
    ) -> None:
        super(MultiEncoderVAE, self).__init__()
        self.model = model
        self.optim = optim_config
        self.n_samples = n_samples
        self.sample_step = sample_step
        self.step = 0
        self.colors = ["Blues", "Greens", "Reds", "Purples", "Oranges"]

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input1, input2, output, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input1, input2, output = batch
        self.curr_device = input1.device

        results = self.forward(input1, input2, output)
        train_loss = self.model.loss_function(
            *results,
            M_N=batch[0].size(0) / self.n_samples,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        for k, v in train_loss.items():
            self.log(k, v)

        if batch_idx % self.sample_step == 0:
            recons = results[0]
            output = results[1]
            #save_path = self.save_dir + f"{self.step}.png"
            for k in range(output.size(1)): 
                comp = torch.cat(
                    (
                        output[:64, k, :, :].unsqueeze(1),
                        recons[:64, k, :, :].unsqueeze(1)
                    ), 
                    -1)
                grid = make_grid(comp, normalize=True)
                assert (
                    (grid[0, :, :] == grid[1, :, :]).all().item() and 
                    (grid[0, :, :] == grid[2, :, :]).all().item()
                )
                self.logger.log_image(
                    "Inputs-Reconstructions", [grid[0,:,:].detach().cpu().numpy()])

            msg = f"loss: {train_loss['loss']:.4f} -- rec: {train_loss['reconstruction_loss']:.4f} -- kld (scaled): {train_loss['KLD_scaled']:.4f} -- kld: {train_loss['KLD']:.4f}"
            print(msg)

        for k, v in train_loss.items():
            if k != "loss" and v.grad_fn:
                train_loss[k] = v.detach()

        self.step += 1

        #self.log_dict(train_loss)

        return train_loss

    def configure_optimizers(self):

        optimizers = {"adam": optim.Adam, "sgd": optim.SGD}
        schedulers = {}

        optims = []
        scheds = []

        optimizer = optimizers[self.optim.type](
            self.model.parameters(),
            lr=self.optim.lr,
            weight_decay=self.optim.weight_decay,
        )

        optims.append(optimizer)

        if not self.optim.scheduler is None:
            scheduler = schedulers[self.optim.scheduler](**self.optim.scheduler_params)
            scheds.append(scheduler)

        return optims, scheds
