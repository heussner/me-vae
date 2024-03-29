import torch
from torch._C import Value
from torch import nn
from torch.nn import functional as F
from typing import List
from math import sqrt


class MEVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        inter_dim: int,
        hidden_dims: List = None,
        img_size: int = 128,
        dsample: int = -1,
        likehood_dist: str = "bern",
        **kwargs
    ) -> None:
        
        super(MEVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim
        self.img_size = img_size
        self.likelihood_dist = likehood_dist
        self.in_channels = in_channels
        self.dsample = dsample
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.last_hidden = hidden_dims[-1]
        
        modules = []
        # Build Encoder 1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    # nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        modules.append(nn.Flatten(start_dim=1),)
        self.encoder1 = nn.Sequential(*modules)
        self.dsample = img_size // (2**len(hidden_dims))
        self.dsample = 2 ** self.dsample
        self.inter1 = nn.Linear(hidden_dims[-1]*self.dsample, inter_dim)

        self.fc_mu1 = nn.Linear(inter_dim, latent_dim)
        self.fc_var1 = nn.Linear(inter_dim, latent_dim)

        # Build Encoder 2
        in_channels = self.in_channels
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    # nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim
        
        modules.append(nn.Flatten(start_dim=1),)
        self.encoder2 = nn.Sequential(*modules)
        self.inter2 = nn.Linear(hidden_dims[-1]*self.dsample, inter_dim)

        self.fc_mu2 = nn.Linear(inter_dim, latent_dim)
        self.fc_var2 = nn.Linear(inter_dim, latent_dim)
        
        #Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*16)

        hidden_dims.reverse()
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels=h_dim,
                        stride=2,
                        kernel_size=3,
                        padding=1,
                        output_padding=1,
                    ),
                    # nn.BatchNorm2d(hidden_dims[-1]),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1
            ),
            nn.Sigmoid(),
        )
    
    def encode1(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder1(input)
        result = self.inter1(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu1(result)
        log_var = self.fc_var1(result)

        return [mu, log_var]
    
    def encode2(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder2(input)
        result = self.inter2(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu2(result)
        log_var = self.fc_var2(result)

        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        shape = int(sqrt(self.dsample))
        result = result.view(-1, self.last_hidden, shape, shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (torch.Tensor) Mean of the latent Gaussian
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, output: torch.Tensor, **kwargs) -> torch.Tensor:
        #encode inputs
        mu1, log_var1 = self.encode1(input1)
        mu2, log_var2 = self.encode2(input2)
        #reparameterize
        z1 = self.reparameterize(mu1, log_var1)
        z2 = self.reparameterize(mu2, log_var2)
        #multiply z1,z1
        z = torch.mul(z1, z2)
        #decode
        return [self.decode(z), output, mu1, log_var1, mu2, log_var2]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        """
        #unpack args
        recons = args[0]
        output = args[1]
        mu1 = args[2]
        log_var1 = args[3]
        mu2 = args[4]
        log_var2 = args[5]
        
        #reconstruction for decoder + KLD loss for both encoders
        kld_weight = kwargs["M_N"]# Account for the minibatch samples from the dataset
        kld_weight = 1
        if self.likelihood_dist == "gauss":
            recons_loss = F.mse_loss(recons, output)
        elif self.likelihood_dist == "bern":
            assert recons.isfinite().all()
            assert (recons <= 1).all()
            assert (recons >= 0).all()
            assert output.isfinite().all()
            assert (output <= 1).all()
            assert (output >= 0).all()

            recons_loss = F.binary_cross_entropy(recons, output)
            recons_loss *= (self.img_size ** 2)
        else:
            raise ValueError("Undefined likelihood distribution.")
        
        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - torch.exp(log_var1), dim=1), dim=0)
        
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - torch.exp(log_var2), dim=1), dim=0)
        
        kld_loss = (kld_loss1 + kld_loss2)/2 #averaging kld loss
        kld_scaled = kld_weight * kld_loss
        loss = recons_loss + kld_scaled
        return {"loss": loss, "reconstruction_loss": recons_loss, "KLD": kld_loss, "KLD_scaled": kld_scaled}
    
    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
