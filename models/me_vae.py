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
        hidden_dims: List = None,
        img_size: int = 128,
        likehood_dist: str = "gauss",
        **kwargs
    ) -> None:
        
        super(MEVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.likelihood_dist = likehood_dist
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # Build Encoder 1
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
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim
        
        self.encoder1 = nn.Sequential(*modules)
        self.dsample = self.img_size // (2 ** len(hidden_dims))
        self.dsample **= 2
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.dsample, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.dsample, latent_dim)
        
        # added second encoder
        # Build Encoder 2
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
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim
        
        self.encoder2 = nn.Sequential(*modules)
        
        #Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.dsample)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
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
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    #added second encode fn
    def encode2(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder2(input)
        result = self.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

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
        result = result.view(-1, 512, shape, shape)
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

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, **kwargs) -> torch.Tensor:
        #mu/var from both encoders
        mu1, log_var1 = self.encode1(input1)
        mu2, log_var2 = self.encode2(input2)
        #reparameterize from separate latent spaces
        z1 = self.reparameterize(mu1, log_var1)
        z2 = self.reparameterize(mu2, log_var2)
        #multiply z1,z1
        z = torch.mul(z1, z2)
        #return decoded z, both inputs/mus/vars
        return [self.decode(z), input1, mu1, log_var1, input2, mu2, log_var2]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        """
        #unpack output of forward fn
        recons = args[0]
        input1 = args[1]
        mu1 = args[2]
        log_var1 = args[3]
        input2 = args[4]
        mu2 = args[5]
        log_var2 = args[6]
        
        #reconstruction + KLD loss for both inputs/latents
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        if self.likelihood_dist == "gauss":
            recons_loss1 = F.mse_loss(recons, input1)
            recons_loss2 = F.mse_loss(recons, input2)
        elif self.likelihood_dist == "bern":
            recons_loss1 = F.binary_cross_entropy_with_logits(recons, input1)
            recons_loss2 = F.binary_cross_entropy_with_logits(recons, input2)
        else:
            raise ValueError("Undefined likelihood distribution.")
        
        recons_loss = recons_loss1 + recons_loss2
        
        kld_loss1 = torch.mean(
            -0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=1), dim=0
        )
        
        kld_loss2 = torch.mean(
            -0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=1), dim=0
        )
        
        kld_loss = kld_loss1 + kld_loss2
        
        loss1 = recons_loss1 + kld_weight * kld_loss1 
        loss2 = recons_loss2 + kld_weight * kld_loss2
        loss = loss1 + loss2
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}
    
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
