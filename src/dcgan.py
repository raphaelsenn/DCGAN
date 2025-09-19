"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-18
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Implementation of the CelebFaces-Generator.

    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """
    def __init__(
            self, 
            z_dim: int=100,
            channels_img: int=3,
            features_g: int=128,
            kernel_size: int=4,
            stride: int=2,
            padding: int=1
        ) -> None:
        super().__init__() 
        
        self.projection = nn.Linear(z_dim, features_g * 8 * 4 * 4, bias=False)
        self.features_g = features_g
        self.net = nn.Sequential(
            nn.BatchNorm2d(8 * features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(8*features_g, 4*features_g, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(4*features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*features_g, 2*features_g, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(2*features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*features_g, features_g, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels_img, kernel_size, stride, padding),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.view(-1, self.features_g * 8, 4, 4)
        return self.net(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Implementation of the CelebFaces-Discriminator.
 
    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """ 
    def __init__(
            self,
            channels_img: int=3,
            features_d: int=128,
            kernel_size: int=4,
            stride: int=2,
            padding: int=1
        ) -> None:
        super().__init__()
        self.features_d = features_d
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(features_d, 2*features_d, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(2*features_d), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*features_d, 4*features_d, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(4*features_d),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4*features_d, 8*features_d, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(8*features_d),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(8*features_d, 1, kernel_size, stride, padding),
            nn.Flatten(start_dim=1),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)