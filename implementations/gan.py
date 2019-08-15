from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
import test_tube


# Generator
class Generator(nn.Module):
    def __init__(self, z_ch=128, image_shape=(1, 28, 28)):
        super(Generator, self).__init__()

        self.z_ch = z_ch
        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(z_ch, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self, inputs):
        h = self.model(inputs)
        out = h.view(inputs.size(0), *self.image_shape)

        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()

        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        h = inputs.view(inputs.size(0), -1)
        out = self.model(h)

        return out


class GAN(pl.LightningModule):
    def __init__(self, z_ch=128, image_shape=(1, 28, 28),
                 lr=0.001, b1=0.5, b2=0.999, 
                 data_root="/tmp/data", batch_size=64, device=torch.device("cuda")):
        super(GAN, self).__init__()

        self.z_ch = z_ch
        self.image_shape = image_shape

        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.data_root = data_root
        self.batch_size = batch_size

        self.device = device

        self.generator = Generator(z_ch=z_ch, image_shape=image_shape).to(device)
        self.discriminator = Discriminator(image_shape=image_shape).to(device)

    def forward(self, z):
        return self.generator(z)

    def adv_loss(self, outputs, targets):
        return F.binary_cross_entropy(outputs, targets)

    def training_step(self, batch, batch_nb, optimizer_idx):
        x_real, _ = batch
        x_real = x_real.to(self.device)

        z = torch.randn(x_real.size(0), self.z_ch).to(self.device)
        x_fake = self.forward(z)

        self.sample_images(x_fake, 0)

        if optimizer_idx == 0:

            y_fake = torch.ones(x_real.size(0), 1).to(self.device)
            d_fake = self.discriminator(x_fake)

            g_loss = self.adv_loss(d_fake, y_fake)

            return g_loss

        if optimizer_idx == 1:
            y_real = torch.ones(x_real.size(0), 1).to(self.device)
            y_fake = torch.zeros(x_real.size(0), 1).to(self.device)

            d_real = self.discriminator(x_real)
            d_fake = self.discriminator(x_fake.detach())

            d_loss = (self.adv_loss(d_real, y_real) + self.adv_loss(d_fake, y_fake)) / 2

            return d_loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        return [g_optimizer, d_optimizer], []

    @pl.data_loader
    def tng_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

        dataset = torchvision.datasets.MNIST(root=self.data_root,
                                             train=True,
                                             download=True,
                                             transform=transform)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def sample_images(self, images, it):
        sample = images[:6]
        grid = torchvision.utils.make_grid(sample)

        self.experiment.add_image("generated_images", grid, it)


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/tmp/data")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--z_ch", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=256)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    experiment = test_tube.Experiment(save_dir=args.log_dir)
    model = GAN(z_ch=args.z_ch,
                image_shape=(1, 28, 28),
                lr=args.lr,
                b1=args.b1,
                b2=args.b2,
                data_root=args.data_root,
                batch_size=args.batch_size,
                device=device)

    trainer = pl.Trainer(experiment=experiment, max_nb_epochs=args.n_epochs)
    trainer.fit(model)


if __name__ == "__main__":
    main()