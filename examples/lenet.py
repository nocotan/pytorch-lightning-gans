from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
import test_tube


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, inputs):
        h = self.conv(inputs)
        h = h.view(inputs.size(0), -1)
        out = self.fc(h)

        return out


class LeNetModel(pl.LightningModule):
    def __init__(self, lr=0.001, b1=0.5, b2=0.999,
                 data_root="/tmp/data", batch_size=64, device=torch.device("cuda")):
        super(LeNetModel, self).__init__()

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.data_root = data_root
        self.batch_size = batch_size
        self.device = device

        self.network = LeNet().to(device)

    def forward(self, inputs):
        return self.model(inputs)

    def loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.network(inputs)
        loss = self.loss(outputs, targets)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        return [optimizer], []

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
    model = LeNetModel(lr=args.lr,
                       b1=args.b1,
                       b2=args.b2,
                       data_root=args.data_root,
                       batch_size=args.batch_size,
                       device=device)

    trainer = pl.Trainer(experiment=experiment, max_nb_epochs=args.n_epochs)
    trainer.fit(model)


if __name__ == "__main__":
    main()
