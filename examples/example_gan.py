import sys
sys.path.append(".")
from pytorch_lightning.trainer import Trainer
from models import GAN


def main():
    model = GAN()
    trainer = Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    main()
