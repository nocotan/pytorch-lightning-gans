PyTorch Lightning GANs

This repository is highly inspired by [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) repository.

# Example
The minimum code for training GAN is as follows:

```python
from pytorch_lightning.trainer import Trainer
from models import GAN


model = GAN()
trainer = Trainer()
trainer.fit(model)
```

# Implementations
* GAN (Goodfellow et al.)
* DCGAN (Radford et al.)

# References
* Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
* Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).