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
* BEGAN: Boundary equilibrium generative adversarial networks (Berthelot et al.)
* DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al.)
* GAN: Generative Adversarial Networks (Goodfellow et al.)

# References
* Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
* Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
* Berthelot, David, Thomas Schumm, and Luke Metz. "Began: Boundary equilibrium generative adversarial networks." arXiv preprint arXiv:1703.10717 (2017).