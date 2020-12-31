# PyTorch Lightning GANs

[![DOI](https://zenodo.org/badge/202523756.svg)](https://zenodo.org/badge/latestdoi/202523756)

Collection of PyTorch Lightning implementations of Generative Adversarial Network varieties presented in research papers.

## Installation

```bash
$ pip install -r requirements.txt
```

## Example
The minimum code for training GAN is as follows:

```python
from pytorch_lightning.trainer import Trainer
from models import GAN


model = GAN()
trainer = Trainer()
trainer.fit(model)
```

or you can run the following command:

```bash
$ python models/gan.py --gpus=2
```

## Implementations
* BEGAN: Boundary equilibrium generative adversarial networks (Berthelot et al.)
* DCGAN: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al.)
* GAN: Generative Adversarial Networks (Goodfellow et al.)
* LSGAN: Least squares generative adversarial networks (Mao et al.)
* WGAN: Wasserstein GAN (Arjovsky et al.)
* WGAN-GP: Improved Training of Wasserstein GANs (Gulrajani et al.)

## Acknowledgements
This repository is highly inspired by [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) repository.

## References
* Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
* Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
* Berthelot, David, Thomas Schumm, and Luke Metz. "Began: Boundary equilibrium generative adversarial networks." arXiv preprint arXiv:1703.10717 (2017).
* Mao, Xudong, et al. "Least squares generative adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.
* Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein generative adversarial networks." Proceedings of the 34th International Conference on Machine Learning-Volume 70. 2017.
* Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." Advances in neural information processing systems. 2017.

## Citation

```bibtex
@software{https://doi.org/10.5281/zenodo.4404867,
  doi = {10.5281/ZENODO.4404867},
  url = {https://zenodo.org/record/4404867},
  author = {Masanari Kimura},
  title = {pytorch-lightning-gans},
  publisher = {Zenodo},
  year = {2020},
  copyright = {Open Access}
}
```
