# Pre-trained checkpoints

Keras 3 (`.keras`) archives. At recovery time only the **generators** are used:
they are the prior $G$ that `csgm.recovery.recover` inverts. The discriminators
are kept so the adversarial training can be resumed or audited.

| file | role | latent dim | trained by |
|---|---|---|---|
| `vae_decoder_dim20.keras` | generator $G$ | 20 | `train_vae.py --latent-dim 20 --seed 1` |
| `vae_decoder_dim30.keras` | generator $G$ | 30 | `train_vae.py --latent-dim 30 --seed 1` |
| `fc_vae_decoder_dim20.keras` | generator $G$ | 20 | `train_vae.py --latent-dim 20 --architecture fc --seed 1` |
| `gan_gen_dim20.keras` | generator $G$ | 20 | `train_dcgan.py --latent-dim 20 --epochs 50 --seed 1`, then `select_dcgan.py` |
| `gan_gen_dim30.keras` | generator $G$ | 30 | `train_dcgan.py --latent-dim 30 --epochs 50 --seed 1`, then `select_dcgan.py` |
| `vae_encoder_dim*.keras`, `fc_vae_encoder_dim20.keras` | encoder $q_\phi(z \mid x)$ | as above | saved with the decoder |
| `gan_disc_dim20.keras`, `gan_disc_dim30.keras` | discriminator $D$ | n/a | saved with the generator |

The `fc_` files are the fully connected `784-500-500-20` architecture of Bora et
al., kept so that the reference setup can be reproduced rather than
approximated. The others are the convolutional networks described in the report.

Which seed each VAE checkpoint comes from, and why, is recorded in
[`docs/model_selection.md`](../docs/model_selection.md).

Each GAN generator is the epoch `select_dcgan.py` chose out of the seven saved
during its run, on the lowest representation error over images held out of
training. Which epoch won, and what every candidate scored, is in
`dcgan_selection_dim20.txt` and `dcgan_selection_dim30.txt` beside them. The
per-epoch generators themselves are not committed, since those two files record
everything the choice rested on.

The VAE checkpoints are chosen the other way round, on validation loss rather
than on representation error, because a variational autoencoder has one to be
chosen on. Which seed won for each configuration, and what it scored, is in
`vae_selection.txt`.

All five generators map $z \in \mathbb{R}^k$ to a $28 \times 28 \times 1$ image
with sigmoid outputs in $[0, 1]$, and were trained on MNIST scaled to the same
range.

```python
from csgm.models import load_generator

G = load_generator("dcgan", 20)  # or load_generator("vae", 30)
```

## File format

The archives are in the Keras 3 format and load with the TensorFlow version
pinned in `requirements.txt`. Note that Keras dispatches on the file extension:
a `.keras` file holding HDF5 content, which is what Keras 2 used to write, is
rejected by Keras 3 with a confusing "not a zip file" error. If you hit that,
rebuild the architecture with `csgm.models` and load the weights from a copy of
the file renamed to `.h5`.
