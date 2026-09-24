# Compressed Sensing using Generative Models

[![CI](https://github.com/RenatoGallicola/compressed-sensing-using-generative-models/actions/workflows/suite.yml/badge.svg)](https://github.com/RenatoGallicola/compressed-sensing-using-generative-models/actions/workflows/suite.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A1703.03208-b31b1b.svg)](https://arxiv.org/abs/1703.03208)

Reproduction and extension of **Bora, Jalal, Price & Dimakis, *Compressed Sensing
using Generative Models* (ICML 2017)** on MNIST: a DCGAN generator and a VAE
decoder are used as *learned priors* to reconstruct images from a handful of
random linear measurements, and benchmarked against Lasso in two sparsifying
bases as the classical sparsity prior.

> Course project for **Numerical Analysis for Machine Learning**, MSc in Computer
> Science and Engineering, Politecnico di Milano.
> The write-up is in [`docs/report/`](docs/report), one LaTeX file per section,
> building with any pdfTeX toolchain.

<p align="center">
  <img src="results/figures/error_vs_measurements.png" width="88%"
       alt="Reconstruction error against the number of measurements, for two Lasso baselines and five generative priors">
</p>

**A VAE prior recovers an MNIST digit from 75 random measurements about as
accurately as Lasso does from 400, a 5.3x saving, and the figure is the same
whichever of the two Lasso baselines is taken as the reference.** From 500
measurements the ranking reverses and Lasso wins outright, because a generative
prior can only ever return an image its generator is able to produce. Both
effects are what Bora et al. report, and the threshold they give for the
reversal is more than 500 measurements.

---

## The problem

Compressed sensing reconstructs an unknown signal $x^{\ast} \in \mathbb{R}^n$ from far
fewer measurements than its dimension:

$$y = A x^{\ast} + \eta, \qquad A \in \mathbb{R}^{m \times n}, \quad \eta \sim \mathcal{N}(0, \sigma^2 I), \quad m \ll n.$$

With $m \ll n$ the system is underdetermined and has infinitely many solutions,
so recovery is only possible by assuming *structure*. Classical theory assumes
**sparsity in a fixed basis**, $x^{\ast} = \Psi\theta$ with few non-zero
$\theta_i$, and recovers $x^{\ast}$ with an $\ell_1$ program such as Lasso.

This project replaces that assumption with a much stronger, *learned* one: that
$x^{\ast}$ lies near the range of a trained generator $G : \mathbb{R}^k \to \mathbb{R}^n$.
Instead of "few active coefficients", the prior says "**looks like a digit**".

## The method

Recovery becomes a search in the latent space of the generator rather than in
signal space:

$$\hat z = \arg\min_{z \in \mathbb{R}^k} \; \lVert A\,G(z) - y \rVert_2^2 + \lambda \lVert z \rVert_2^2, \qquad \hat x = G(\hat z).$$

Since $G$ is a deep network the objective is non-convex, so it is minimised with
**Adam on $z$** (the generator weights stay frozen) from **several random
restarts**, keeping the restart with the smallest measurement error. The penalty term is
excluded from that ranking: it would reward a small $\lVert z \rVert$ rather
than a faithful reconstruction. The
theoretical pay-off, and the reason the approach works with so few
measurements, is that the sample complexity scales with the *latent* dimension:
if $G$ is $L$-Lipschitz, $O(k \log L)$ Gaussian measurements suffice for an
$\ell_2/\ell_2$ recovery guarantee.

Three priors are compared on MNIST. Two are our own, trained at both
$k = 20$ and $k = 30$; the third is the network of the reference paper, kept so
that our numbers can be compared with theirs directly:

| | prior | trained by | used at recovery time |
|---|---|---|---|
| **VAE** | convolutional encoder/decoder, diagonal Gaussian posterior | maximising the ELBO | the **decoder** is $G$ |
| **DCGAN** | strided conv generator + discriminator | adversarial minimax game | the **generator** is $G$ |
| **VAE, paper architecture** | fully connected $784$-$500$-$500$-$20$ | maximising the ELBO | the **decoder** is $G$ |

Both generators map $z \in \mathbb{R}^k$ to a $28 \times 28$ image through a
projection followed by transposed convolutions, but at very different scales:

| stage | VAE decoder | DCGAN generator |
|---|---|---|
| project | dense to $14 \times 14 \times 64$ | dense to $3 \times 3 \times 128$ |
| upsample | transposed conv, 32 filters, to $28 \times 28$ | transposed convs with 128, 256 and 512 filters, to $6^2$, $14^2$, $28^2$ |
| output | transposed conv, 1 filter, sigmoid | conv, 1 filter, sigmoid |
| parameters ($k=20$) | 282,177 | 2,921,473 |

The paper's decoder has 653,784 parameters, between the two. Size is only part
of the story for the cost of inverting them, as the Cost section below shows,
and it buys nothing at all in accuracy.

The measurement matrix $A$ has i.i.d. $\mathcal{N}(0, 1/m)$ entries, which makes
it an approximate isometry in expectation ($\mathbb{E}\lVert Ax \rVert^2 = \lVert x \rVert^2$),
so errors in measurement space and signal space stay on the same scale as $m$
changes.

## Results

Ten test digits, one per class, recovered by every method from the same
measurement matrices and the same noise, with the latent regulariser at the
value the reference paper recommends. Error is the squared distance to the
ground truth, per pixel; lower is better.

|   m | Lasso (pixel) | Lasso (DCT) | VAE (paper arch.) | VAE k=20 |   VAE k=30 | DCGAN k=20 | DCGAN k=30 |
|----:|--------------:|------------:|------------------:|---------:|-----------:|-----------:|-----------:|
|  10 |        0.1217 |      0.1032 |        **0.0680** |   0.0778 |     0.0755 |     0.0864 |     0.1060 |
|  25 |        0.1189 |      0.0986 |            0.0276 |   0.0288 | **0.0194** |     0.0813 |     0.0635 |
|  50 |        0.1171 |      0.0835 |        **0.0118** |   0.0155 |     0.0191 |     0.0327 |     0.0383 |
|  75 |        0.1127 |      0.0710 |        **0.0084** |   0.0088 |     0.0091 |     0.0167 |     0.0306 |
| 100 |        0.1046 |      0.0628 |            0.0083 |   0.0083 | **0.0077** |     0.0142 |     0.0147 |
| 200 |        0.0827 |      0.0325 |            0.0075 |   0.0079 | **0.0072** |     0.0125 |     0.0155 |
| 300 |        0.0408 |      0.0202 |            0.0072 |   0.0076 | **0.0071** |     0.0125 |     0.0102 |
| 400 |        0.0107 |      0.0115 |            0.0072 |   0.0076 | **0.0070** |     0.0120 |     0.0088 |
| 500 |    **0.0008** |      0.0064 |            0.0072 |   0.0076 |     0.0071 |     0.0117 |     0.0096 |
| 750 |    **0.0000** |      0.0006 |            0.0070 |   0.0075 |     0.0069 |     0.0116 |     0.0085 |

Full table in [`results/benchmark.csv`](results/benchmark.csv), one row per
method, budget and image, each carrying the hash of the checkpoint and the
revision that produced it. The same sweep without the regulariser is in
[`results/unregularised/`](results/unregularised).

**Two baselines, not one.** Bora et al. run Lasso on MNIST in the *pixel* basis,
since digits are mostly background and therefore already sparse there. The DCT
basis is what the same authors use for natural images. The two behave very
differently here: the DCT basis is the better of the two from 10 to 300
measurements and the pixel basis from 400 up, where it becomes almost exact.
Reporting only one would misrepresent how strong the classical method is. The
shrinkage was swept over eight values spanning seven orders of magnitude, for
each basis and separately at each budget, and set to its best value on a draw of
images the benchmark never scores. Reconstructions are clipped to `[0, 1]`, which
also helps the baseline.

**What a blank image scores.** Predicting all zeros gives 0.1178 per pixel on
these digits. The pixel baseline is at or above that level at 10 and 25
measurements, so at those budgets it is not reconstructing anything and a ratio
against it means very little. This is a property of the baseline the paper
chose for MNIST, not a defect of the implementation, and the number is quoted so
that the low-budget comparisons can be read for what they are.

### Three regimes

**Scarce measurements.** Every VAE prior beats both baselines up to 400
measurements, and so does the DCGAN at `k=30` from 25 up, the DCT baseline
edging it at 10. The DCGAN at `k=20` beats both from 10 to 300. At 25 measurements the
best prior is 5.1x more accurate than the DCT baseline, at a budget where
neither baseline returns anything recognisable as a digit. The ratio against the
pixel baseline is 6.1x, but that one is at the blank-image level here, so the DCT
figure is the meaningful one.

**Sample efficiency.** Each of the three VAE priors reaches the error the Lasso
baseline achieves with 400 measurements using only **75**, a **5.3x** saving.
The figure is the same against either baseline, so it does not depend on which
one is taken as the reference; what does depend on that choice is how many
individual digits the prior beats there, 8 or 9 of 10 against the DCT baseline
and 1 of 10 against the pixel one, for the reason given above. The DCGAN at
`k=30` needs 300, a 1.3x saving; at `k=20` it never reaches the level within the
sweep. Bora et al. report 5 to 10x, so the reproduction lands at the bottom of
that interval, and only the VAE priors land inside it at all. Details in
[`results/sample_efficiency.md`](results/sample_efficiency.md).

**Abundant measurements, from 500 up.** Lasso in the pixel basis overtakes every
learned prior and keeps improving, reaching an error below 1e-4 at 750
measurements against 0.0069 for the best generative model. Once the budget
approaches the 784 dimensions of the signal, sparsity in pixel space recovers a
digit almost exactly while the generative prior stays put. Nothing is wrong with
the optimisation: the generative curves are flat because the reconstruction is
confined to the range of the generator, and the distance from a real digit to
that range does not depend on how many measurements are taken. Averaged over the
budgets from 300 up, that floor is 0.0070 for our convolutional VAE at `k=30`,
0.0072 for the paper architecture and 0.0076 for our `k=20`, against 0.0093 for
the DCGAN at `k=30` and 0.0119 at `k=20`. The five floors span a factor of 1.7,
so the choice of generator matters, but far less than the gap to the baselines at
low budgets.

<p align="center">
  <img src="results/figures/reconstruction_grid.png" width="95%"
       alt="One digit reconstructed by every method at every measurement budget">
</p>

### What the comparisons survive

Every method sees the same images and the same matrices, so the comparisons are
paired and tested as such: a Wilcoxon signed-rank test on the ten per-image
differences, corrected across the ten budgets by the Holm procedure. The tables
are in [`results/significance.md`](results/significance.md), produced by
[`scripts/run_stats.py`](scripts/run_stats.py).

**Generative priors beat sparse recovery, and then lose to it.** At 750
measurements all ten prior-and-baseline pairs put the baseline ahead, with the
difference significant in every one. In the other direction the picture is
narrower than the means suggest: 50, 75, 100 and 300 are the only budgets at
which every prior beats every baseline significantly at once, because the two
DCGAN columns drop out elsewhere. Each VAE prior on its own is significantly
better than both baselines from 25 to 300. The largest corrected p-value among these is 0.0488, and
it belongs to a DCGAN column; every claim involving only the VAEs is at 0.0195,
which is the smallest value attainable, since with ten paired samples Wilcoxon
bottoms out at 0.00195 and the correction across ten budgets multiplies that by
ten.

The correction is applied within each pair of methods across the ten budgets,
which is the family a claim like "significant from 75 up" spans. That choice is
load bearing rather than a formality: the table holds 200 tests in 20 such
families, and correcting across all 200 would put the floor at 0.39, under which
nothing here would be significant.

These tests are conditional on one measurement matrix per budget. The pairing
across images is real, but ten images under a single draw of `A` is not ten
independent sensing realisations, so the intervals and the p-values describe
image-to-image variation only.

**The three VAEs are indistinguishable from each other.** Not one of the
comparisons between the paper's fully connected architecture, our convolutional
`k=20` and our convolutional `k=30` reaches significance at any budget; every
corrected p-value is 1.0. Their floors differ by 0.0005, which ten images cannot
separate. The honest reading is that neither the architecture nor the latent
dimension matters here at this sample size, and any ranking between them read off
the table would be noise.

**The VAE family beats the DCGAN at `k=20` everywhere, and at `k=30` only when
measurements are scarce.** Against `dcgan-20` every VAE is significantly more
accurate at nine of the ten budgets. Against `dcgan-30` the advantage is
significant only while measurements are scarce, and at how many budgets depends
on the VAE: four for `k=30`, three for the paper architecture, one for `k=20`.
Elsewhere the difference is real but too small for ten
images to establish: at 750 measurements `dcgan-30` sits at 0.0085 against 0.0069
for the best VAE. Every VAE has a lower mean error than every DCGAN at every
budget in the sweep; what changes with `dcgan-30` is only whether ten images can
prove it.

**The latent penalty is not what separates the families.** The table uses
$\lambda = 0.1$, the value Bora et al. report for their MNIST VAE, while the only
value they give for a DCGAN is 0.001, on celebA at a different latent dimension
and pixel range. Running both DCGANs at 0.001 as well
([`results/dcgan_paper_penalty/`](results/dcgan_paper_penalty)) leaves the floor
at `k=20` essentially unchanged, 0.0116 against 0.0119, and makes `k=30`
**worse**, 0.0104 against 0.0093. The penalty the table uses is therefore not
handicapping the DCGANs; for one of them it is the better setting.

### The latent regulariser

<p align="center">
  <img src="results/figures/regularisation_comparison.png" width="88%"
       alt="Recovery with and without the latent regularisation term">
</p>

The term helps where the measurements underdetermine the latent code and hurts
where they do not. At 10 measurements it improves the best model from 0.0782 to
0.0680; at 750 it costs, moving 0.0054 to 0.0069. The sweep makes the mechanism
explicit: the norm of the recovered code falls monotonically with lambda, from
9.95 to 2.67, and the value that minimises the error is **not fixed**. It is 1
for all three priors at 10 measurements and 0 for all of them from 200 up; in
between they disagree. The single value the paper recommends is a compromise
across regimes rather than an optimum at any one of them, and the benchmark uses
it everywhere, which means the generative side is running a knowingly suboptimal
regulariser at every budget. The sweep runs on its own draw of images, disjoint
from the ones the benchmark scores, so reading a preferred penalty off it is not
selection on the evaluation set. It covers the three VAE priors only, since
repeating it for the DCGANs would cost hours.

### Cost

Recovering ten images at one budget, ten restarts and a thousand Adam steps:
about 1.5 s with either Lasso baseline, 6.6 s with the paper's decoder, 15 s and
19 s with our convolutional decoders, and about 600 s with a DCGAN generator, a
**93x** gap between the cheapest and the dearest learned prior. Parameter counts
do not explain that: the DCGAN generator is only 4.5x larger than the paper's
decoder, and our convolutional decoder is *smaller* than it yet twice as slow.
What the cost tracks is arithmetic per forward pass, and the DCGAN applies
transposed convolutions with 256 and 512 channels at nearly full resolution. The
two most expensive priors are also the two least accurate, at every budget, so on
this dataset there is no trade-off to arbitrate.

### Training variance is part of the result

Training the same VAE with different seeds produced generators whose quality
varied by a **factor of 3.5**, comparable to the entire spread between the priors
being compared. The cause was partial posterior collapse, and a KL warm-up
addressed it: every model improved, the best representation error at `k=30`
falling from 0.0103 to 0.0062. Across the two seeds of each configuration the
remaining spread is a factor of 2.5, 1.3 and 1.2. Those two sets of figures are
not the same measurement, being taken on different images and over a different
number of seeds, so they are reported side by side rather than as one ratio
shrinking. The selection procedure, fixed before the runs and unchanged
afterwards, is in [`docs/model_selection.md`](docs/model_selection.md).

## Repository layout

```
├── src/csgm/                  installable package -- all the logic lives here
│   ├── measurements.py          random Gaussian sensing operator
│   ├── recovery.py              latent-space optimisation (the core algorithm)
│   ├── baselines.py             Lasso in the pixel or the DCT basis
│   ├── metrics.py               per-pixel L2 error, PSNR
│   ├── data.py, viz.py          MNIST loading, plotting helpers
│   └── models/                  VAE, DCGAN, checkpoint loading
├── scripts/                   command-line entry points
│   ├── train_vae.py             train the VAE, save the decoder
│   ├── train_dcgan.py           train the DCGAN, save the generator
│   ├── select_dcgan.py          picks a generator on held-out data
│   ├── run_benchmark.py         the full sweep -> results/benchmark.csv
│   ├── run_lambda_sweep.py      sensitivity to the latent regulariser
│   ├── tune_lasso.py            picks the baseline's shrinkage, in both bases
│   ├── run_stats.py             paired significance tests -> significance.md
│   └── make_figures.py          csv -> figures and summary tables
├── notebooks/                 narrated walkthrough (01 VAE, 02 DCGAN, 03 Lasso, 04 recovery)
├── models/                    pre-trained checkpoints (k = 20 and k = 30)
├── results/                   benchmark table, summary tables and figures
├── docs/
│   ├── model_selection.md       how each checkpoint was chosen
│   └── report/                  LaTeX source of the write-up
└── tests/                     pytest suite covering the package
```

## Getting started

```bash
git clone https://github.com/RenatoGallicola/compressed-sensing-using-generative-models.git
cd compressed-sensing-using-generative-models

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q

pre-commit install                                     # optional: runs ruff before each commit
```

> **NumPy is pinned below 2.0.** The TensorFlow 2.16/2.17 wheels are built
> against the NumPy 1.x ABI and fail to import otherwise.

Reconstruct a digit from 100 measurements, 13% of its 784 pixels:

```python
from csgm import gaussian_measurement_matrix, load_mnist, measure, per_pixel_l2, recover
from csgm.config import NOISE_SEED_OFFSET
from csgm.models import load_generator

(_, _), (x_test, _) = load_mnist(flatten=True)
x_star = x_test[0]

G = load_generator("vae", latent_dim=20)
A = gaussian_measurement_matrix(m=100, n=784, seed=0)
# The noise gets its own stream: seeding it like A would make it a rescaled
# copy of A's first rows rather than an independent draw.
y = measure(x_star, A, noise_std=0.01, seed=NOISE_SEED_OFFSET)

result = recover(G, y, A, latent_dim=20)
print(f"per-pixel error: {per_pixel_l2(result.x_hat, x_star)[0]:.4f}")
```

## Reproducing the results

```bash
# 1. (optional) retrain the priors -- pre-trained checkpoints ship in models/
python scripts/train_vae.py   --latent-dim 20 --seed 1          # convolutional
python scripts/train_vae.py   --latent-dim 20 --architecture fc # paper's network
python scripts/train_dcgan.py --latent-dim 20 --epochs 50
python scripts/select_dcgan.py --latent-dim 20  # a GAN has no validation loss

# 2. sweep every prior over every measurement budget  (~3.5 h on CPU)
python scripts/tune_lasso.py          # pick the baseline's shrinkage first
python scripts/run_benchmark.py --n-images 10 --steps 1000 --restarts 10 --l2-penalty 0.1

# 3. the same sweep with the regulariser removed  (~4.6 h)
python scripts/run_benchmark.py --l2-penalty 0 --output-dir results/unregularised

# 4. how much the latent regulariser matters (VAE decoders only, ~30 min)
python scripts/run_lambda_sweep.py

# 5. the paired significance tests, and the figures and summary tables
python scripts/run_stats.py
python scripts/make_figures.py
```

`run_benchmark.py` writes `results/benchmark.csv`, one row per
(method, $m$, image), so the analysis can be redone without re-running the sweep.
Every random draw (measurement matrices, noise, latent initialisations and the
choice of test images) is derived from a single `--seed`.

It also writes `benchmark_meta.json` recording the protocol. Passing `--merge`
re-runs a subset of the methods and folds them into an existing table, which
refuses to proceed unless the recorded protocol matches, so results from two
runs cannot be pooled unless they are comparable.

The notebooks are committed **with their outputs**, so every plot is readable
straight from GitHub without installing anything. They were executed top to
bottom against the checkpoints and the benchmark table in this repository.

## Method notes

A few implementation choices determine what the numbers mean, so they are worth
stating explicitly.

**The measurement matrix is scaled by $1/\sqrt{m}$.** Entries are drawn i.i.d.
from $\mathcal{N}(0, 1/m)$, which makes $A$ an approximate isometry in
expectation. Measurement space and signal space then stay on the same scale as
the budget changes, and so does the effective noise level.

**Quality is measured against the ground truth.** Every curve reports
$\lVert \hat x - x^{\ast} \rVert^2 / n$, the metric used by Bora et al. The
measurement residual $\lVert A G(\hat z) - y \rVert$ is the quantity the
optimiser minimises, and it falls as $m$ shrinks simply because fewer
constraints remain to satisfy, so it is recorded as an optimisation diagnostic
and never as a score.

**Latent codes are initialised from $\mathcal{N}(0, I)$**, the prior the
generators were trained under. Starting much closer to the origin biases the
search towards the blurry centre of the latent space.

**The latent regulariser uses $\lambda = 0.1$**, the value Bora et al. report as
best on MNIST. `results/unregularised/` holds the same sweep with $\lambda = 0$,
so both variants can be compared as they are in the reference paper.

**Restarts are ranked on the measurement error alone**, never on the penalised
objective. Ranking on the objective would reward a small $\lVert z \rVert$
rather than a faithful reconstruction, and the measurement error is the only
criterion available when the ground truth is unknown.

**The VAEs are trained with a KL warm-up.** The weight of the KL term is ramped
from zero to one over the first ten epochs. Without it, training frequently ends
in partial posterior collapse and the quality of the resulting prior varies by a
factor of 3.5 between random seeds; see
[`docs/model_selection.md`](docs/model_selection.md).

**The baseline is given its best configuration.** Lasso is reported in two
bases, the pixel basis the reference paper uses for MNIST and the DCT basis it
uses for natural images, with the shrinkage swept per basis and set to its
minimum-error value, and with the reconstruction clipped to `[0, 1]`. A
comparison against a badly tuned baseline would say nothing.

**Two deliberate departures from the reference paper.** It treats MNIST as
binary, calling its input a vectorised binary image with pixel values of 0 or 1
and stating that no pre-processing was performed, while every model here is
trained and evaluated on the grayscale values scaled to `[0, 1]`, so absolute
error values are not directly comparable with the ones it prints. And its experimental section specifies measurement entries with
standard deviation `1/m` where its own theorems use `N(0, 1/m)`; we follow the
theorems, since only that scaling makes `A` an approximate isometry.

**Every method sees the same inputs.** At each budget the same measurement
matrix, the same noise draw and the same ten stratified test digits, one per
class, are handed to Lasso and to each generative prior, and the curves are
averages over those ten images. Following Bora et al., the noise vector has a
fixed expected norm of 0.1 at every budget, so the per-component standard
deviation is $0.1/\sqrt{m}$.

**The noise is drawn independently of the measurement matrix.** The two come
from separate random streams. Seeding them alike would make $\eta$ a rescaled
copy of the first rows of $A$ rather than an independent draw, which is not the
model the recovery guarantees are stated for, even though the marginal law of
$\eta$ would look correct.

**No hyper-parameter and no checkpoint is chosen on the images that are
scored.** The Lasso shrinkage and the sweep over the latent penalty each run on
their own draw of images, matrices and noise, disjoint from the benchmark's; the
generator checkpoints are chosen on held-out data that is never part of the test
split. The latent penalty itself is not tuned here at all, being taken from the
reference paper. One design decision does not meet that standard and is recorded
rather than glossed: the KL warm-up was adopted after observing how much seed
variance showed up on test digits. It changed how the generators are trained, not
which checkpoint was kept, and the reasoning is in
[`docs/model_selection.md`](docs/model_selection.md).

### Limitations

- **Representation error sets the floor.** Recovery can only return an image the
  generator can produce, so beyond a few hundred measurements the error stops
  improving no matter how much optimisation is thrown at it. A sparsity prior
  has no such ceiling.
- **MNIST is easy.** Digits are a low-dimensional, near-binary manifold; the
  gap over Lasso would narrow on richer datasets.
- **Recovery is expensive.** Each reconstruction runs 1000 gradient steps
  by 10 restarts through the generator, orders of magnitude slower than a
  single convex solve.
- **Ten test images.** Enough to separate the generative priors from the
  baselines across the middle of the range, and not enough to separate the three
  VAEs from each other anywhere: every corrected p-value between them is 1.0.
  Which comparisons hold at which budget is stated above rather than averaged
  over.
- **The two model families receive different selection budgets.** Every
  generator is trained on the same 54,000 images with the test split untouched,
  and every one is chosen on held-out data. But the VAEs are the better of two
  seeds, while a DCGAN run costs about fifteen hours of CPU against forty minutes
  for a VAE, so each DCGAN is a single seed whose best epoch is kept. Their
  columns are therefore one draw from a distribution this project has shown to be
  wide: the two DCGANs differ from each other by five times what separates the
  three VAEs, 0.0026 against 0.0005 in the error floor. The
  criteria also differ, on the ELBO for the VAE and on representation error for
  the DCGAN, and the size of that difference is measured in
  [`docs/model_selection.md`](docs/model_selection.md).
- **Checkpoint selection does real work for the DCGAN.** Across the seven saved
  epochs of each run the representation error spans a factor of 1.3 at `k=20` and
  1.4 at `k=30`, it does not fall monotonically with the epoch, and in both runs
  the final epoch was about 11 per cent worse than the one the rule chose. A
  DCGAN column produced by simply stopping the clock would have been noticeably
  weaker, which is worth knowing when comparing these numbers with a GAN result
  reported without a selection rule.
- **The error bars describe image-to-image spread.** They are percentile
  bootstrap intervals for the mean, which stay inside the range a squared error
  can take; a symmetric interval falls below zero at three of the seventy
  points, since ten per-image errors are far from normal. One measurement matrix
  is drawn per budget, so the intervals say nothing about how much the curves
  would move under a different draw of `A`.

## References

1. A. Bora, A. Jalal, E. Price, A. G. Dimakis. *Compressed Sensing using
   Generative Models*. ICML 2017. [arXiv:1703.03208](https://arxiv.org/abs/1703.03208)
2. D. P. Kingma, M. Welling. *Auto-Encoding Variational Bayes*. ICLR 2014.
   [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
3. A. Radford, L. Metz, S. Chintala. *Unsupervised Representation Learning with
   Deep Convolutional Generative Adversarial Networks*. ICLR 2016.
   [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
4. I. Goodfellow et al. *Generative Adversarial Networks*. NeurIPS 2014.
   [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
5. E. J. Candès, J. Romberg, T. Tao. *Robust Uncertainty Principles: Exact Signal
   Reconstruction from Highly Incomplete Frequency Information*. IEEE Trans.
   Inf. Theory, 2006.

## Authors

**Renato Gallicola** and **Matteo Forlivesi**, Politecnico di Milano.

Released under the [MIT License](LICENSE). MNIST is distributed under the
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.
