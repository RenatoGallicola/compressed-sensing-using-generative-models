"""Check the numbers the write-ups take from the code, not from the results.

``test_reported_numbers.py`` compares the prose against ``benchmark.csv``. It
therefore reaches the results and nothing else, which leaves three large classes
of number with no guard at all: the layer sizes and parameter counts in the
architecture tables, the protocol constants the write-ups quote from the
scripts, and the ratios that summarise other numbers. A mutation audit over
every hand-written number showed these were the bulk of what nothing protected.

They are checked here against their source: the models are built and counted,
the parsers are read for their defaults, and every ratio is divided again.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from helpers import script_defaults

from csgm.config import ROOT_DIR

REPORT = ROOT_DIR / "docs" / "report"
PRIORS = ["fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]


def _table_numbers(path, caption_marker):
    """The numbers of the table whose caption starts with ``caption_marker``."""
    text = (ROOT_DIR / path).read_text(encoding="utf-8")
    end = text.index(caption_marker)
    start = text.rindex(r"\begin{table}", 0, end)
    block = text[start:end]
    return [
        int(m.group(0).replace("{,}", "")) for m in re.finditer(r"\d{1,3}(?:\{,\}\d{3})+", block)
    ]


@pytest.fixture(scope="module")
def benchmark():
    path = ROOT_DIR / "results" / "benchmark.csv"
    if not path.exists():
        pytest.skip("benchmark.csv is not present")
    return pd.read_csv(path)


def test_the_architecture_tables_count_the_models_that_are_built():
    """Each table in the report must be the network the code produces.

    The tables are read by anyone reimplementing this, and nothing else in the
    suite looks at them, so a layer edited in ``csgm.models`` would leave them
    describing a network that no longer exists.
    """
    from csgm.models import (
        build_decoder,
        build_discriminator,
        build_encoder,
        build_fc_decoder,
        build_generator,
    )

    cases = [
        (build_generator(20), "docs/report/dcgan.tex", "caption{Structure of the DCGAN generator"),
        (
            build_discriminator(),
            "docs/report/dcgan.tex",
            "caption{Structure of the DCGAN discriminator",
        ),
        (build_encoder(20), "docs/report/vae.tex", "caption{Structure of the VAE encoder"),
        (build_decoder(20), "docs/report/vae.tex", "caption{Structure of the VAE decoder"),
    ]
    for model, path, marker in cases:
        stated = _table_numbers(path, marker)
        assert model.count_params() in stated, (
            f"{path}: the table totals {stated}, the built model has {model.count_params():,}"
        )
        for layer in model.layers:
            counted = layer.count_params()
            if counted >= 1000:
                assert counted in stated, (
                    f"{path}: layer {layer.name} has {counted:,} parameters, absent from the table"
                )

    # Both captions give the total for the other latent dimension in prose.
    dcgan = (REPORT / "dcgan.tex").read_text(encoding="utf-8")
    vae = (REPORT / "vae.tex").read_text(encoding="utf-8")
    assert f"{build_generator(30).count_params():,}".replace(",", "{,}") in dcgan
    assert f"{build_decoder(30).count_params():,}".replace(",", "{,}") in vae
    assert build_fc_decoder(20).count_params() > build_decoder(20).count_params(), (
        "the write-ups say our convolutional decoder is the smaller of the two"
    )


def test_the_protocol_the_write_ups_quote_is_the_scripts_defaults():
    """Running the documented commands has to reproduce the documented protocol.

    The commands in the write-ups are given with no flags, so the defaults are
    the protocol, and this pins them. It is the other half of
    ``test_prose_quotes_the_protocol.py``, which reads the same quantities out of
    the prose: that one catches a document drifting from the code, this one
    catches the code drifting from what both were written to describe.
    """
    benchmark = script_defaults("run_benchmark")
    vae = script_defaults("train_vae")
    dcgan = script_defaults("train_dcgan")
    select = script_defaults("select_dcgan")

    assert benchmark["--steps"] == 1000
    assert benchmark["--restarts"] == 10
    assert benchmark["--learning-rate"] == 0.01
    assert benchmark["--l2-penalty"] == 0.1
    assert benchmark["--noise-norm"] == 0.1
    assert benchmark["--n-images"] == 10

    assert vae["--epochs"] == 100
    assert vae["--batch-size"] == 100
    assert vae["--learning-rate"] == 0.001
    assert vae["--kl-warmup-epochs"] == 10
    assert vae["--validation-fraction"] == 0.1

    assert dcgan["--epochs"] == 50
    assert dcgan["--batch-size"] == 64
    assert dcgan["--learning-rate"] == 0.0002
    assert dcgan["--beta-1"] == 0.5
    assert dcgan["--generator-updates"] == 2
    assert dcgan["--checkpoint-every"] == 5
    assert dcgan["--checkpoint-from"] == 20
    assert dcgan["--validation-fraction"] == 0.1

    assert (select["--n-images"], select["--restarts"], select["--steps"]) == (32, 3, 500)

    saved = sorted(
        int(re.search(r"epoch(\d+)", path.name).group(1))
        for path in (ROOT_DIR / "models" / "dcgan_checkpoints").glob("*dim20*.keras")
    )
    if saved:
        expected = list(
            range(
                dcgan["--checkpoint-from"],
                dcgan["--epochs"] + 1,
                dcgan["--checkpoint-every"],
            )
        )
        assert saved == expected, "the write-ups describe seven candidates from the twentieth epoch"
        assert len(saved) == 7


def test_every_ratio_in_the_prose_divides_out(benchmark):
    """A ratio is right only if both values behind it are, and the pair is.

    These summarise numbers from three different sources, so none of them is
    reached by a check against ``benchmark.csv`` alone.
    """
    means = benchmark.pivot_table(index="m", columns="method", values="per_pixel_error")
    seconds = benchmark.groupby("method")["seconds_per_batch"].mean()
    floors = {k: means[k][means.index >= 300].mean() for k in PRIORS}

    assert round(400 / 75, 1) == 5.3
    assert round(400 / 300, 1) == 1.3
    assert round(means.loc[25, "lasso-dct"] / means.loc[25, PRIORS].min(), 1) == 5.1
    assert round(means.loc[25, "lasso"] / means.loc[25, PRIORS].min(), 1) == 6.1
    assert round(max(floors.values()) / min(floors.values()), 1) == 1.7
    assert round(seconds[PRIORS].max() / seconds[PRIORS].min()) == 93

    for latent_dim, spread in [(20, 1.28), (30, 1.42)]:
        record = ROOT_DIR / "models" / f"dcgan_selection_dim{latent_dim}.txt"
        if not record.exists():
            continue
        text = record.read_text(encoding="utf-8")
        scores = [float(v) for v in re.findall(r"epoch\d+\.keras: (0\.\d+)", text)]
        assert round(max(scores) / min(scores), 2) == spread
        last = float(re.search(r"last epoch[^\n]*at (0\.\d+)", text).group(1))
        assert round(100 * (last / min(scores) - 1)) == 11, (
            "the write-ups say the last epoch was about 11 per cent worse"
        )


def test_the_seed_spread_is_the_one_its_own_table_gives():
    """The factor 3.5 is quoted in five documents and rests on one table.

    That table is the only record of the experiment behind it: the runs were not
    kept, so nothing can recompute it from an artefact. What can be checked is
    that the ratio the documents repeat is the ratio the table gives, which is
    what would break if a cell or a quotation were edited on its own.
    """
    selection = (ROOT_DIR / "docs" / "model_selection.md").read_text(encoding="utf-8")
    header = selection.index("| latent dim | seed")
    block = selection[header : selection.index("\n\n", header)]
    rows = re.findall(r"^\| (20|30) \|([^\n]*)\|\s*$", block, re.M)
    assert len(rows) == 2, "the seed table is no longer where this reads it"

    spreads = {}
    for latent_dim, body in rows:
        scores = [float(v) for v in re.findall(r"0\.\d+", body)]
        assert len(scores) >= 3, f"k={latent_dim} lost its seeds"
        spreads[latent_dim] = max(scores) / min(scores)

    assert round(spreads["20"], 1) == 2.4
    assert round(spreads["30"], 1) == 3.5
    quoted = round(spreads["30"], 1)

    for path in [
        ROOT_DIR / "README.md",
        REPORT / "results.tex",
        REPORT / "conclusions.tex",
        REPORT / "vae.tex",
    ]:
        text = re.sub(r"\\\w+\{([^}]*)\}", r"\1", path.read_text(encoding="utf-8"))
        text = re.sub(r"[\\{}`*]", "", text)
        text = re.sub(r"\s+", " ", text)
        stated = re.findall(r"(?:factor of|varied by|by a factor of) (\d\.\d)", text)
        assert str(quoted) in stated, (
            f"{path.name} does not state the seed spread of {quoted}, it states {stated}"
        )


def _values_the_artefacts_contain():
    """Every number the committed results and records hold, as the prose spells it."""
    known: set[str] = set()

    def remember(value):
        for digits in (3, 4, 5):
            known.add(f"{value:.{digits}f}")
            known.add(f"{value:.{digits}f}".rstrip("0").rstrip("."))

    # Only quantities a write-up could quote: the per-image rows themselves are
    # left out, since admitting all seven hundred would admit almost any digit.
    sweeps = [
        "benchmark.csv",
        "unregularised/benchmark.csv",
        "dcgan_paper_penalty/benchmark.csv",
    ]
    for name in sweeps:
        path = ROOT_DIR / "results" / name
        if not path.exists():
            continue
        table = pd.read_csv(path)
        grouped = table.groupby(["method", "m"])["per_pixel_error"]
        for summary in (grouped.mean(), grouped.median(), grouped.min(), grouped.max()):
            for value in summary:
                remember(float(value))
        means = table.pivot_table(index="m", columns="method", values="per_pixel_error")
        floors = {c: float(means[c][means.index >= 300].mean()) for c in means.columns}
        for value in floors.values():
            remember(value)
        for column in means.columns:
            remember(float(means[column][means.index >= 400].mean()))
        # The write-ups compare floors as well as quoting them.
        for one in floors.values():
            for other in floors.values():
                if one > other:
                    remember(one - other)

    sweep = ROOT_DIR / "results" / "lambda_sweep.csv"
    if sweep.exists():
        frame = pd.read_csv(sweep)
        for column in ("per_pixel_error", "latent_norm"):
            for value in frame.groupby(["method", "l2_penalty", "m"])[column].mean():
                remember(float(value))

    significance = ROOT_DIR / "results" / "significance.csv"
    if significance.exists():
        frame = pd.read_csv(significance)
        for column in ("p_wilcoxon", "p_holm"):
            for value in frame[column].unique():
                remember(float(value))

    tuning = ROOT_DIR / "results" / "lasso_tuning.csv"
    if tuning.exists():
        frame = pd.read_csv(tuning)
        for value in frame.groupby(["basis", "m"])["error"].min():
            remember(float(value))
        for value in frame["alpha"].unique():
            remember(float(value))

    archive_path = ROOT_DIR / "results" / "reconstructions.npz"
    if archive_path.exists():
        with np.load(archive_path) as archive:
            remember(float((archive["ground_truth"] ** 2).mean()))

    # The selection records, and the seed experiment whose only record is its
    # own table: those runs were not kept, so the table is the source and the
    # quotations of it elsewhere are what this checks.
    for record in sorted((ROOT_DIR / "models").glob("*selection*.txt")):
        for value in re.findall(r"\d+\.\d+", record.read_text(encoding="utf-8")):
            remember(float(value))
    selection = (ROOT_DIR / "docs" / "model_selection.md").read_text(encoding="utf-8")
    for line in selection.splitlines():
        if line.startswith("|"):
            for value in re.findall(r"\d+\.\d+", line):
                remember(float(value))

    # Constants of the setup rather than measurements of it.
    for constant in [0.0002, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.025, 0.975, 0.0, 1.0]:
        remember(constant)
    return known


def test_no_quoted_decimal_is_a_number_the_project_never_measured():
    """Every decimal in the write-ups must be a value some artefact contains.

    The guards above each know the sentence they check. This one needs no
    pattern: it builds the set of numbers the results hold and asks whether each
    quoted decimal is one of them.

    Its reach is worth stating plainly, since a guard that overstates itself is
    worse than none. Measured against every decimal in the write-ups, it rejects
    about three in ten single-digit corruptions: the rest land on some other
    genuine value, because seventy budget-by-method means and their medians are
    numbers of the same shape. What it does catch reliably is a figure that
    matches nothing the project ever computed, which is how it found a floor
    difference quoted as 0.0027 where the floors give 0.0026.
    """
    known = _values_the_artefacts_contain()
    if len(known) < 500:
        pytest.skip("the artefacts are not present")

    documents = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "docs" / "model_selection.md",
        ROOT_DIR / "models" / "README.md",
        ROOT_DIR / "results" / "README.md",
        *sorted(REPORT.glob("*.tex")),
        *sorted((ROOT_DIR / "notebooks").glob("*.ipynb")),
    ]

    orphans, checked = [], 0
    for path in documents:
        if path.suffix == ".ipynb":
            cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
            text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
        else:
            text = path.read_text(encoding="utf-8")
        text = re.sub(r"\s+", " ", text)
        for m in re.finditer(r"(?<![\w.])0\.\d{3,5}(?![\w.])", text):
            checked += 1
            token = m.group(0)
            if token in known or token.rstrip("0") in known:
                continue
            context = text[max(0, m.start() - 60) : m.end() + 30].strip()
            orphans.append(f"{path.name}: {token} in '{context}'")

    assert checked >= 240, f"only {checked} decimals were found to check"
    assert not orphans, "these values appear in no artefact:\n" + "\n".join(orphans)
