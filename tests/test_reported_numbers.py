"""Check that the numbers in the write-ups still match the benchmark.

The report, the README and the notebooks all quote figures that come
from ``results/benchmark.csv``. Nothing stops those from drifting apart when the
experiment is re-run, and a stale number in a report is worse than no number, so
they are verified here rather than by proofreading.

These tests read the committed results; they do not run the experiment.
"""

from __future__ import annotations

import json
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from csgm.config import RESULTS_DIR, ROOT_DIR

REPORT = ROOT_DIR / "docs" / "report"
PRIORS = ["fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]
BASELINES = ["lasso", "lasso-dct"]
REFERENCE, REFERENCE_M = "lasso-dct", 400
#: The names scripts/make_figures.py prints for each method.
LABELS = {
    "fcvae-20": "VAE, paper architecture, k=20",
    "vae-20": "VAE, k=20",
    "vae-30": "VAE, k=30",
    "dcgan-20": "DCGAN, k=20",
    "dcgan-30": "DCGAN, k=30",
}


@pytest.fixture(scope="module")
def benchmark():
    path = RESULTS_DIR / "benchmark.csv"
    if not path.exists():
        pytest.skip("benchmark.csv is not present")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def means(benchmark):
    return benchmark.pivot_table(index="m", columns="method", values="per_pixel_error")


def numbers_in(text):
    """Every decimal number in a piece of prose, as strings, in order."""
    return re.findall(r"\d+\.\d+", text)


def test_readme_table_matches_the_benchmark(means):
    """Every cell of the README results table is the mean it claims to be."""
    text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    body = text[text.index("|   m | Lasso (pixel)") :].splitlines()
    header = [c.strip(" *") for c in body[0].strip("|").split("|")]
    columns = dict(
        zip(
            header[1:],
            ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"],
            strict=True,
        )
    )

    checked = 0
    for line in body[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = int(cells[0])
        for name, cell in zip(header[1:], cells[1:], strict=True):
            claimed = float(cell.strip("* "))
            assert claimed == pytest.approx(means[columns[name]][m], abs=5e-5), (
                f"README row m={m}, column {name}"
            )
            checked += 1
    assert checked == 70


def test_report_table_matches_the_benchmark(means):
    """Same for the LaTeX table in the report."""
    text = (REPORT / "results.tex").read_text(encoding="utf-8")
    body = text[text.index(r"\label{tab:results}") - 3000 : text.index(r"\label{tab:results}")]
    order = ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]

    checked = 0
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        m = int(cells[0])
        values = [
            c.replace(r"\textbf{", "").replace("}", "").replace(r"\\", "").strip()
            for c in cells[1:]
        ]
        for method, value in zip(order, values, strict=True):
            assert float(value) == pytest.approx(means[method][m], abs=5e-5), (
                f"report row m={m}, column {method}"
            )
            checked += 1
    assert checked == 70


def test_bold_marks_the_best_method_in_the_report(means):
    """A bold entry must be the row minimum, and every row must have one."""
    text = (REPORT / "results.tex").read_text(encoding="utf-8")
    body = text[text.index(r"\hline") : text.index(r"\label{tab:results}")]
    order = ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]

    rows = 0
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        m = int(cells[0])
        bolded = [order[i] for i, c in enumerate(cells[1:]) if r"\textbf{" in c]
        best = means.loc[m, order].idxmin()
        assert bolded == [best], f"row m={m}: bold on {bolded}, best is {best}"
        rows += 1
    assert rows == 10


def test_sample_efficiency_table_is_reproducible(benchmark, means):
    """The generated table matches a fresh computation, per-image counts included."""
    path = RESULTS_DIR / "sample_efficiency.md"
    if not path.exists():
        pytest.skip("sample_efficiency.md is not present")
    text = path.read_text(encoding="utf-8")

    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")

    # The table is generated for both baselines, so both sections are checked.
    sections = text.split("## Against ")[1:]
    assert len(sections) == 2, "expected one section per baseline"

    checked = 0
    for baseline, section in zip(["lasso-dct", "lasso"], sections, strict=True):
        reference = wide.loc[REFERENCE_M, baseline]
        target = reference.mean()
        assert f"{target:.4f}" in section, f"{baseline}: reference level missing"

        for method in PRIORS:
            budgets = means[method]
            matching = budgets.index[budgets <= target]
            label = LABELS[method]
            row = next(line for line in section.splitlines() if line.startswith(f"| {label} |"))
            if len(matching) == 0:
                assert "never" in row, f"{method}: expected no matching budget in {row!r}"
                checked += 1
                continue
            needed = int(matching.min())
            wins = int((wide.loc[needed, method].to_numpy() <= reference.to_numpy()).sum())
            assert f"| {needed} |" in row, f"{method}: expected budget {needed} in {row!r}"
            assert f"{wins} of 10" in row, f"{method}: expected {wins} wins in {row!r}"
            checked += 1
    assert checked == 2 * len(PRIORS)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("vae-30", 0.0070),
        ("fcvae-20", 0.0072),
        ("vae-20", 0.0076),
        ("dcgan-30", 0.0093),
        ("dcgan-20", 0.0119),
    ],
)
def test_quoted_error_floors(means, method, expected):
    """The floors quoted in the prose are the means from 300 measurements up."""
    floor = means[method][means.index >= 300].mean()
    assert round(floor, 4) == expected

    for path in (REPORT / "results.tex", ROOT_DIR / "README.md"):
        assert f"{expected:.4f}" in path.read_text(encoding="utf-8"), (
            f"{expected} missing from {path.name}"
        )


def test_the_reference_baseline_is_the_well_behaved_one(benchmark):
    """The stated reason for preferring the DCT reference has to hold."""
    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    pixel, dct = wide.loc[REFERENCE_M, "lasso"], wide.loc[REFERENCE_M, "lasso-dct"]

    # The pixel baseline is mid-transition here: mean far above median.
    assert pixel.mean() > 10 * pixel.median()
    assert int((pixel < 1e-3).sum()) == 6
    # The DCT baseline is not.
    assert dct.mean() == pytest.approx(dct.median(), rel=0.15)
    assert int((dct < 1e-3).sum()) == 0


def test_quoted_low_budget_factors(means):
    """The factors quoted at 25 measurements, against both baselines."""
    best = means.loc[25, PRIORS].min()
    assert means["lasso"][25] / best == pytest.approx(6.1, abs=0.05)
    assert means["lasso-dct"][25] / best == pytest.approx(5.1, abs=0.05)


def test_the_trivial_predictor_level_is_quoted_where_it_matters(benchmark, means):
    """A baseline at or above the blank-image error is not reconstructing anything.

    The write-ups quote that level and name the budgets where the pixel baseline
    sits at or above it, so both have to keep matching the data.
    """
    path = RESULTS_DIR / "reconstructions.npz"
    if not path.exists():
        pytest.skip("reconstructions.npz is not present")
    with np.load(path) as archive:
        trivial = float((archive["ground_truth"] ** 2).mean())

    degenerate = [int(m) for m in means.index if means["lasso"][m] >= trivial]
    assert degenerate == [10, 25]

    for doc in (ROOT_DIR / "README.md", REPORT / "results.tex"):
        text = doc.read_text(encoding="utf-8")
        assert f"{trivial:.4f}" in text, f"the blank-image level is missing from {doc.name}"


def test_crossover_budget(means):
    """Both baselines overtake every prior from 500 measurements up."""
    for m in (500, 750):
        assert means.loc[m, BASELINES].max() < means.loc[m, PRIORS].min()
    assert means.loc[400, PRIORS].min() < means.loc[400, BASELINES].min()


def test_quoted_recovery_costs(benchmark):
    """The per-configuration timings quoted in the cost sections."""
    seconds = benchmark.groupby("method")["seconds_per_batch"].mean()
    assert seconds["lasso"] == pytest.approx(1.5, abs=0.3)
    assert seconds["lasso-dct"] == pytest.approx(1.5, abs=0.3)
    assert seconds["fcvae-20"] == pytest.approx(6.6, abs=0.5)

    # The prose calls this the gap between the cheapest and the dearest learned
    # prior, so the test has to measure that and not an average over the two
    # DCGANs, which is a different number.
    ratio = seconds[PRIORS].max() / seconds[PRIORS].min()
    assert round(ratio) == 93
    # Two documents give the figure, a third spells it out in words. What matters
    # is that none of them names a different number.
    for path in (ROOT_DIR / "README.md", REPORT / "conclusions.tex"):
        assert "93" in path.read_text(encoding="utf-8"), f"{path.name} quotes another ratio"
    spelled = (REPORT / "summary.tex").read_text(encoding="utf-8")
    assert "more than ninety times" in spelled, "summary.tex no longer gives the cost ratio"
    cheapest = seconds[PRIORS].min()
    assert all(seconds[d] / cheapest > 90 for d in ("dcgan-20", "dcgan-30")), (
        "summary.tex says both DCGANs cost more than ninety times the cheapest VAE"
    )


def test_lambda_sweep_endpoints():
    """The claim that the best penalty is 1 at the smallest budgets and 0 at the largest."""
    path = RESULTS_DIR / "lambda_sweep.csv"
    if not path.exists():
        pytest.skip("lambda_sweep.csv is not present")
    sweep = pd.read_csv(path)
    table = sweep.pivot_table(index="m", columns=["method", "l2_penalty"], values="per_pixel_error")

    for method in table.columns.get_level_values(0).unique():
        best = table[method].idxmin(axis=1)
        assert best[10] == 1.0, f"{method} at the smallest budget"
        assert all(best[m] == 0.0 for m in (200, 300, 400, 500, 750)), f"{method} at the largest"
        # In between the three priors disagree, which is the point the write-ups
        # make about a single recommended value being a compromise.
    disagree = {
        float(table[method].idxmin(axis=1)[25])
        for method in table.columns.get_level_values(0).unique()
    }
    assert len(disagree) > 1, "the priors are expected to disagree at 25 measurements"


def normalise(text: str) -> str:
    r"""Strip the markup a phrase may be wrapped in, and collapse whitespace.

    The guard below matches phrases against prose that is written in Markdown and
    in LaTeX, where the same sentence may carry ``**bold**``, ``\\textbf{}``,
    backticks or a line break in the middle of it. Matching the raw text meant
    every phrase in the list silently failed to match, so the guard passed while
    the values it names were still present.
    """
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = text.replace("**", "").replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", text)


DOCUMENTS = [
    ROOT_DIR / "README.md",
    ROOT_DIR / "docs" / "model_selection.md",
    ROOT_DIR / "models" / "README.md",
]


def test_the_guard_would_notice_markup():
    """The normaliser has to see through the markup the prose actually uses."""
    assert "an 8x saving" in normalise("an **8x** saving")
    assert "a factor 73 between" in normalise("a factor 73\nbetween")
    assert "the paper beats ours" in normalise(r"the \textbf{paper} beats ours")


def test_no_superseded_figures_survive_in_the_prose():
    """Values from earlier states of the experiment must not linger anywhere."""
    stale = {
        "an 8x saving": "the headline is 5.3x, against either baseline",
        "5.6 and 4.7 times": "the m=25 factors are 6.1x and 5.1x",
        "5.6x more accurate": "the m=25 factors are 6.1x and 5.1x",
        "factor 73": "the cost ratio is 103",
        "73x gap": "the cost ratio is 103",
        "swept over six values": "the shrinkage sweep covers eight values, per budget",
        "simpler architecture beats ours": "the three VAE priors are indistinguishable",
        "simpler network beats ours": "the three VAE priors are indistinguishable",
        "simplest of the three generators was the best": (
            "the three VAE priors are indistinguishable"
        ),
        "at or below 0.004": "no comparison among the VAE priors is significant",
        "p = 0.028": "no comparison among the VAE priors is significant",
        "factor of three across seeds": "the seed spread is 3.5",
    }
    prose = {path: path.read_text(encoding="utf-8") for path in DOCUMENTS}
    prose |= {path: path.read_text(encoding="utf-8") for path in REPORT.glob("*.tex")}
    # Notebooks carry prose too, and their stored output is megabytes of base64,
    # so only the authored cells are searched.
    for path in sorted((ROOT_DIR / "notebooks").glob("*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        prose[path] = "\n".join("".join(cell["source"]) for cell in cells)

    for path, text in prose.items():
        text = normalise(text)
        for phrase, why in stale.items():
            assert phrase not in text, f"{path.name} still contains {phrase!r}: {why}"


def test_the_headline_figures_appear_in_every_write_up(means, benchmark):
    """The saving and the m=25 factor are recomputed and looked for by value.

    A blacklist only catches values someone thought to list. These are the two
    figures every write-up leads with, so they are checked positively instead.
    """
    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    target = wide.loc[REFERENCE_M, REFERENCE].mean()
    needed = min(int(means[p].index[means[p] <= target].min()) for p in PRIORS[:3])
    saving = REFERENCE_M / needed

    best25 = means.loc[25, PRIORS].min()
    factor = means["lasso-dct"][25] / best25

    for path in [ROOT_DIR / "README.md"]:
        text = normalise(path.read_text(encoding="utf-8"))
        assert f"{saving:.1f}x" in text, f"{path.name} is missing the {saving:.1f}x saving"
        assert f"{factor:.1f}" in text, f"{path.name} is missing the {factor:.1f} factor at m=25"


def test_the_recorded_revision_is_in_the_history():
    """A run records the revision it came from; that pointer has to resolve.

    Rewriting commit messages changes every hash, which silently orphans the
    provenance recorded by an earlier run.

    A shallow checkout has no older commits to resolve against, so there the
    question cannot be asked and the test skips rather than reporting an absence
    it cannot distinguish from a broken pointer.
    """
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    if shallow.returncode != 0:
        pytest.skip("not a git checkout")
    if shallow.stdout.strip() == "true":
        pytest.skip("the checkout is shallow, so earlier revisions are not present")

    for name in ("benchmark_meta.json", "unregularised/benchmark_meta.json"):
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        revision = json.loads(path.read_text(encoding="utf-8")).get("git_revision")
        if revision in (None, "unknown"):
            continue
        resolved = subprocess.run(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT_DIR
        )
        assert resolved.returncode == 0, (
            f"{name} records revision {revision}, which is not in the history"
        )


def test_bold_marks_the_best_method_in_the_readme(means):
    """The README table bolds a winner too, and it has to be the row minimum."""
    body = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    body = body[body.index("|   m | Lasso (pixel)") :]
    order = ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]

    rows = 0
    for line in body.splitlines()[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = int(cells[0])
        bolded = [order[i] for i, c in enumerate(cells[1:]) if "**" in c]
        best = means.loc[m, order].idxmin()
        assert bolded == [best], f"README row m={m}: bold on {bolded}, best is {best}"
        rows += 1
    assert rows == 10


def test_the_sample_efficiency_prose_covers_every_prior(means, benchmark):
    """Not only the VAEs: the DCGAN budgets are quoted too and must be right."""
    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    target = wide.loc[REFERENCE_M, REFERENCE].mean()
    readme = normalise((ROOT_DIR / "README.md").read_text(encoding="utf-8"))

    reaching = [m for m in ("dcgan-20", "dcgan-30") if (means[m] <= target).any()]
    missing = [m for m in ("dcgan-20", "dcgan-30") if m not in reaching]
    assert len(reaching) == 1 and len(missing) == 1, (
        f"the README describes one DCGAN reaching the level and one not; data says {reaching}"
    )

    hit, miss = reaching[0], missing[0]
    needed = int(means[hit].index[means[hit] <= target].min())
    assert f"DCGAN at {hit.replace('dcgan-', 'k=')} needs {needed}" in readme, (
        f"the README should say the {hit} needs {needed} measurements"
    )
    assert f"at {miss.replace('dcgan-', 'k=')} it never reaches the level" in readme


def test_the_unregularised_comparison_is_quoted_correctly():
    """The regulariser paragraph and the k=20 caveat rest on the second sweep."""
    path = RESULTS_DIR / "unregularised" / "benchmark.csv"
    if not path.exists():
        pytest.skip("the unregularised sweep is not present")
    unregularised = pd.read_csv(path).pivot_table(
        index="m", columns="method", values="per_pixel_error"
    )
    regularised = pd.read_csv(RESULTS_DIR / "benchmark.csv").pivot_table(
        index="m", columns="method", values="per_pixel_error"
    )

    documents = [
        normalise((ROOT_DIR / "README.md").read_text(encoding="utf-8")),
        normalise((REPORT / "results.tex").read_text(encoding="utf-8")),
    ]
    for m in (10, 750):
        with_penalty = regularised.loc[m, PRIORS].min()
        without = unregularised.loc[m, PRIORS].min()
        for text in documents:
            assert f"{without:.4f}" in text, f"the lambda=0 error at m={m} is {without:.4f}"
            assert f"{with_penalty:.4f}" in text, (
                f"the lambda=0.1 error at m={m} is {with_penalty:.4f}"
            )


def test_the_paper_penalty_sweep_is_quoted_correctly():
    """The DCGAN columns at the penalty the paper gives for a GAN.

    The write-ups use this sweep to say that the main table's penalty is not
    handicapping the DCGANs, so both floors have to match the data.
    """
    path = RESULTS_DIR / "dcgan_paper_penalty" / "benchmark.csv"
    if not path.exists():
        pytest.skip("the paper-penalty sweep is not present")

    at_paper = pd.read_csv(path).pivot_table(index="m", columns="method", values="per_pixel_error")
    main = pd.read_csv(RESULTS_DIR / "benchmark.csv").pivot_table(
        index="m", columns="method", values="per_pixel_error"
    )
    meta = json.loads(
        (RESULTS_DIR / "dcgan_paper_penalty" / "benchmark_meta.json").read_text(encoding="utf-8")
    )
    assert meta["protocol"]["l2_penalty"] == 0.001

    documents = [
        normalise((ROOT_DIR / "README.md").read_text(encoding="utf-8")),
        normalise((REPORT / "results.tex").read_text(encoding="utf-8")),
    ]
    for method in ("dcgan-20", "dcgan-30"):
        paper = at_paper[method][at_paper.index >= 300].mean()
        table = main[method][main.index >= 300].mean()
        for text in documents:
            assert f"{paper:.4f}" in text, f"{method} at 0.001 has floor {paper:.4f}"
            assert f"{table:.4f}" in text, f"{method} at 0.1 has floor {table:.4f}"

    # The claim the prose rests on: 0.001 is not the better setting for k=30.
    assert (
        at_paper["dcgan-30"][at_paper.index >= 300].mean()
        > main["dcgan-30"][main.index >= 300].mean()
    )


#: Figures that several documents quote. Each entry pairs the phrase a document
#: uses with the value recomputed from the data. The error this catches is a
#: correction applied to four places out of five: every copy is individually
#: plausible, so only reading the same figure out of every document at once shows
#: the disagreement. The expected value is never taken from a document.
SHARED_PHRASES = [
    ("largest corrected p-value", r"largest corrected p-value among these is (\d\.\d{4})"),
    ("blank-image level", r"(?:all-zero|blank) image[^.]*?(0\.\d{4})"),
    ("spread of the three VAE floors", r"floors?(?: differ by| span) (0\.\d{4})"),
    ("cost ratio", r"\*{0,2}(\d{2,3})x?\*{0,2} (?:gap|times the cost)"),
    ("sweep duration", r"(?:about|~) (\d\.\d) (?:h on CPU|hours of computation)"),
]


def test_shared_figures_agree_with_the_data_in_every_document(benchmark, means):
    """A figure quoted in several documents must be the recomputed one in each.

    The documents are never compared against each other, so a value that is
    wrong in all of them still fails rather than agreeing with itself.
    """
    vaes = ["fcvae-20", "vae-20", "vae-30"]
    floors = {k: means[k][means.index >= 300].mean() for k in PRIORS}
    seconds = benchmark.groupby("method")["seconds_per_batch"].mean()

    npz, sig_path = RESULTS_DIR / "reconstructions.npz", RESULTS_DIR / "significance.csv"
    if not npz.exists() or not sig_path.exists():
        pytest.skip("the derived artefacts are not present")
    with np.load(npz) as archive:
        blank = (archive["ground_truth"] ** 2).mean()
    significance = pd.read_csv(sig_path)

    largest_p = significance[significance.verdict == "better"].p_holm.max()
    vae_span = max(floors[k] for k in vaes) - min(floors[k] for k in vaes)
    hours = benchmark.groupby(["method", "m"])["seconds_per_batch"].first().sum() / 3600
    truth = {
        "largest corrected p-value": f"{largest_p:.4f}",
        "blank-image level": f"{blank:.4f}",
        "spread of the three VAE floors": f"{vae_span:.4f}",
        "cost ratio": f"{round(seconds[PRIORS].max() / seconds[PRIORS].min())}",
        "sweep duration": f"{hours:.1f}",
    }

    documents = {
        path.name: normalise(path.read_text(encoding="utf-8"))
        for path in (
            ROOT_DIR / "README.md",
            REPORT / "results.tex",
            REPORT / "conclusions.tex",
            REPORT / "summary.tex",
        )
    }

    checked = 0
    for label, pattern in SHARED_PHRASES:
        for name, text in documents.items():
            for stated in re.findall(pattern, text, re.IGNORECASE):
                assert stated == truth[label], (
                    f"{name} gives the {label} as {stated}, the data says {truth[label]}"
                )
                checked += 1
    assert checked >= len(SHARED_PHRASES), "the phrases stopped matching any document"
