"""Read the protocol out of the prose and check it against the code.

The other guards assert that the code is as intended. That is not the same as
asserting the write-ups describe it: a test comparing a script default to a
constant written in the test passes whatever the README says. A mutation audit
over every hand-written number made the gap measurable, so the quantities the
documents state about the setup are extracted from the documents here and
compared with the value the code actually produces.

Every pattern is anchored on the words around the number, and each quantity
declares how many times it must be found, so a pattern that stops matching after
a rewording fails instead of passing silently.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest
from helpers import script_defaults

from csgm.config import ROOT_DIR

DOCUMENTS = [
    ROOT_DIR / "README.md",
    ROOT_DIR / "docs" / "model_selection.md",
    ROOT_DIR / "models" / "README.md",
    *sorted((ROOT_DIR / "docs" / "report").glob("*.tex")),
    *sorted((ROOT_DIR / "notebooks").glob("*.ipynb")),
]


def _prose(path):
    """The authored text of a document, with markup that hides numbers removed."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    else:
        text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\(?:textbf|texttt|emph|mathrm)\{([^}]*)\}", r"\1", text)
    text = text.replace("\\(", " ").replace("\\)", " ").replace("{,}", ",")
    text = re.sub(r"[\\{}`*$]", " ", text)
    return re.sub(r"\s+", " ", text)


@pytest.fixture(scope="module")
def prose():
    """Each document's text, keyed by its path.

    Keyed by path and not by name: two of these files are called README.md, and
    keying by name silently dropped the larger of them, which is the one most of
    these numbers live in.
    """
    return {str(path.relative_to(ROOT_DIR)).replace("\\", "/"): _prose(path) for path in DOCUMENTS}


VAE_DOCUMENTS = ("vae.tex", "01_vae_training.ipynb")
DCGAN_DOCUMENTS = ("dcgan.tex", "02_dcgan_training.ipynb", "model_selection.md")


def _quantities():
    """(label, pattern, expected text, how many times it must appear, documents).

    The last field scopes a pattern to the documents that state that quantity,
    since the two families use the same words for different values: a pattern
    reading ``batch size of`` everywhere would compare the DCGAN's 64 with the
    VAE's 100 and call one of them wrong.
    """
    benchmark = script_defaults("run_benchmark")
    vae = script_defaults("train_vae")
    dcgan = script_defaults("train_dcgan")
    select = script_defaults("select_dcgan")
    training_images = int(60_000 * (1 - dcgan["--validation-fraction"]))

    return [
        (
            "images the generators train on",
            r"(?:same|on the|trains on) (\d{2},\d{3}) images",
            f"{training_images:,}",
            3,
            None,
        ),
        (
            "DCGAN learning rate",
            r"learning rate of (0\.\d+)",
            str(dcgan["--learning-rate"]),
            2,
            DCGAN_DOCUMENTS,
        ),
        ("DCGAN momentum", r"beta_?1 *=? *(0\.\d+)", str(dcgan["--beta-1"]), 2, DCGAN_DOCUMENTS),
        (
            "DCGAN batch size",
            r"(?:mini-?batches of|batch size of) (\d+)",
            str(dcgan["--batch-size"]),
            2,
            DCGAN_DOCUMENTS,
        ),
        (
            "DCGAN epochs",
            r"(?:for|runs for|was|,) (\d+) epochs",
            str(dcgan["--epochs"]),
            3,
            DCGAN_DOCUMENTS,
        ),
        (
            "VAE batch size",
            r"(?:mini-?batches of|batch size of) (\d+)",
            str(vae["--batch-size"]),
            1,
            VAE_DOCUMENTS,
        ),
        (
            "VAE learning rate",
            r"learning rate of (0\.\d+)",
            str(vae["--learning-rate"]),
            1,
            VAE_DOCUMENTS,
        ),
        ("VAE epochs", r"up to (\d+) epochs", str(vae["--epochs"]), 1, VAE_DOCUMENTS),
        ("recovery steps", r"--steps (\d+)", str(benchmark["--steps"]), 1, None),
        ("recovery restarts", r"--restarts (\d+)", str(benchmark["--restarts"]), 1, None),
        ("images scored", r"--n-images (\d+)", str(benchmark["--n-images"]), 1, None),
        (
            "expected noise norm",
            r"expected norm[^.]{0,30}?(0\.\d+)",
            str(benchmark["--noise-norm"]),
            3,
            None,
        ),
        (
            "images the selection scores",
            r"(\d+) (?:images )?held out of training",
            str(select["--n-images"]),
            1,
            None,
        ),
        ("ambient dimension", r"(\d{3})[ -]pixel", "784", 2, None),
    ]


def test_the_documents_quote_the_protocol_the_code_runs(prose):
    """Every documented setup number must be the one the scripts default to."""
    wrong, counts = [], {}
    for label, pattern, expected, minimum, documents in _quantities():
        found = 0
        for name, text in prose.items():
            if documents and not any(name.endswith(d) for d in documents):
                continue
            for stated in re.findall(pattern, text, re.IGNORECASE):
                found += 1
                if stated.rstrip("0").rstrip(".") != expected.rstrip("0").rstrip("."):
                    wrong.append(f"{name}: {label} is stated as {stated}, the code uses {expected}")
        counts[label] = (found, minimum)

    assert not wrong, "; ".join(wrong)
    thin = {k: v for k, v in counts.items() if v[0] < v[1]}
    assert not thin, f"these patterns no longer find what they guard, so they guard nothing: {thin}"


def test_the_dataset_the_documents_describe_is_mnist(prose):
    """The dataset facts are quoted in several places and never checked."""
    facts = [
        ("training set size", r"training set consists of (\d{2},\d{3}) images", "60,000", 1),
        ("test set size", r"test set consists of (\d{2},\d{3}) images", "10,000", 1),
        ("image side", r"(\d{2}) ?(?:x|times) ?\d{2} pixels", "28", 1),
        ("features", r"total of (\d{3}) features", "784", 1),
        ("classes", r"There are (\d+) classes", "10", 1),
    ]
    wrong, thin = [], []
    for label, pattern, expected, minimum in facts:
        found = 0
        for name, text in prose.items():
            for stated in re.findall(pattern, text, re.IGNORECASE):
                found += 1
                if stated != expected:
                    wrong.append(f"{name}: {label} is {stated}, MNIST has {expected}")
        if found < minimum:
            thin.append(label)
    assert not wrong, "; ".join(wrong)
    assert not thin, f"these patterns find nothing any more: {thin}"


def _shapes(text):
    """Normalise the two ways the write-ups spell a tensor shape."""
    text = re.sub(r"\\times|\\cdot", "x", text)
    text = re.sub(r"(\d+)\^2", r"\1x\1", text)
    text = re.sub(r"[\\${}`*]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"(\d) x (\d)", r"\1x\2", re.sub(r"(\d) x (\d)", r"\1x\2", text))


def test_the_architecture_the_prose_describes_is_the_one_that_is_built():
    """Layer sizes are quoted in sentences and tables outside the report too.

    The report's own tables are checked against the built networks elsewhere.
    The same numbers appear in the README's comparison table and in the prose of
    the other write-ups, where nothing reaches them, so a layer changed in
    ``csgm.models`` would leave those descriptions behind.
    """
    from csgm.models import build_decoder, build_encoder, build_generator

    decoder, generator, encoder = build_decoder(20), build_generator(20), build_encoder(20)

    def tensor_shapes(model):
        return [tuple(int(d) for d in layer.output.shape[1:]) for layer in model.layers]

    def filters(model):
        return [int(layer.filters) for layer in model.layers if getattr(layer, "filters", None)]

    raw_readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    readme = _shapes(raw_readme)

    projections = re.findall(r"dense to (\d+x\d+x\d+)", readme)
    assert len(projections) == 2, f"the README no longer states both projections: {projections}"
    assert {tuple(int(v) for v in p.split("x")) for p in projections} == {
        next(s for s in tensor_shapes(decoder) if len(s) == 3),
        next(s for s in tensor_shapes(generator) if len(s) == 3),
    }

    upsample = re.search(r"transposed conv, (\d+) filters, to (\d+x\d+)", readme)
    assert upsample, "the README no longer describes the decoder's upsampling"
    assert int(upsample.group(1)) == filters(decoder)[0]
    assert upsample.group(2) == "x".join(str(d) for d in tensor_shapes(decoder)[-1][:2])

    widest = re.search(r"transposed convs with ([\d, and]+) filters", readme)
    assert widest, "the README no longer describes the generator's upsampling"
    assert [int(v) for v in re.findall(r"\d+", widest.group(1))] == filters(generator)[:3]

    counts = re.findall(r"\| ([\d,]{6,}) \| ([\d,]{6,}) \|", raw_readme)
    assert (f"{decoder.count_params():,}", f"{generator.count_params():,}") in counts, (
        f"the README gives {counts} as the two parameter counts"
    )

    encoder_prose = _shapes((ROOT_DIR / "docs" / "report" / "vae.tex").read_text(encoding="utf-8"))
    listed = re.search(r"using ([\d, and]+) filters respectively", encoder_prose)
    assert listed, "vae.tex no longer lists the encoder's filters"
    assert [int(v) for v in re.findall(r"\d+", listed.group(1))] == filters(encoder)


def test_the_paper_architecture_is_spelled_out_correctly(prose):
    """The reference network is quoted as a shape and as a parameter count.

    It is the one prior taken from someone else's paper, so a reader comparing
    our numbers with theirs starts here. The shape appears in three documents and
    the count in one, and both come from a network the code can build.
    """
    from csgm.models import build_fc_decoder

    decoder = build_fc_decoder(20)
    widths = [int(w.shape[0]) for w in decoder.weights if len(w.shape) == 2]
    spelled = "-".join(str(w) for w in reversed([*widths, 784]))

    found = 0
    for name, text in prose.items():
        # The README sets each width in maths, so the dashes come back spaced.
        joined = re.sub(r"(\d) *- *(\d)", r"\1-\2", text)
        for stated in re.findall(r"fully connected (\d[\d-]+\d)", joined):
            found += 1
            assert stated == spelled, f"{name}: states {stated}, the built network is {spelled}"
    assert found >= 2, "the reference architecture is no longer spelled out anywhere"

    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    quoted = re.search(r"decoder has ([\d,]+) parameters", readme)
    assert quoted, "the README no longer gives the size of the paper's decoder"
    assert quoted.group(1) == f"{decoder.count_params():,}"


def test_the_badges_promise_the_versions_the_project_supports():
    """A badge is the first thing a reader believes and the last thing anyone edits.

    The workflow is found by glob rather than by name: its file gets renamed
    whenever the run numbering is restarted, and a guard that breaks on a rename
    would be one more thing to remember.
    """
    readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    workflows = sorted((ROOT_DIR / ".github" / "workflows").glob("*.yml"))
    assert len(workflows) == 1, f"expected one workflow, found {[p.name for p in workflows]}"
    workflow = workflows[0].read_text(encoding="utf-8")

    badge_target = re.search(r"actions/workflows/([\w.-]+)/badge\.svg", readme)
    assert badge_target, "the README no longer carries a CI badge"
    assert badge_target.group(1) == workflows[0].name, (
        f"the badge points at {badge_target.group(1)}, the workflow is {workflows[0].name}"
    )

    requires = re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject).group(1)
    lowest = re.search(r">=\s*(\d+\.\d+)", requires).group(1)
    tested = sorted(set(re.findall(r'"(3\.\d+)"', workflow)))
    assert tested, "the workflow no longer names the versions it runs on"
    assert lowest in tested, f"requires-python allows {lowest}, which the CI does not run"

    badge = re.search(r"python-([\d.%|\s]+)-blue", readme)
    assert badge, "the README no longer carries a Python badge"
    promised = sorted(set(re.findall(r"3\.\d+", badge.group(1))))
    assert promised == tested, f"the badge promises {promised}, the CI runs {tested}"


def test_every_budget_and_latent_dimension_named_in_prose_exists(prose):
    """A budget or a latent dimension the project never ran cannot be discussed.

    These two are the most repeated numbers in the write-ups and the easiest to
    mistype, and unlike an error value they are drawn from a set of ten and a set
    of two, so membership settles them outright. The audit that measures what the
    suite catches found them to be the largest unguarded group.
    """
    table = pd.read_csv(ROOT_DIR / "results" / "benchmark.csv")
    budgets = {str(m) for m in sorted(table["m"].unique())}
    shipped = {
        re.search(r"dim(\d+)", path.name).group(1)
        for path in (ROOT_DIR / "models").glob("*decoder_dim*.keras")
    }
    assert budgets and shipped, "the artefacts that define these sets are missing"

    # Two dimensions appear legitimately that no shipped model has: the two
    # dimensional latent space that notebook 01 plots, and the celebA DCGAN of
    # the reference paper, which it runs at 100.
    allowed_dimensions = shipped | {"2", "100"}

    checks = [
        ("budget", r"(\d+) measurements", budgets, 40),
        ("budget", r"\bm *= *(\d+)", budgets, 4),
        ("latent dimension", r"\bk *= *(\d+)", allowed_dimensions, 60),
        ("latent dimension", r"latent dimension (?:of )?(\d+)", allowed_dimensions, 20),
    ]

    wrong, totals = [], {}
    for label, pattern, permitted, minimum in checks:
        found = 0
        for name, text in prose.items():
            for stated in re.findall(pattern, text):
                found += 1
                if stated not in permitted:
                    wrong.append(f"{name}: {label} {stated}, which the project never ran")
        totals[pattern] = (found, minimum)

    assert not wrong, "; ".join(wrong)
    thin = {k: v for k, v in totals.items() if v[0] < v[1]}
    assert not thin, f"these patterns no longer find what they guard: {thin}"
