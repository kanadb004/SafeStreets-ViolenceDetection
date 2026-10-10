"""Sprint 6 DoD checks: the six notebooks execute with saved outputs and no errors, the
submission docs exist, and README's results table matches docs/RESULTS.md. See
docs/SPRINT_PLAN.md Sprint 6.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"

EXPECTED_NOTEBOOKS = [
    "01_data_and_preprocessing.ipynb",
    "02_augmentation.ipynb",
    "03_model_and_training.ipynb",
    "04_evaluation.ipynb",
    "05_hpo_and_finetune.ipynb",
    "06_inference_and_demo.ipynb",
]


@pytest.mark.sprint6
@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_exists_and_has_no_cached_errors(name):
    path = NOTEBOOKS_DIR / name
    assert path.exists(), f"{path} is missing"
    nb = json.loads(path.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{name} has no code cells"

    errors = []
    any_outputs = False
    for i, cell in enumerate(code_cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename"), out.get("evalue")))
            if out:
                any_outputs = True
    assert not errors, f"{name} has cached cell errors: {errors}"
    assert any_outputs, f"{name} has no saved outputs; it was never executed (DoD: outputs saved)"


@pytest.mark.sprint6
def test_notebook_is_thin_no_reimplementation_markers():
    """A weak proxy for "no copy-pasted package reimplementation": every notebook must import
    from safestreets, and none should define its own build_model/build_lstm_head/etc. function.
    """
    banned_defs = ("def build_lstm_head", "def build_scratch_cnn_lstm", "def build_model(",
                   "def train_lstm_head", "def compute_metrics(")
    for name in EXPECTED_NOTEBOOKS:
        nb = json.loads((NOTEBOOKS_DIR / name).read_text())
        source = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "safestreets" in source, f"{name} never imports from safestreets"
        for banned in banned_defs:
            assert banned not in source, f"{name} reimplements {banned!r} instead of importing it"


@pytest.mark.sprint6
@pytest.mark.parametrize(
    "doc",
    ["MODEL_CARD.md", "RESULTS.md", "REPRODUCE.md", "PRESENTATION_GUIDE.md", "ETHICS.md",
     "DATASETS.md"],
)
def test_submission_doc_exists_and_nonempty(doc):
    path = DOCS_DIR / doc
    assert path.exists(), f"{path} is missing"
    assert len(path.read_text().strip()) > 200, f"{path} looks empty or stub-only"


def _extract_metric_table(text: str) -> dict[str, str]:
    """Pulls {metric_name: value} from the first `| Metric | Value | ...` table in `text`."""
    rows = {}
    in_table = False
    for line in text.splitlines():
        if line.strip().startswith("| Metric | Value"):
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            if line.strip().startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                rows[cells[0]] = cells[1]
    return rows


@pytest.mark.sprint6
def test_readme_results_table_matches_results_md():
    readme = (REPO_ROOT / "README.md").read_text()
    results = (DOCS_DIR / "RESULTS.md").read_text()

    readme_rows = _extract_metric_table(readme)
    results_rows = _extract_metric_table(results)

    assert readme_rows, "README.md has no `| Metric | Value |` table to compare"
    for metric, value in readme_rows.items():
        assert metric in results_rows, f"README reports {metric!r} not present in docs/RESULTS.md"
        assert value == results_rows[metric], (
            f"README's {metric} ({value}) disagrees with docs/RESULTS.md ({results_rows[metric]})"
        )


@pytest.mark.sprint6
def test_progress_md_requirements_table_fully_dispositioned():
    """Every row of BUILD_PLAN.md §2's requirements table must be marked delivered, scaled
    down, or cut in docs/PROGRESS.md, per the Sprint 6 DoD."""
    progress = (DOCS_DIR / "PROGRESS.md").read_text()
    has_section = (
        "Requirements disposition" in progress or "requirements disposition" in progress.lower()
    )
    assert has_section, "docs/PROGRESS.md has no requirements-disposition section"
    build_plan = (DOCS_DIR / "BUILD_PLAN.md").read_text()
    req_rows = re.findall(r"^\| §[^|]+ \| (.+?) \| [\d, ]+ \|$", build_plan, re.MULTILINE)
    n = len(req_rows)
    assert n == 14, f"expected 14 rows in BUILD_PLAN.md SS2's requirements table, got {n}"
    for requirement in req_rows:
        short = requirement.replace("**", "")[:25]
        assert short in progress, (
            f"requirement {requirement!r} has no matching disposition note in docs/PROGRESS.md"
        )


@pytest.mark.sprint6
def test_architecture_figure_exists():
    fig = REPO_ROOT / "artifacts" / "figures" / "architecture.png"
    assert fig.exists(), "artifacts/figures/architecture.png is missing"
    assert fig.stat().st_size > 1000, "architecture.png looks empty"
