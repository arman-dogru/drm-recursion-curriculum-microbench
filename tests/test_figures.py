import json

import pytest

import make_figures
from test_aggregate import grid  # noqa: F401  (fixture)
import aggregate as A


def test_figures_from_fabricated_summary(grid, tmp_path):  # noqa: F811
    raw, out, _, _ = grid
    A.main(raw, out)
    make_figures.main(out, tmp_path / "figs")
    names = {p.name for p in (tmp_path / "figs").glob("*.png")}
    assert {"primary.png", "forest.png", "compute_curves.png", "schedule_heatmaps.png", "mismatch.png",
            "learning_curves.png", "ksweep.png", "tasks.png", "mismatch_own_schedule.png"} <= names


def test_figures_refuse_partial(grid, tmp_path):  # noqa: F811
    raw, out, _, jobs = grid
    A.main(raw, out)
    s = json.loads((out / "summary.json").read_text())
    s["meta"]["partial"] = True
    (out / "summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit) as e:
        make_figures.main(out, tmp_path / "figs")
    assert e.value.code == 2
