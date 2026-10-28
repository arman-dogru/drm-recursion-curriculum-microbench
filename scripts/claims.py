"""Derive every number quoted in README/PR alt text and prose from result files.

Writes results/claim_numbers.json (flat key -> string). render_readme.py fills
{{key}} placeholders in docs/README.template.md from it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drm.config import METHODS, ROOT  # noqa: E402
from drm.tasks import TASKS  # noqa: E402

R = ROOT / "results"


def pct(x):
    return f"{100 * x:.1f}"


def pp(x):
    return f"{100 * x:+.1f}"


def ci(r):
    return f"{pct(r['mean'])}% [{pct(r['lo'])}, {pct(r['hi'])}]"


def build(s, mut, n_tests):
    c = {"code_hash": s["meta"]["code_hash"], "n_params": f"{s['meta']['n_params'][0]:,}",
         "n_eval_seeds": str(len(s["meta"]["eval_seeds"])), "holm_m": str(s["meta"]["holm_family_size"]),
         "n_runs_main": str(s["meta"]["n_runs_main"]), "n_runs_ksweep": str(s["meta"]["n_runs_ksweep"]),
         "n_tests": str(n_tests)}
    if mut:
        c["mut_killed"] = str(mut["counts"]["KILLED"])
        c["mut_total"] = str(mut["n"])
    for g, ms in s["primary"].items():
        for m, r in ms.items():
            c[f"p1_{g}_{m}"] = ci(r)
            c[f"p1m_{g}_{m}"] = pct(r["mean"])
    for g in TASKS:
        for m in METHODS:
            ch = s["tuning"][g][m]
            c[f"lr_{g}_{m}"] = f"{ch['lr']:g}"
            c[f"sched_{g}_{m}"] = f"S={ch['schedule'][0]}, k_in={ch['schedule'][1]}"
    sig = 0
    for k, r in s["tests"].items():
        key = k.replace(":", "_").replace("-", "_vs_")
        c[f"d_{key}"] = f"{pp(r['diff'])} pp [{pp(r['lo'])}, {pp(r['hi'])}]"
        c[f"wl_{key}"] = f"{r['wins']}/{r['losses']}"
        c[f"ph_{key}"] = f"{r['p_holm']:.2g}"
        sig += r["significant"]
    c["n_sig"] = str(sig)
    for t in TASKS:
        tr = s["traces"][t]
        for m in METHODS:
            own = tr[m]["S16_k1"]["own"]
            ora = tr[m]["S16_k1"]["oracle"]
            vw = [x["visible_wrong_frac"] for x in own[1:] if x["visible_wrong_frac"] is not None]
            sc = [x["self_correct_rate"] for x in own[1:] if x["self_correct_rate"] is not None]
            c[f"mm_{t}_{m}_own_final_exact"] = pct(own[-1]["exact"])
            c[f"mm_{t}_{m}_gap_mid"] = pp(ora[len(ora) // 2]["cell_acc"] - own[len(own) // 2]["cell_acc"])
            c[f"mm_{t}_{m}_vis_wrong"] = pct(float(np.mean(vw))) if vw else "n/a"
            c[f"mm_{t}_{m}_selfcorr"] = pct(float(np.mean(sc))) if sc else "n/a"
    for t, ms in s["ksweep"].items():
        for m, ks in ms.items():
            for k, v in ks.items():
                c[f"ks_{t}_{m}_k{k}"] = ci(v["native"])
                c[f"ksl_{t}_{m}_k{k}"] = ci(v["latent16"])
    for t in TASKS:
        for m in METHODS:
            cur = s["compute_curves"][t][m]
            for q in cur:
                c[f"cc_{t}_{m}_{q}"] = pct(cur[q]["mean"])
    c["train_s_drm"] = f"{s['train_seconds']['drm']:.0f}"
    c["train_s_fwd"] = f"{s['train_seconds']['fwd']:.0f}"
    return c


def count_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                       capture_output=True, text=True)
    return sum("::" in ln for ln in r.stdout.splitlines())


def main():
    s = json.loads((R / "summary.json").read_text())
    if s["meta"].get("partial"):
        print("claims: refusing partial summary", file=sys.stderr)
        sys.exit(2)
    mut = json.loads((R / "mutation.json").read_text()) if (R / "mutation.json").exists() else None
    c = build(s, mut, count_tests())
    (R / "claim_numbers.json").write_text(json.dumps(c, indent=1, sort_keys=True))
    print(f"{len(c)} claim numbers")


if __name__ == "__main__":
    main()
