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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import MAIN_TASKS  # noqa: E402

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
            c[f"dev_{g}_{m}"] = pct(ch["dev_pass1"])
            c[f"sched_{g}_{m}"] = f"S={ch['schedule'][0]}, k_in={ch['schedule'][1]}"
    sig = 0
    for k, r in s["tests"].items():
        key = k.replace(":", "_").replace("-", "_vs_")
        c[f"d_{key}"] = f"{pp(r['diff'])} pp [{pp(r['lo'])}, {pp(r['hi'])}]"
        c[f"wl_{key}"] = f"{r['wins']}/{r['losses']}"
        c[f"ph_{key}"] = f"{r['p_holm']:.2g}"
        sig += r["significant"]
    c["n_sig"] = str(sig)
    for t in MAIN_TASKS:
        tr = s["traces"][t]
        for m in METHODS:
            own = tr[m]["S16_k1"]["own"]
            ora = tr[m]["S16_k1"]["oracle"]
            vw = [x["visible_wrong_frac"] for x in own[1:] if x["visible_wrong_frac"] is not None]
            sc = [x["self_correct_rate"] for x in own[1:] if x["self_correct_rate"] is not None]
            c[f"mm_{t}_{m}_own_final_exact"] = pct(own[-1]["exact"])
            c[f"mm_{t}_{m}_own_last_cell"] = pct(own[-1]["cell_acc"])
            c[f"mm_{t}_{m}_oracle_last_cell"] = pct(ora[-1]["cell_acc"])
            c[f"mm_{t}_{m}_first_cell"] = pct(own[0]["cell_acc"])
            c[f"mm_{t}_{m}_gap_mid"] = pp(ora[len(ora) // 2]["cell_acc"] - own[len(own) // 2]["cell_acc"])
            c[f"mm_{t}_{m}_vis_wrong"] = pct(float(np.mean(vw))) if vw else "n/a"
            c[f"mm_{t}_{m}_selfcorr"] = pct(float(np.mean(sc))) if sc else "n/a"
    for t, ms in s["ksweep"].items():
        for m, ks in ms.items():
            for k, v in ks.items():
                c[f"ks_{t}_{m}_k{k}"] = ci(v["native"])
                c[f"ksl_{t}_{m}_k{k}"] = ci(v["latent16"])
                c[f"ksh_{t}_{m}_k{k}"] = ci(v["horizon"])
                c[f"ksh_kin_{t}_{m}_k{k}"] = str(v["horizon_k_in"])
    for t in MAIN_TASKS:
        for m in METHODS:
            cur = s["compute_curves"][t][m]
            for q in cur:
                c[f"cc_{t}_{m}_{q}"] = pct(cur[q]["mean"])
    for t in MAIN_TASKS:
        for m in METHODS:
            cur = s["compute_curves"][t][m]
            for q in cur:
                c[f"ccs_{t}_{m}_{q}"] = "S={}, k_in={}".format(*cur[q]["schedule"])
                c[f"ccci_{t}_{m}_{q}"] = ci(cur[q])
    for g, ms in s["primary"].items():
        for m, r in ms.items():
            c[f"p1t_{g}_{m}"] = f"[{pct(r['t_lo'])}, {pct(r['t_hi'])}]"
    for k, r in s["tests"].items():
        key = k.replace(":", "_").replace("-", "_vs_")
        c[f"dt_{key}"] = f"[{pp(r['t_lo'])}, {pp(r['t_hi'])}]"
    for t, ms in s["traces_at_choice"].items():
        for m, r in ms.items():
            c[f"mmo_{t}_{m}_sched"] = "S={}, k_in={}".format(*r["schedule"])
            c[f"mmo_{t}_{m}_own_last"] = pct(r["own"][-1]["cell_acc"])
            c[f"mmo_{t}_{m}_oracle_last"] = pct(r["oracle"][-1]["cell_acc"])
            c[f"mmo_{t}_{m}_gap_last"] = pp(r["oracle"][-1]["cell_acc"] - r["own"][-1]["cell_acc"])
            sc = [x["self_correct_rate"] for x in r["own"][1:] if x["self_correct_rate"] is not None]
            c[f"mmo_{t}_{m}_selfcorr"] = pct(float(np.mean(sc))) if sc else "n/a"
    ch = s["cpu_hours"]
    for k in ("dev", "eval", "ksweep", "total"):
        c[f"cpu_h_{k}"] = f"{ch[k]:.1f}"
    c["run_first_start"] = (ch["first_start"] or "n/a").replace("T", " ")[:16]
    c["run_last_finish"] = (ch["last_finish"] or "n/a").replace("T", " ")[:16]
    edges = s["tuning_meta"].get("lr_edges", [])
    c["lr_edges"] = ", ".join(edges) if edges else "none"
    c["n_lr_edges"] = str(len(edges))
    for b in ("fwd", "fwd_steps", "bwd_k", "bwd_tbptt"):
        for name, ts in (("dev5", TASKS), ("dev3", ("shift", "gravity", "flood"))):
            v = np.mean([s["tuning"][t]["drm"]["dev_pass1"] - s["tuning"][t][b]["dev_pass1"] for t in ts])
            c[f"{name}_drm_vs_{b}"] = f"{pp(v)} pp"
    r0 = ROOT / "docs" / "r0_claim_numbers.json"     # copy of results/r0_932be94/claim_numbers.json
    if r0.exists():
        for k, v in json.loads(r0.read_text()).items():
            if k.startswith(("p1_pooled_", "d_pooled_", "d_gravity_", "d_flood_", "d_shift_", "cc_flood_",
                             "cc_gravity_", "lr_")) or k in ("mut_killed", "mut_total", "n_tests"):
                c[f"r0_{k}"] = v
    r1 = ROOT / "docs" / "r1_claim_numbers.json"     # copy of results/r1_c4532b5/claim_numbers.json
    if r1.exists():
        for k, v in json.loads(r1.read_text()).items():
            if k.startswith(("p1_", "p1m_", "d_", "ph_", "wl_", "lr_", "sched_", "cc_")) or k in ("n_sig",):
                c[f"r1_{k}"] = v
    r1t = ROOT / "docs" / "r1_tuning.json"           # copy of results/r1_c4532b5/tuning.json
    if r1t.exists():
        old = json.loads(r1t.read_text())["choice"]
        changed = [f"{t} {m}: {old[t][m]['lr']:g} -> {s['tuning'][t][m]['lr']:g}"
                   for t in old for m in old[t] if old[t][m]["lr"] != s["tuning"][t][m]["lr"]]
        c["n_cells_lr_changed"] = str(len(changed))
        c["cells_lr_changed"] = "; ".join(changed)
        c["n_runs_r2"] = str(len(s["meta"]["eval_seeds"]) * len(changed))
        secs = 0.0
        for t in old:
            for m in old[t]:
                if old[t][m]["lr"] != s["tuning"][t][m]["lr"]:
                    lr = f"{s['tuning'][t][m]['lr']:g}".replace(".", "p")
                    for js in (R / "raw" / "eval" / t).glob(f"{m}_k4_lr{lr}_b*_s*.json"):
                        d = json.loads(js.read_text())
                        secs += d["train"].get("train_seconds", 0) + d["meta"].get("eval_seconds", 0)
        c["cpu_h_r2"] = f"{secs / 3600:.1f}"
    # code_hash = current code; code_hash_parent = the hash the results were computed under
    # (equal unless a verified lineage link, results/code_lineage.json, connects them)
    from drm.config import code_hash as current_code_hash
    c["code_hash_parent"] = ", ".join(s["meta"].get("computed_under") or [s["meta"]["code_hash"]])
    c["code_hash"] = current_code_hash()
    lin = R / "code_lineage.json"
    e = json.loads(lin.read_text())["equivalent"].get(c["code_hash"], {}) if lin.exists() else {}
    c["lineage_n"] = str(e.get("n_checks", 0))
    if mut:
        for pre, name in (("X", "rev"), ("Q", "r1"), ("Y", "rev2"), ("Z", "r2"), ("W", "rev3"), ("P", "rev4")):
            sel = [r for r in mut.get("mutants", []) if r["id"].startswith(pre)]
            c[f"mut_{name}_killed"] = str(sum(r["status"] == "KILLED" for r in sel))
            c[f"mut_{name}_total"] = str(len(sel))
    # ---- review r3: seed spread of headline cells, robustness checks, total compute spent
    for g, ms in s["primary"].items():
        for m, r in ms.items():
            if "sd" in r:
                c[f"sd_{g}_{m}"] = pct(r["sd"])
                c[f"range_{g}_{m}"] = f"{pct(r['min'])}-{pct(r['max'])}"
    sens = s.get("sensitivity", {})
    for g, r in sens.get("bwd_k_native", {}).items():
        d = r["drm_minus_bwd_k_native"]
        c[f"snat_{g}_bwd_k"] = ci(r["bwd_k_native"])
        c[f"snat_{g}_bwd_k_sd"] = pct(r["bwd_k_native"]["sd"])
        c[f"snat_{g}_d"] = f"{pp(d['diff'])} pp [{pp(d['lo'])}, {pp(d['hi'])}]"
        c[f"snat_{g}_p"] = f"{d['p']:.2g}"
        c[f"snat_{g}_wl"] = f"{d['wins']}/{d['losses']}"
    # review r4 m1: DRM and bwd_k at the SAME schedule (scripts/same_schedule.py, descriptive)
    ssp = R / "same_schedule.json"
    for t in MAIN_TASKS:            # every key exists even without the file (template-key test)
        for role in ("bwd_k_native", "bwd_k_pick", "drm_pick"):
            for f in ("sched", "d", "p", "wl"):
                c[f"ss_{t}_{role}_{f}"] = "n/a"
    if ssp.exists():
        for t, rows in json.loads(ssp.read_text())["same_schedule"].items():
            for role, x in rows.items():
                d = x["drm_minus_bwd_k"]
                c[f"ss_{t}_{role}_sched"] = "S{}/k{}".format(*x["schedule"])
                c[f"ss_{t}_{role}_d"] = f"{pp(d['diff'])} pp [{pp(d['lo'])}, {pp(d['hi'])}]"
                c[f"ss_{t}_{role}_p"] = f"{d['p']:.2g}"
                c[f"ss_{t}_{role}_wl"] = f"{d['wins']}/{d['losses']}"
    tie = sens.get("tie_symfill_tbptt_r1")
    c["tie_symfill_tbptt_r1"] = ci(tie["symfill_bwd_tbptt"]) if tie else "n/a"
    c["tie_pooled_d_vs_tbptt"] = (f"{pp(tie['pooled_drm_minus_bwd_tbptt']['diff'])} pp [{pp(tie['pooled_drm_minus_bwd_tbptt']['lo'])}, "
                                  f"{pp(tie['pooled_drm_minus_bwd_tbptt']['hi'])}]") if tie else "n/a"

    # exact cross-lr dev ties decided by the tie rule (lowest lr first; review r3 m3), colormap excluded
    tp = R / "tuning.json"
    ties = []
    if tp.exists():
        td = json.loads(tp.read_text())
        menu = {f"S{a}_k{b}" for a, b in td["meta"]["menu"]}
        for t, ms in td.get("dev_scores", {}).items():
            for m, lrs in ms.items():
                best = max(v for sc in lrs.values() for k, v in sc.items() if k in menu)
                tied = {lr for lr, sc in lrs.items() for k, v in sc.items() if k in menu and abs(v - best) < 1e-12}
                if len(tied) > 1 and t != "colormap":
                    ties.append(f"{t} {m}")
    c["n_tie_cells"] = str(len(ties)) if tp.exists() else "n/a"
    c["tie_cells"] = ", ".join(sorted(ties)) if tp.exists() else "n/a"

    def raw_hours(root):
        files = list(root.rglob("*.json")) if root.exists() else []
        tot = 0.0
        for f in files:
            d = json.loads(f.read_text())
            if "train" in d and "meta" in d:
                tot += d["train"].get("train_seconds", 0) + d["meta"].get("eval_seconds", 0)
        return (f"{tot / 3600:.1f}" if files else "n/a"), len(files)
    c["cpu_h_superseded"], c["n_runs_superseded"] = (lambda h, n: (h, str(n)))(*raw_hours(R / "r1_c4532b5" / "raw"))
    c["cpu_h_r0"], c["n_runs_r0"] = (lambda h, n: (h, str(n)))(*raw_hours(R / "r0_932be94" / "raw"))
    try:
        c["cpu_h_spent_r12"] = f"{float(c['cpu_h_total']) + float(c['cpu_h_superseded']):.1f}"
    except ValueError:
        c["cpu_h_spent_r12"] = "n/a"
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
