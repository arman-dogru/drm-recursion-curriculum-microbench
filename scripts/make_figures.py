"""All README figures, from committed result files only.

Inputs: results/summary.json, results/tuning.json, results/demo/trajectories.json,
drm.tasks generators. Outputs: figures/*.png (deterministic: no timestamps).
Refuses (exit 2) on a partial summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drm.config import K_GRID, METHODS, ROOT, S_GRID  # noqa: E402
from drm.tasks import H, TASKS, W, make_task_data  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import MAIN_TASKS  # noqa: E402

FIG = ROOT / "figures"
LABEL = {"fwd": "one-step denoise (FLOP-matched)", "fwd_steps": "one-step denoise (step-matched)",
         "bwd_k": "backward, 1 window", "bwd_tbptt": "backward TBPTT (TRM-style)", "drm": "DRM"}
SHORT = {"fwd": "fwd", "fwd_steps": "fwd-steps", "bwd_k": "bwd-k", "bwd_tbptt": "bwd-TBPTT", "drm": "DRM"}
COL = {"fwd": "#4c72b0", "fwd_steps": "#8fa8d4", "bwd_k": "#dd8452", "bwd_tbptt": "#c44e52", "drm": "#55a868"}
GRID_CMAP = ListedColormap(["#111111", "#1e93ff", "#f93c31", "#4fcc30", "#ffdc00", "#999999", "#e53aa3",
                            "#ff851b", "#ffffff"])   # 8 colours + MASK (white)
META = {"Software": None}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=110, bbox_inches="tight", metadata=META)
    plt.close(fig)


def fig_primary(s):
    groups = list(MAIN_TASKS) + ["pooled", "pooled3"]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.16
    for i, m in enumerate(METHODS):
        r = [s["primary"][g][m] for g in groups]
        x = np.arange(len(groups)) + (i - 2) * w
        ax.bar(x, [v["mean"] for v in r], w, color=COL[m], label=LABEL[m],
               yerr=[[v["mean"] - v["lo"] for v in r], [v["hi"] - v["mean"] for v in r]], capsize=2)
    names = {"pooled": "pooled\n(all 5, primary)", "pooled3": "pooled3\n(shift+gravity+flood,\ndescriptive)"}
    ax.set_xticks(range(len(groups)), [names.get(g, g) for g in groups])
    ax.axvline(len(MAIN_TASKS) - 0.5, color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("eval pass@1 (exact match)")
    ax.set_ylim(0, 1.05)
    ax.set_title("pass@1 at most matched test compute (S x k_in <= 16 Phi), schedule chosen on dev; 95% seed-bootstrap CI")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    save(fig, "primary.png")


def fig_forest(s):
    tests = s["tests"]
    keys = sorted(tests, key=lambda k: (tests[k]["baseline"], (list(MAIN_TASKS) + ["pooled"]).index(tests[k]["group"])))
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(keys) + 1))
    for i, k in enumerate(keys):
        r = tests[k]
        c = COL[r["baseline"]]
        ax.plot([r["lo"], r["hi"]], [i, i], color=c)
        ax.plot(r["diff"], i, "o" if r["significant"] else "o", color=c, mfc=c if r["significant"] else "white")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(keys)), [f"{tests[k]['group']}: DRM - {SHORT[tests[k]['baseline']]}" for k in keys],
                  fontsize=7)
    ax.set_xlabel("pass@1 difference (DRM minus baseline), 95% CI; filled = Holm p < 0.05")
    ax.invert_yaxis()
    save(fig, "forest.png")


def fig_compute(s):
    fig, axes = plt.subplots(1, len(MAIN_TASKS), figsize=(18, 3.4), sharey=True)
    for ax, t in zip(axes, MAIN_TASKS):
        for m in METHODS:
            cur = s["compute_curves"][t][m]
            cs = sorted(map(int, cur))
            ax.errorbar(cs, [cur[str(c)]["mean"] for c in cs],
                        yerr=[[cur[str(c)]["mean"] - cur[str(c)]["lo"] for c in cs],
                              [cur[str(c)]["hi"] - cur[str(c)]["mean"] for c in cs]],
                        color=COL[m], marker=None, label=SHORT[m], capsize=2)
            lat = [c for c in cs if cur[str(c)]["schedule"][0] == 1]
            rem = [c for c in cs if cur[str(c)]["schedule"][0] != 1]
            ax.plot(lat, [cur[str(c)]["mean"] for c in lat], "o", ms=4, color=COL[m])
            ax.plot(rem, [cur[str(c)]["mean"] for c in rem], "o", ms=4, mfc="white", color=COL[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        ax.set_title(t)
        ax.set_xlabel("test-time Phi applications (S x k_in)")
    axes[0].set_ylabel("eval pass@1")
    axes[0].legend(fontsize=7)
    fig.suptitle("dev-best schedule at each budget, any S x k_in = c (filled = latent S=1, hollow = re-masking S>1); "
                 "lr fixed at the headline dev choice; 95% seed-bootstrap CI", fontsize=9)
    save(fig, "compute_curves.png")


def fig_heatmaps(s, tasks=("flood", "gravity", "shift", "symfill", "colormap")):
    fig, axes = plt.subplots(len(tasks), len(METHODS), figsize=(19, 2.5 * len(tasks)))
    for r, t in enumerate(tasks):
        for c, m in enumerate(METHODS):
            ax = axes[r, c]
            tab = s["schedules"][t][m]
            A = np.full((len(S_GRID), len(K_GRID)), np.nan)
            for i, S in enumerate(S_GRID):
                for j, k in enumerate(K_GRID):
                    if f"S{S}_k{k}" in tab:
                        A[i, j] = tab[f"S{S}_k{k}"]
            ax.imshow(A, vmin=0, vmax=1, cmap="viridis", origin="lower")
            for i in range(len(S_GRID)):
                for j in range(len(K_GRID)):
                    if not np.isnan(A[i, j]):
                        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5,
                                color="w" if A[i, j] < 0.5 else "k")
            ch = s["tuning"][t][m]["schedule"]
            ax.add_patch(plt.Rectangle((K_GRID.index(ch[1]) - .5, S_GRID.index(ch[0]) - .5), 1, 1,
                                       fill=False, ec="red", lw=2))
            ax.set_xticks(range(len(K_GRID)), K_GRID, fontsize=7)
            ax.set_yticks(range(len(S_GRID)), S_GRID, fontsize=7)
            if r == len(tasks) - 1:
                ax.set_xlabel("inner steps k_in")
            if c == 0:
                ax.set_ylabel(f"{t}\nouter steps S")
            if r == 0:
                ax.set_title(SHORT[m])
    fig.suptitle(f"eval pass@1 for every inference schedule (mean over {len(s['meta']['eval_seeds'])} seeds); "
                 "red box = schedule chosen on dev from the S x k_in <= 16 menu")
    save(fig, "schedule_heatmaps.png")


def fig_mismatch(s, task="flood"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.0))
    tr = s["traces"][task]
    for m in ("fwd", "drm", "bwd_tbptt"):
        own, ora = tr[m]["S16_k1"]["own"], tr[m]["S16_k1"]["oracle"]
        st = [r["step"] for r in own]
        axes[0].plot(st, [r["cell_acc"] for r in own], color=COL[m], label=f"{SHORT[m]}: own trajectory")
        axes[0].plot(st, [r["cell_acc"] for r in ora], color=COL[m], ls="--", label=f"{SHORT[m]}: noised target")
        axes[1].plot(st[1:], [r["visible_wrong_frac"] for r in own][1:], color=COL[m], label=SHORT[m])
        sc = [r["self_correct_rate"] for r in own][1:]
        axes[2].plot(st[1:], [np.nan if v is None else v for v in sc], color=COL[m], label=SHORT[m])
    axes[0].set_title(f"{task}, S=16, k_in=1: cell accuracy", fontsize=10)
    axes[0].set_ylabel("cell accuracy")
    axes[1].set_title("visible input cells that are wrong", fontsize=10)
    axes[2].set_title("wrong visible cells fixed at the next step", fontsize=10)
    for ax in axes:
        ax.set_xlabel("outer denoising step")
        ax.legend(fontsize=7)
    save(fig, "mismatch.png")


def fig_mismatch_own(s, methods=("fwd", "drm", "bwd_tbptt")):
    """Last-step cell accuracy on own trajectory vs oracle (noised clean target), each method at its own
    dev-chosen schedule (review m1)."""
    tac = s["traces_at_choice"]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    w = 0.13
    for i, m in enumerate(methods):
        for j, kind in enumerate(("own", "oracle")):
            x = np.arange(len(MAIN_TASKS)) + (2 * i + j - 2.5) * w
            v = [tac[t][m][kind][-1]["cell_acc"] for t in MAIN_TASKS]
            ax.bar(x, v, w, color=COL[m], alpha=1.0 if kind == "own" else 0.45, hatch=None if kind == "own" else "//",
                   label=f"{SHORT[m]}: {'own trajectory' if kind == 'own' else 'noised clean target'}")
            if kind == "own":       # review r2 n3: one short vertical schedule label per method pair
                for xi, t in zip(x, MAIN_TASKS):
                    sch = tac[t][m]["schedule"]
                    ax.text(xi + w / 2, 1.02, f"S{sch[0]} k{sch[1]}", rotation=90, ha="center", va="bottom",
                            fontsize=6.5, color=COL[m])
    ax.set_xticks(range(len(MAIN_TASKS)), MAIN_TASKS, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("cell accuracy at the last outer step")
    ax.set_title("own-trajectory vs training-distribution input, each method at its own dev-best 16-Phi schedule "
                 "(label above each pair)", fontsize=10)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    save(fig, "mismatch_own_schedule.png")


def fig_learning(s):
    fig, axes = plt.subplots(1, len(MAIN_TASKS), figsize=(18, 3.0), sharey=True)
    for ax, t in zip(axes, MAIN_TASKS):
        for m in METHODS:
            lc = s["learning_curves"][t][m]
            ax.plot([p["phi_used"] for p in lc], [p["dev_pass1"] for p in lc], color=COL[m], marker=".",
                    label=SHORT[m])
        ax.set_title(t)
        ax.set_xlabel("training Phi applications used")
    axes[0].set_ylabel("dev pass@1 (S=4, k_in=4, 200 ex.)")
    axes[0].legend(fontsize=7)
    save(fig, "learning_curves.png")


def fig_ksweep(s):
    ks = s["ksweep"]
    tasks = sorted(ks)
    fig, axes = plt.subplots(1, len(tasks), figsize=(10, 3.3), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), tasks):
        for m in sorted(ks[t]):
            kk = sorted(ks[t][m], key=int)
            kinds = [("native", "-", "S=16/k, k_in=k"), ("latent16", ":", "fixed latent S=1, k_in=16")]
            if m == "bwd_tbptt":     # review r2 m3: latent at its training horizon n_sup*k
                kinds.append(("horizon", "--", "horizon latent S=1, k_in=4k"))
            for kind, ls, lab in kinds:
                v = [ks[t][m][k][kind] for k in kk]
                ax.errorbar([int(k) for k in kk], [x["mean"] for x in v],
                            yerr=[[x["mean"] - x["lo"] for x in v], [x["hi"] - x["mean"] for x in v]],
                            color=COL[m], ls=ls, marker="o", ms=3, capsize=2, label=f"{SHORT[m]} ({lab})")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_title(t)
        ax.set_xlabel("training recursion window k (FLOP-matched)")
    np.atleast_1d(axes)[0].set_ylabel("eval pass@1")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    save(fig, "ksweep.png")


def fig_tasks():
    fig, axes = plt.subplots(2, len(TASKS), figsize=(12, 4.6))
    for j, t in enumerate(TASKS):
        X, Y = make_task_data(t, 1, 1, 1)["eval"]
        for i, (g, name) in enumerate(((X[0], "input"), (Y[0], "target"))):
            axes[i, j].imshow(g.reshape(H, W), cmap=GRID_CMAP, vmin=0, vmax=8)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_title(f"{t}: {name}", fontsize=9)
    save(fig, "tasks.png")


def pick_example(demo):
    """First shown example where DRM's final grid is right and fwd's is wrong (else 0)."""
    d, f = demo["methods"]["drm"], demo["methods"]["fwd"]
    for i in range(len(d["y"])):
        if d["steps"][-1]["pred"][i] == d["y"][i] and f["steps"][-1]["pred"][i] != f["y"][i]:
            return i
    return 0


def fig_trajectory(demo, ex=None):
    ex = pick_example(demo) if ex is None else ex
    ms = list(demo["methods"])
    nmax = max(len(demo["methods"][m]["steps"]) for m in ms)
    fig, axes = plt.subplots(2 * len(ms), nmax + 2, figsize=(1.3 * (nmax + 2), 2.7 * len(ms)))
    for r, m in enumerate(ms):
        d = demo["methods"][m]
        for row in (2 * r, 2 * r + 1):
            for a in axes[row]:
                a.axis("off")
        axes[2 * r, 0].imshow(np.array(d["x"][ex]).reshape(H, W), cmap=GRID_CMAP, vmin=0, vmax=8)
        axes[2 * r, 0].set_title("input", fontsize=8)
        axes[2 * r, 1].imshow(np.array(d["y"][ex]).reshape(H, W), cmap=GRID_CMAP, vmin=0, vmax=8)
        axes[2 * r, 1].set_title("target", fontsize=8)
        for i, st in enumerate(d["steps"]):
            ok = st["pred"][ex] == d["y"][ex]
            axes[2 * r, i + 2].imshow(np.array(st["input"][ex]).reshape(H, W), cmap=GRID_CMAP, vmin=0, vmax=8)
            axes[2 * r, i + 2].set_title(f"step {i + 1} in", fontsize=7)
            axes[2 * r + 1, i + 2].imshow(np.array(st["pred"][ex]).reshape(H, W), cmap=GRID_CMAP, vmin=0, vmax=8)
            axes[2 * r + 1, i + 2].set_title("pred" + (" ✓" if ok else ""), fontsize=7)
        S, k = d["schedule"]
        final_ok = d["steps"][-1]["pred"][ex] == d["y"][ex]
        axes[2 * r + 1, 0].text(0, 0.5, f"{SHORT[m]}\nS={S}, k_in={k}\nfinal: {'correct' if final_ok else 'wrong'}",
                                fontsize=10, transform=axes[2 * r + 1, 0].transAxes)
    save(fig, "trajectory.png")


def main(results: Path = ROOT / "results", fig_dir: Path | None = None):
    global FIG
    if fig_dir is not None:
        FIG = fig_dir
    s = json.loads((results / "summary.json").read_text())
    if s["meta"].get("partial"):
        print("make_figures: refusing: summary.json is partial", file=sys.stderr)
        sys.exit(2)
    fig_primary(s)
    fig_forest(s)
    fig_compute(s)
    fig_heatmaps(s)
    fig_mismatch(s)
    fig_mismatch_own(s)
    fig_learning(s)
    fig_ksweep(s)
    fig_tasks()
    demo = results / "demo" / "trajectories.json"
    if demo.exists():
        fig_trajectory(json.loads(demo.read_text()))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
