# DRM recursion-curriculum microbench (CPU toy)

A small, CPU-only reproduction of the core idea of **Denoising Recursion Models** (DRM): corrupt the target, then train a shared recursive block to recover it over **k recursive steps** rather than in one jump. It compares DRM with **one-step denoising** (diffusion-style "forward" training) and with **backward recursion from a full mask** (TRM-style TBPTT), all on the same block Phi, at matched parameter count and matched training FLOPs.

> Credit: Cameron, Wang, Ivanov, Bhattacharyya, Chételat and Zhang, *One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models*, Huawei Technologies, University of Toronto and University of Waterloo, [arXiv:2604.18839](https://arxiv.org/html/2604.18839v1). Official code: [wwwwwwwwz/DenoisingRecursionModels](https://github.com/wwwwwwwwz/DenoisingRecursionModels). This repo is an independent didactic toy on 6×6 grids. It makes no ARC-AGI or SOTA claim.

## TL;DR

- **DRM beats one-step denoising on every eval task at matched parameters and matched training FLOPs.** Pooled pass@1 is {{p1_pooled_drm}} for DRM against {{p1_pooled_fwd}} for the FLOP-matched one-step denoiser ({{d_pooled_drm_vs_fwd}}, {{wl_pooled_drm_vs_fwd}} seeds won/lost, Holm p = {{ph_pooled_drm_vs_fwd}}). This is the deliverable in the brief, and it holds per task: shift {{d_shift_drm_vs_fwd}}, gravity {{d_gravity_drm_vs_fwd}}, flood {{d_flood_drm_vs_fwd}}.
- **The noise curriculum matters.** DRM is the same k = 4 recursion as `bwd_k` but starts from a corrupted target instead of a full mask. It beats `bwd_k` by {{d_pooled_drm_vs_bwd_k}} pooled.
- **DRM does not beat TRM-style backward TBPTT overall.** Pooled it is {{p1_pooled_drm}} against {{p1_pooled_bwd_tbptt}} ({{d_pooled_drm_vs_bwd_tbptt}}). DRM is better on flood ({{d_flood_drm_vs_bwd_tbptt}}) and shift ({{d_shift_drm_vs_bwd_tbptt}}), but clearly worse on gravity ({{d_gravity_drm_vs_bwd_tbptt}}). All three differences survive Holm correction. TBPTT here gets 4 windows (16 recursive steps) per batch for the same FLOPs, which is the regime where the paper's App. F.8 also reports TRM ahead (Sudoku).
- **The one-step denoiser's train-test mismatch is directly visible.** It essentially never corrects a wrong visible cell: self-correction is {{mm_flood_fwd_selfcorr}}% on flood and {{mm_gravity_fwd_selfcorr}}% on gravity and shift, because in training every visible cell is correct. Along its own trajectory, flood cell accuracy stays at {{mm_flood_fwd_own_last_cell}}% after 16 steps, while the same model fed noised clean targets reaches {{mm_flood_fwd_oracle_last_cell}}%.

All numbers are eval-split pass@1 (whole-grid exact match) over **{{n_eval_seeds}} fresh training seeds**, at a matched test compute of 16 Phi applications, with the inference schedule chosen on dev. Intervals are 95% seed-bootstrap CIs, and p-values are exact paired sign-flip tests with Holm correction over all {{holm_m}} comparisons. The model has {{n_params}} parameters, the code hash is `{{code_hash}}`, and there are {{n_tests}} tests with {{mut_killed}}/{{mut_total}} planted bugs caught (see [Verification](#verification)).

![Grouped bar chart of eval pass@1 for five trainers on shift, gravity, flood and their pooled mean. DRM {{p1m_pooled_drm}}%, backward TBPTT {{p1m_pooled_bwd_tbptt}}%, one-step denoising FLOP-matched {{p1m_pooled_fwd}}%, step-matched {{p1m_pooled_fwd_steps}}%, single-window backward {{p1m_pooled_bwd_k}}% pooled.](figures/primary.png)

*[figures/primary.png](figures/primary.png): data in [results/summary.json](results/summary.json) (`primary`) and [results/tables/primary.csv](results/tables/primary.csv); script [scripts/make_figures.py](scripts/make_figures.py).*

## What was compared

```mermaid
flowchart TB
  subgraph A["(a) one-step denoising: fwd"]
    direction LR
    Y1[clean target Y] -->|"mask cells, tau ~ U(0,1), cosine"| N1[corrupted Y~]
    N1 --> P1["Phi x1"] --> L1[CE to Y]
  end
  subgraph B["(b) backward TBPTT: bwd_tbptt"]
    direction LR
    M2[all MASK] --> P2["Phi x k"] --> L2a[CE] --> S2["stop-grad"] --> P2b["Phi x k"] --> L2b["CE ... (4 windows)"]
  end
  subgraph C["(c) DRM: drm"]
    direction LR
    Y3[clean target Y] -->|"mask cells, tau ~ U(0,1)"| N3[corrupted Y~]
    N3 --> P3["Phi x k (backprop through all k)"] --> L3[CE to Y]
  end
  A ~~~ B ~~~ C
```

| id | what | start state | Phi per opt step | opt steps | training Phi |
|---|---|---|---|---|---|
| `fwd` | one-step denoising (paper Eq. 3), **FLOP-matched** | corrupted target | 1 | 12,000 | 12,000 |
| `fwd_steps` | one-step denoising, **step-matched** | corrupted target | 1 | 3,000 | 3,000 |
| `bwd_k` | backward recursion, one window of k = 4 (= DRM with tau = 1) | full mask | 4 | 3,000 | 12,000 |
| `bwd_tbptt` | backward TBPTT, 4 windows × k = 4 with stop-gradient (Eq. 4-5, TRM-style) | full mask | 4 | 3,000 | 12,000 |
| `drm` | **DRM** (Eq. 6), k = 4 | corrupted target | 4 | 3,000 | 12,000 |

Every method uses the same block, initialisation (paired by seed), optimiser and batch size. The learning rate and the inference schedule are tuned **per method on dev**, so each baseline gets its best setting. Details are in [docs/method.md](docs/method.md), and the section-by-section mapping to the paper is in [docs/paper_mapping.md](docs/paper_mapping.md).

**Why local attention.** Phi is one transformer layer whose attention only reaches neighbouring cells (radius 1), so information moves at most one cell per Phi step. That gives each task a known minimum recursion depth (shift: 2; gravity and flood: up to the grid size), and a test checks this. This is a design choice, not an empirical finding: I have no committed evidence that global attention would make the tasks one-step solvable (an earlier version of this README claimed it did; see [docs/pilots.md](docs/pilots.md)).

### Inference: the same for every method

```mermaid
flowchart LR
  F["all MASK"] --> E["embed tokens"] --> R["Phi x k_in"] --> D["argmax grid"]
  D -->|"i < S: re-mask floor(r(1 - i/S) * M) random cells"| E
  D -->|"i = S"| O["prediction"]
```

S = 1 is pure latent recursion, which is how a backward-trained model is natively used. Every run is scored on all 25 schedules with S ∈ {1,…,16}, k_in ∈ {1,…,64} and S·k_in ≤ 64, so pure latent recursion (S = 1) is available at every budget up to 64. The headline uses the matched-compute menu S·k_in = 16, picking the schedule that is best on dev for each (task, method).

## Tasks

![Example eval input and target grids for the five tasks: colormap, shift, symfill, gravity, flood.](figures/tasks.png)

*[figures/tasks.png](figures/tasks.png): generated by [drm/tasks.py](drm/tasks.py) through [scripts/make_figures.py](scripts/make_figures.py).*

| task | rule | in eval grid? | dev evidence (best dev pass@1 per method) |
|---|---|---|---|
| `colormap` | per-cell colour permutation (depth 1) | **no, dev only** | every method {{dev_colormap_drm}}% (fwd {{dev_colormap_fwd}}%, bwd-TBPTT {{dev_colormap_bwd_tbptt}}%) |
| `shift` | move objects (+1, +2), no wrap (depth 2) | yes | |
| `symfill` | fill an occluder from the left-right mirror (depth ≤ 5) | **no, dev only** | DRM {{dev_symfill_drm}}%, fwd {{dev_symfill_fwd}}%, fwd-steps {{dev_symfill_fwd_steps}}%, bwd-TBPTT {{dev_symfill_bwd_tbptt}}%, bwd-k {{dev_symfill_bwd_k}}% |
| `gravity` | cells fall to the bottom of their column, keeping order | yes | |
| `flood` | mark cells 4-connected to the source | yes | |

### Scope cut and dev-only evidence (please read)

- The plan was to evaluate all five tasks with 12 eval seeds. At about 10:00 EDT on 2026-10-28 the box CPU slowed about 3.5× (per-run training went from about 140 s to about 510 s). **Before any eval-split run**, and using **dev results only**, I cut the eval grid to **shift, gravity and flood × {{n_eval_seeds}} seeds**, and the k-sweep to flood × 4 seeds. The cut is recorded in [scripts/run_grid.py](scripts/run_grid.py) and [docs/method.md](docs/method.md).
- **colormap** is solved by every method at every learning rate on dev, so it cannot separate them. It serves as a sanity check that every trainer and inference path works.
- **symfill** is near the floor for every method on dev. The only non-trivial cell is `bwd_k` at lr 2e-3, which reached 0.18 on one dev seed and 0.004 on the other. I make **no eval claim** about symfill. Under radius-1 attention it needs precise transport over up to 5 cells, and within this budget none of the trainers learns it.
- The colormap and symfill numbers above are **dev-split numbers from the tuning grid** (2 dev seeds, best lr and best matched schedule), so they are optimistic by construction. They are not comparable to the eval numbers below.
- **Learning-rate edge.** The lr grid is {5e-4, 1e-3, 2e-3} plus 4e-3 on the eval tasks. I added 4e-3 after dev showed the recursive methods peaking at 2e-3. Several (task, method) optima still sit at the 4e-3 edge (see the table below), so every method, baselines included, might gain a little from an even larger lr.

## Results (eval split)

| task | DRM | bwd-TBPTT | bwd-k | fwd (FLOP-matched) | fwd (step-matched) |
|---|---|---|---|---|---|
| shift | {{p1_shift_drm}} | {{p1_shift_bwd_tbptt}} | {{p1_shift_bwd_k}} | {{p1_shift_fwd}} | {{p1_shift_fwd_steps}} |
| gravity | {{p1_gravity_drm}} | {{p1_gravity_bwd_tbptt}} | {{p1_gravity_bwd_k}} | {{p1_gravity_fwd}} | {{p1_gravity_fwd_steps}} |
| flood | {{p1_flood_drm}} | {{p1_flood_bwd_tbptt}} | {{p1_flood_bwd_k}} | {{p1_flood_fwd}} | {{p1_flood_fwd_steps}} |
| **pooled** | **{{p1_pooled_drm}}** | {{p1_pooled_bwd_tbptt}} | {{p1_pooled_bwd_k}} | {{p1_pooled_fwd}} | {{p1_pooled_fwd_steps}} |

Dev-chosen settings (lr; schedule):

| task | DRM | bwd-TBPTT | bwd-k | fwd | fwd-steps |
|---|---|---|---|---|---|
| shift | {{lr_shift_drm}}; {{sched_shift_drm}} | {{lr_shift_bwd_tbptt}}; {{sched_shift_bwd_tbptt}} | {{lr_shift_bwd_k}}; {{sched_shift_bwd_k}} | {{lr_shift_fwd}}; {{sched_shift_fwd}} | {{lr_shift_fwd_steps}}; {{sched_shift_fwd_steps}} |
| gravity | {{lr_gravity_drm}}; {{sched_gravity_drm}} | {{lr_gravity_bwd_tbptt}}; {{sched_gravity_bwd_tbptt}} | {{lr_gravity_bwd_k}}; {{sched_gravity_bwd_k}} | {{lr_gravity_fwd}}; {{sched_gravity_fwd}} | {{lr_gravity_fwd_steps}}; {{sched_gravity_fwd_steps}} |
| flood | {{lr_flood_drm}}; {{sched_flood_drm}} | {{lr_flood_bwd_tbptt}}; {{sched_flood_bwd_tbptt}} | {{lr_flood_bwd_k}}; {{sched_flood_bwd_k}} | {{lr_flood_fwd}}; {{sched_flood_fwd}} | {{lr_flood_fwd_steps}}; {{sched_flood_fwd_steps}} |

### DRM minus each baseline (paired by seed)

| | vs fwd (FLOP) | vs fwd (steps) | vs bwd-k | vs bwd-TBPTT |
|---|---|---|---|---|
| shift | {{d_shift_drm_vs_fwd}} (p_H={{ph_shift_drm_vs_fwd}}) | {{d_shift_drm_vs_fwd_steps}} (p_H={{ph_shift_drm_vs_fwd_steps}}) | {{d_shift_drm_vs_bwd_k}} (p_H={{ph_shift_drm_vs_bwd_k}}) | {{d_shift_drm_vs_bwd_tbptt}} (p_H={{ph_shift_drm_vs_bwd_tbptt}}) |
| gravity | {{d_gravity_drm_vs_fwd}} (p_H={{ph_gravity_drm_vs_fwd}}) | {{d_gravity_drm_vs_fwd_steps}} (p_H={{ph_gravity_drm_vs_fwd_steps}}) | {{d_gravity_drm_vs_bwd_k}} (p_H={{ph_gravity_drm_vs_bwd_k}}) | {{d_gravity_drm_vs_bwd_tbptt}} (p_H={{ph_gravity_drm_vs_bwd_tbptt}}) |
| flood | {{d_flood_drm_vs_fwd}} (p_H={{ph_flood_drm_vs_fwd}}) | {{d_flood_drm_vs_fwd_steps}} (p_H={{ph_flood_drm_vs_fwd_steps}}) | {{d_flood_drm_vs_bwd_k}} (p_H={{ph_flood_drm_vs_bwd_k}}) | {{d_flood_drm_vs_bwd_tbptt}} (p_H={{ph_flood_drm_vs_bwd_tbptt}}) |
| **pooled** | {{d_pooled_drm_vs_fwd}} (p_H={{ph_pooled_drm_vs_fwd}}) | {{d_pooled_drm_vs_fwd_steps}} (p_H={{ph_pooled_drm_vs_fwd_steps}}) | {{d_pooled_drm_vs_bwd_k}} (p_H={{ph_pooled_drm_vs_bwd_k}}) | {{d_pooled_drm_vs_bwd_tbptt}} (p_H={{ph_pooled_drm_vs_bwd_tbptt}}) |

{{n_sig}} of {{holm_m}} comparisons are significant after Holm (p_H < 0.05). With {{n_eval_seeds}} seeds the smallest attainable raw p is 2/1024.

![Forest plot of DRM minus each baseline in eval pass@1, with 95% seed-bootstrap CIs for shift, gravity, flood and pooled. Pooled: vs fwd {{d_pooled_drm_vs_fwd}}, vs bwd-TBPTT {{d_pooled_drm_vs_bwd_tbptt}}. Filled markers are Holm-significant.](figures/forest.png)

*[figures/forest.png](figures/forest.png): data in [results/summary.json](results/summary.json) (`tests`) and [results/tables/tests.csv](results/tables/tests.csv); script [scripts/make_figures.py](scripts/make_figures.py).*

## pass@1 vs test-time compute

![Line plots of eval pass@1 against test-time Phi applications (1 to 64, log scale) for each method on shift, gravity and flood, using the dev-best schedule at each compute level. On flood at 16 Phi: DRM {{cc_flood_drm_16}}%, fwd {{cc_flood_fwd_16}}%, bwd-TBPTT {{cc_flood_bwd_tbptt_16}}%; at 64 Phi: DRM {{cc_flood_drm_64}}%, fwd {{cc_flood_fwd_64}}%, bwd-TBPTT {{cc_flood_bwd_tbptt_64}}%.](figures/compute_curves.png)

*[figures/compute_curves.png](figures/compute_curves.png): data in [results/summary.json](results/summary.json) (`compute_curves`); the schedule per compute level was chosen on dev ([results/tuning.json](results/tuning.json), `by_compute`).*

![Heatmaps of eval pass@1 for every inference schedule, outer steps S by inner steps k_in, for each method (columns) on flood, gravity and shift (rows). The red box marks the schedule chosen on dev.](figures/schedule_heatmaps.png)

*[figures/schedule_heatmaps.png](figures/schedule_heatmaps.png): data in [results/summary.json](results/summary.json) (`schedules`); script [scripts/make_figures.py](scripts/make_figures.py).*

What the compute curves and heatmaps show:

- **One-step denoising does not improve with more test compute.** On flood it peaks at {{cc_flood_fwd_16}}% at 16 Phi and falls to {{cc_flood_fwd_64}}% at 64. On gravity it stays near 0 at every budget.
- **DRM keeps improving past the matched budget on flood**, from {{cc_flood_drm_16}}% at 16 Phi to {{cc_flood_drm_64}}% at 64. TBPTT peaks at 16 Phi ({{cc_flood_bwd_tbptt_16}}%) and degrades at 64 ({{cc_flood_bwd_tbptt_64}}%). So the flood gap to TBPTT widens with test compute.
- **Backward-trained models are tied to their training horizon.** They do well with latent recursion near 16 steps and badly with re-masking (S > 1), because they never saw a visible label token in training. The heatmaps show this as a bright S = 1 row.
- **DRM does well whenever k_in ≥ 4**, its training window, at almost any S, and fails with k_in ≤ 2. That matches its training: k = 4 latent steps that start from a partially visible grid.

## Train-test mismatch of one-step denoising

Under the masking process, every **visible** cell a one-step denoiser sees in training is correct. At test time, the visible cells are its own earlier guesses. The probe below runs S = 16, k_in = 1 from a full mask on flood. At each step it compares accuracy along the model's own trajectory with accuracy when the same step's input is the training distribution (the clean target masked at the same level). It also measures how many wrong visible cells each model fixes.

![Three panels for flood with S=16, k_in=1. Left: cell accuracy per denoising step on the model's own trajectory (solid) versus a noised clean target (dashed), for fwd, DRM and bwd-TBPTT. Middle: fraction of visible input cells that are wrong (fwd averages {{mm_flood_fwd_vis_wrong}}%, DRM {{mm_flood_drm_vis_wrong}}%). Right: fraction of those wrong cells fixed at the next step (fwd {{mm_flood_fwd_selfcorr}}%, DRM {{mm_flood_drm_selfcorr}}%).](figures/mismatch.png)

*[figures/mismatch.png](figures/mismatch.png): data in [results/summary.json](results/summary.json) (`traces.flood`), recorded by [scripts/run_one.py](scripts/run_one.py); script [scripts/make_figures.py](scripts/make_figures.py).*

How to read this probe:

- **One-step denoiser (`fwd`).** The dashed and solid lines diverge: the same model is much better on training-distribution states than on its own states. Its self-correction rate is {{mm_shift_fwd_selfcorr}}% on shift, {{mm_gravity_fwd_selfcorr}}% on gravity and {{mm_flood_fwd_selfcorr}}% on flood. A wrong visible cell is copied, so errors made in the first step are frozen in. This is the train-test mismatch described in paper §1 and §3.1.1, measured directly.
- **DRM.** DRM also starts every training window from correct visible cells. Its token-level self-correction is low too: {{mm_flood_drm_selfcorr}}% on flood, {{mm_shift_drm_selfcorr}}% on shift and {{mm_gravity_drm_selfcorr}}% on gravity. Its advantage comes from the k = 4 latent steps *inside* each denoising step, which are trained on its own intermediate states (the paper's "on-policy" part, §6). It does not come from fixing visible tokens.
- **Backward TBPTT.** TBPTT never sees visible tokens in training. It fixes {{mm_flood_bwd_tbptt_selfcorr}}% of wrong visible cells on flood ({{mm_gravity_bwd_tbptt_selfcorr}}% on gravity), but for the same reason it gains little from noised clean targets. Its dashed line stays within about 2 points of its solid line ({{mm_flood_bwd_tbptt_oracle_last_cell}}% vs {{mm_flood_bwd_tbptt_own_last_cell}}% at the last flood step).
- **Scope.** The probe is measured on S = 16, k_in = 1, which is the one-step denoiser's dev-chosen schedule on every eval task. It is not each model's best schedule.

![Denoising trajectory on one flood eval example for DRM (S=4, k_in=4) and for one-step denoising (S=16, k_in=1), each at its dev-chosen schedule. Each column shows the masked input state (white = MASK) and the predicted grid at one outer step; a tick marks a fully correct prediction.](figures/trajectory.png)

*[figures/trajectory.png](figures/trajectory.png): data in [results/demo/trajectories.json](results/demo/trajectories.json), from [scripts/demo_trajectories.py](scripts/demo_trajectories.py), which retrains the seed-100 models and asserts they match the main-grid pass@1. The example shown is chosen by a fixed rule, the first of the 6 recorded eval examples where DRM ends correct and fwd ends wrong, so it is illustrative and not typical.*

## Training timelines and the k sweep

![Dev pass@1 (S=4, k_in=4, 200 dev examples) against training Phi applications used, for each method on shift, gravity and flood.](figures/learning_curves.png)

*[figures/learning_curves.png](figures/learning_curves.png): data in [results/summary.json](results/summary.json) (`learning_curves`), logged on **dev** examples only during training ([drm/train.py](drm/train.py)).*

![Eval pass@1 on flood against the training recursion window k in {1, 2, 4, 8} at a fixed 12,000-Phi training budget, for DRM and bwd-TBPTT, at the native schedule (S=16/k, k_in=k, solid) and at latent S=1, k_in=16 (dotted). DRM native: k=1 {{ks_flood_drm_k1}}, k=4 {{ks_flood_drm_k4}}, k=8 {{ks_flood_drm_k8}}; bwd-TBPTT latent: k=4 {{ksl_flood_bwd_tbptt_k4}}.](figures/ksweep.png)

*[figures/ksweep.png](figures/ksweep.png): data in [results/summary.json](results/summary.json) (`ksweep`), 4 eval seeds (100-103); script [scripts/make_figures.py](scripts/make_figures.py).*

On flood, at a fixed 12,000-Phi training budget and with 4 seeds, DRM at its native schedule (S = 16/k, k_in = k) gives k = 1 {{ks_flood_drm_k1}}, k = 2 {{ks_flood_drm_k2}}, k = 4 {{ks_flood_drm_k4}} and k = 8 {{ks_flood_drm_k8}}. It rises steeply up to k = 4 and then plateaus. TBPTT is native only in latent mode, where S = 1, k_in = 16 gives k = 1 {{ksl_flood_bwd_tbptt_k1}}, k = 2 {{ksl_flood_bwd_tbptt_k2}}, k = 4 {{ksl_flood_bwd_tbptt_k4}} and k = 8 {{ksl_flood_bwd_tbptt_k8}}. With re-masking it collapses (k = 4: {{ks_flood_bwd_tbptt_k4}}), as in the heatmaps. DRM at k = 1 is one-step denoising, so its low value is the headline effect again. Going from k = 1 to k = 4, DRM gains more than TBPTT does, which is in line with the paper's Table 5. This sweep is exploratory: it has 4 seeds and is not part of the Holm family.

## How this maps to the paper (§3)

| paper claim (§3, §5, App. F) | this toy |
|---|---|
| Forward (one-step) training misaligns train and test, which causes compounding and greedy errors (§1, §3.1.1) | **Reproduced.** One-step denoising never corrects its own visible errors and loses to DRM by {{d_pooled_drm_vs_fwd}} pooled. |
| Masked diffusion without recursion fails badly (Table 1: about 0%) | Consistent: `fwd` reaches {{p1m_gravity_fwd}}% on gravity and {{p1m_shift_fwd}}% on shift. |
| DRM's noise curriculum helps compared with recursion from noise without it (§3.2, §6) | **Reproduced** against `bwd_k`, a single-window backward recursion at the same k: {{d_pooled_drm_vs_bwd_k}}. |
| DRM beats TRM (Table 1) | **Not reproduced overall.** DRM wins flood and shift, but loses gravity and the pooled mean to TBPTT ({{d_pooled_drm_vs_bwd_tbptt}}). There is no scratchpad, warm-up loop or halting here, and TBPTT gets 4 windows per batch. |
| DRM gains more per recursive step k than TRM (Table 5) | Consistent on flood from k = 1 to 4, and DRM plateaus at k = 8. Exploratory, 4 seeds. |

See [docs/paper_mapping.md](docs/paper_mapping.md) for the equation-by-equation mapping and for the simplifications. None of the following are used, for any method: scratchpad z, warm-up loops, Q-head and halting, EMA, task embedding, augmentation and voting, or SPRM.

## Pipeline and reproduction

```mermaid
flowchart TD
  T["run_grid.py tune<br/>dev split, seeds 1-2, 8 lrs x 5 tasks x 5 methods"] --> U["tune.py --fail-on-edge<br/>lr + matched schedule per task x method"]
  U --> G["git commit of dev outputs<br/>(before any eval run)"]
  G --> M["run_grid.py main<br/>eval split, 5 tasks, seeds 100-111"]
  G --> K["run_grid.py ksweep<br/>gravity, flood; k in 1,2,8; seeds 100-103"]
  M --> A["aggregate.py<br/>guards: complete, no extras, one code hash"]
  K --> A
  A --> D["demo_trajectories.py<br/>retrains seed-100 flood models"]
  A --> F["make_figures.py"]
  D --> F
  A --> C["claims.py -> claim_numbers.json"]
  C --> R["render_readme.py -> README.md"]
```

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2 pandas==3.1.0 pytest==9.1.1
python -m pytest -q tests                  # fast; uses tiny configs
WORKERS=7 sh scripts/pipeline.sh            # whole chain, foreground; skips up-to-date runs
python scripts/claims.py && python scripts/render_readme.py
python scripts/mutation_check.py --workers 2
```

`pipeline.sh` runs each stage to completion before starting the next. `tune.py` and `aggregate.py` exit 2 on an incomplete, stale, extra-file or mixed-code grid. One run takes about {{train_s_drm}} s of training on one core of this box, so the full chain is roughly 4-6 CPU-hours.

## Verification

- **{{n_tests}} tests** cover the pipeline wiring end to end, not just helpers. Examples: training Phi counts and budgets per method; DRM and fwd starting from corrupted targets while bwd starts from a full mask; TBPTT detaching between windows; loss targets being clean labels; the remask schedule; eval metrics using the eval split while the learning curve uses dev; the main grid using tuned lrs and eval seeds; tuning choosing only from the matched-compute menu; and aggregation pairing by seed, applying one Holm family, and refusing partial, extra or stale grids.
- **Mutation check** ([scripts/mutation_check.py](scripts/mutation_check.py), [results/mutation.csv](results/mutation.csv)): an unmutated control run must pass first, and a mutant counts as killed only when pytest exits 1. **{{mut_killed}}/{{mut_total}} killed.**
- **Provenance:** every raw result stores the code hash (`{{code_hash}}`) and its config hash. Results are deterministic per seed, which the trajectory demo re-checks.

## Limitations

- This is a toy: 6×6 grids, about 52k parameters, one model per task, no augmentation. Effects here say nothing direct about ARC-AGI.
- The scope cut (colormap and symfill dev-only, 10 seeds, k-sweep on flood only) is described above.
- Several dev optima sit at the 4e-3 edge of the lr grid.
- The seed is the replication unit. CIs resample seeds only, and the 1,000 eval examples are shared across seeds.
- Inference uses evenly spaced remask levels and uniform remasking, with no confidence-based remasking.
