# DRM recursion-curriculum microbench (CPU toy)

A small, CPU-only reproduction of the core idea of **Denoising Recursion Models** (DRM): corrupt the target, then train a shared recursive block to recover it over **k recursive steps** rather than in one jump. It compares DRM with **one-step denoising** (diffusion-style "forward" training) and with **backward recursion from a full mask** (TRM-style TBPTT), all on the same block Phi, at matched parameter count and matched training FLOPs.

> Credit: Cameron, Wang, Ivanov, Bhattacharyya, Chételat and Zhang, *One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models*, Huawei Technologies, University of Toronto and University of Waterloo, [arXiv:2604.18839](https://arxiv.org/html/2604.18839v1). Official code: [wwwwwwwwz/DenoisingRecursionModels](https://github.com/wwwwwwwwz/DenoisingRecursionModels). This repo is an independent didactic toy on 6×6 grids. It makes no ARC-AGI or SOTA claim.

## TL;DR

> **Round 1 (after review).** The whole pipeline was rerun under a new code hash: all **5 tasks** × **12 eval seeds** (the round-0 task cut is reverted), an lr grid of 5e-4 … 6.4e-2 for **every** task and method, and pure latent inference up to 64 steps. Round-0 artifacts are kept in [results/r0_932be94/](results/r0_932be94/). What changed is summarised in [Round 1 changes](#round-1-changes).

- **DRM beats one-step denoising, but on 3 of the 5 tasks, not all.** Pooled over all five tasks, pass@1 is 71.2% [69.8, 72.6] for DRM against 29.1% [28.9, 29.3] for the FLOP-matched one-step denoiser (+42.1 pp [+40.6, +43.5], 12/0 seeds won/lost, Holm p = 0.012). The gain comes from shift (+93.6 pp [+93.4, +93.9]), gravity (+76.7 pp [+69.4, +83.8]) and flood (+40.1 pp [+38.3, +41.9]). On colormap every method scores 100%, and on symfill DRM and fwd are both near 0 (0.2% vs 0.2%, +0.0 pp [-0.1, +0.1]).
- **The noise curriculum helps on average, not everywhere.** DRM is the same k = 4 recursion as `bwd_k` but starts from a corrupted target instead of a full mask. Pooled it is ahead by +10.7 pp [+8.5, +12.9] (shift +8.6 pp [+3.2, +15.8], gravity +48.0 pp [+38.6, +56.8]; flood +13.2 pp [+3.5, +26.0], not significant after Holm), but it is **behind** on symfill (-16.4 pp [-20.1, -12.1]).
- **DRM loses to TRM-style backward TBPTT overall.** Pooled it is 71.2% [69.8, 72.6] against 76.7% [75.7, 77.6] (-5.5 pp [-6.8, -4.0], Holm-significant). DRM is ahead on flood (+10.3 pp [+7.7, +13.2]) and slightly on shift (+1.4 pp [+1.2, +1.7]), and behind on gravity (-19.0 pp [-26.2, -12.1]) and symfill (-20.1 pp [-24.2, -14.1]). TBPTT here gets 4 windows (16 recursive steps) per batch for the same FLOPs.
- **With more test compute, DRM keeps improving on flood and TBPTT does not**, now that both can use their best inference mode at every budget: from 16 to 64 Phi, DRM goes 79.2% → 86.8% (S=16, k_in=4) and latent TBPTT 68.9% → 60.7% (S=1, k_in=64). On gravity both are flat (DRM 77.4%, TBPTT 92.3% at 64 Phi).
- **A train-test-mismatch symptom shows up for one-step denoising**, but this probe does not prove it is the cause. At each method's own dev schedule, flood cell accuracy on its own trajectory vs on noised clean targets is 87.0% vs 100.0% for fwd, and 96.9% vs 99.9% for DRM.

All numbers are eval-split pass@1 (whole-grid exact match) over **12 fresh training seeds** (100-111), at a matched test compute of 16 Phi applications, with the lr and the inference schedule chosen on dev for each (task, method). Intervals are 95% percentile seed-bootstrap CIs (they under-cover a little at n = 12; Student-t intervals are in [results/tables/](results/tables/)). p-values are exact paired sign-flip tests, Holm-corrected over all 24 comparisons. The model has 52,488 parameters, the code hash is `91de577ed2a30aae`, and there are 102 tests with 92/92 planted bugs caught (see [Verification](#verification)). Everything ran on CPU; no GPU was used.

![Grouped bar chart of eval pass@1 for five trainers on all five tasks, the pooled mean over all five (primary) and the descriptive pooled mean over shift, gravity and flood. Pooled over all five: DRM 71.2%, backward TBPTT 76.7%, single-window backward 60.5%, one-step denoising FLOP-matched 29.1%, step-matched 28.4%.](figures/primary.png)

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

| task | rule | min. Phi steps (radius-1 attention) |
|---|---|---|
| `colormap` | per-cell colour permutation | 1 |
| `shift` | move objects (+1, +2), no wrap | 2 |
| `symfill` | fill an occluder from the left-right mirror | up to 5 |
| `gravity` | cells fall to the bottom of their column, keeping order | up to 5 |
| `flood` | mark cells 4-connected to the source | path length |

All five tasks are in the eval grid. `colormap` needs no recursion and acts as a sanity check (every method should solve it).

### Round 1 changes

- **All 5 tasks, 12 seeds (M1).** Round 0 dropped colormap and symfill from eval after looking at dev, and justified it with a "3.5× CPU slowdown". The logs show short bursts of slow jobs, not a sustained slowdown, and the two dropped tasks were the ones where DRM did not win on dev. The cut is reverted. The full story is in [docs/method.md](docs/method.md#round-0-scope-cut-reverted).
- **lr grid (M2).** Every (task, method) gets the same grid {5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2}. The last two points were added on dev evidence, for every cell, before any eval run; 6.4e-2 was declared the last extension. Cells whose dev optimum is still at an edge: flood:fwd_steps:0.064, gravity:bwd_k:0.064, gravity:fwd:0.0005, shift:bwd_k:0.064.
- **Inference (M3).** Pure latent recursion (S = 1) is now offered at 32 and 64 Phi, so at every budget each method uses its dev-best mode (latent or re-masking).
- **Provenance (m4).** Dev outputs and `tuning.json` were committed (3d7f9c4, 18:18 EDT) and pushed before any eval run, and every raw result records start and end timestamps, load average and wall time.
- **Smaller items:** collision-free run names and input validation (m7), lock file and no hard-coded interpreter (m5), softened pilot claim (m6), DRM shown next to fwd in the mismatch probe (m1), corrected CPU-hours (m2), more mutants (m3), and the nits. See the PR thread for a per-item list.

## Results (eval split)

| task | DRM | bwd-TBPTT | bwd-k | fwd (FLOP-matched) | fwd (step-matched) |
|---|---|---|---|---|---|
| colormap | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| shift | 99.9% [99.8, 100.0] | 98.5% [98.2, 98.7] | 91.3% [84.2, 96.7] | 6.3% [6.0, 6.5] | 6.3% [6.1, 6.5] |
| symfill | 0.2% [0.2, 0.3] | 20.4% [14.4, 24.5] | 16.6% [12.3, 20.3] | 0.2% [0.1, 0.3] | 0.3% [0.2, 0.4] |
| gravity | 76.8% [69.5, 83.8] | 95.8% [95.4, 96.1] | 28.8% [23.0, 34.9] | 0.1% [0.1, 0.1] | 0.1% [0.0, 0.1] |
| flood | 79.2% [77.3, 81.2] | 68.9% [66.5, 71.4] | 66.0% [54.4, 74.7] | 39.1% [38.2, 40.1] | 35.4% [34.6, 36.2] |
| **pooled, all 5 (primary)** | **71.2% [69.8, 72.6]** | 76.7% [75.7, 77.6] | 60.5% [58.4, 62.6] | 29.1% [28.9, 29.3] | 28.4% [28.2, 28.6] |
| pooled3: shift, gravity, flood (descriptive) | 85.3% [83.0, 87.5] | 87.7% [86.9, 88.6] | 62.0% [58.6, 65.3] | 15.2% [14.8, 15.5] | 13.9% [13.6, 14.2] |

Dev-chosen settings (lr; schedule):

| task | DRM | bwd-TBPTT | bwd-k | fwd | fwd-steps |
|---|---|---|---|---|---|
| colormap | 0.0005; S=1, k_in=16 | 0.0005; S=1, k_in=16 | 0.0005; S=1, k_in=16 | 0.0005; S=1, k_in=16 | 0.0005; S=1, k_in=16 |
| shift | 0.016; S=4, k_in=4 | 0.002; S=1, k_in=16 | 0.064; S=4, k_in=4 | 0.032; S=16, k_in=1 | 0.016; S=16, k_in=1 |
| symfill | 0.032; S=1, k_in=16 | 0.032; S=1, k_in=16 | 0.004; S=1, k_in=16 | 0.016; S=16, k_in=1 | 0.001; S=16, k_in=1 |
| gravity | 0.032; S=2, k_in=8 | 0.004; S=1, k_in=16 | 0.064; S=1, k_in=16 | 0.0005; S=16, k_in=1 | 0.001; S=16, k_in=1 |
| flood | 0.008; S=2, k_in=8 | 0.002; S=1, k_in=16 | 0.016; S=2, k_in=8 | 0.004; S=16, k_in=1 | 0.064; S=16, k_in=1 |

On colormap every setting reaches 100% on dev, so the tie-break (first lr in grid order) picks 5e-4 for everyone.

### DRM minus each baseline (paired by seed)

| | vs fwd (FLOP) | vs fwd (steps) | vs bwd-k | vs bwd-TBPTT |
|---|---|---|---|---|
| colormap | +0.0 pp [+0.0, +0.0] (p_H=1) | +0.0 pp [+0.0, +0.0] (p_H=1) | +0.0 pp [+0.0, +0.0] (p_H=1) | +0.0 pp [+0.0, +0.0] (p_H=1) |
| shift | +93.6 pp [+93.4, +93.9] (p_H=0.012) | +93.6 pp [+93.5, +93.8] (p_H=0.012) | +8.6 pp [+3.2, +15.8] (p_H=0.012) | +1.4 pp [+1.2, +1.7] (p_H=0.012) |
| symfill | +0.0 pp [-0.1, +0.1] (p_H=1) | -0.1 pp [-0.2, +0.1] (p_H=1) | -16.4 pp [-20.1, -12.1] (p_H=0.012) | -20.1 pp [-24.2, -14.1] (p_H=0.012) |
| gravity | +76.7 pp [+69.4, +83.8] (p_H=0.012) | +76.7 pp [+69.5, +83.8] (p_H=0.012) | +48.0 pp [+38.6, +56.8] (p_H=0.012) | -19.0 pp [-26.2, -12.1] (p_H=0.012) |
| flood | +40.1 pp [+38.3, +41.9] (p_H=0.012) | +43.8 pp [+41.8, +45.8] (p_H=0.012) | +13.2 pp [+3.5, +26.0] (p_H=0.1) | +10.3 pp [+7.7, +13.2] (p_H=0.012) |
| **pooled** | +42.1 pp [+40.6, +43.5] (p_H=0.012) | +42.8 pp [+41.3, +44.2] (p_H=0.012) | +10.7 pp [+8.5, +12.9] (p_H=0.012) | -5.5 pp [-6.8, -4.0] (p_H=0.012) |

17 of 24 comparisons are significant after Holm (p_H < 0.05). With 12 seeds the smallest attainable raw p is 2/4096. On colormap every difference is exactly 0 (p = 1).

![Forest plot of DRM minus each baseline in eval pass@1, with 95% seed-bootstrap CIs for all five tasks and the pooled mean. Pooled: vs fwd +42.1 pp [+40.6, +43.5], vs bwd-k +10.7 pp [+8.5, +12.9], vs bwd-TBPTT -5.5 pp [-6.8, -4.0]. Filled markers are Holm-significant.](figures/forest.png)

*[figures/forest.png](figures/forest.png): data in [results/summary.json](results/summary.json) (`tests`) and [results/tables/tests.csv](results/tables/tests.csv); script [scripts/make_figures.py](scripts/make_figures.py).*

**Round 0 vs round 1 (pooled, DRM minus baseline).** Round 0 pooled only shift, gravity and flood with 10 seeds and a narrower lr grid: vs fwd +67.4 pp [+65.0, +69.7], vs bwd-k +42.2 pp [+37.8, +46.6], vs bwd-TBPTT -5.5 pp [-7.8, -3.4]. The comparable round-1 number is pooled3 above; the primary round-1 number is the 5-task pool.

## pass@1 vs test-time compute

![Line plots of eval pass@1 against test-time Phi applications (1 to 64, log scale) for each method on all five tasks, using the dev-best schedule at each compute level among all schedules with that budget. Filled markers are latent schedules (S=1), hollow markers re-masking (S>1). On flood at 64 Phi: DRM 86.8% (S=16, k_in=4), bwd-TBPTT 60.7% (S=1, k_in=64), fwd 19.3%. On gravity at 64 Phi: DRM 77.4%, bwd-TBPTT 92.3% (S=1, k_in=64).](figures/compute_curves.png)

*[figures/compute_curves.png](figures/compute_curves.png): data in [results/summary.json](results/summary.json) (`compute_curves`, which also records the schedule used at each point); the schedule per compute level was chosen on dev ([results/tuning.json](results/tuning.json), `by_compute`).*

![Heatmaps of eval pass@1 for all 25 inference schedules, outer steps S by inner steps k_in up to 64, for each method (columns) and task (rows). The red box marks the schedule chosen on dev from the 16-Phi menu.](figures/schedule_heatmaps.png)

*[figures/schedule_heatmaps.png](figures/schedule_heatmaps.png): data in [results/summary.json](results/summary.json) (`schedules`); script [scripts/make_figures.py](scripts/make_figures.py).*

What the compute curves and heatmaps show (each point uses the dev-best schedule at that budget, latent or re-masking):

- **One-step denoising does not use extra test compute.** On flood it peaks at 39.1% at 16 Phi and falls to 19.3% at 64; on gravity and symfill it is near 0 at every budget, and on shift it stays at about 6.3%.
- **DRM keeps improving past the matched budget on flood**, 79.2% at 16 Phi to 84.2% at 32 and 86.8% at 64, using re-masking schedules. On gravity it is flat (76.8% → 77.4%).
- **Backward-trained models are best with latent recursion near their training horizon.** TBPTT now gets pure latent recursion at every budget and the dev picks it every time (S = 1). Going past 16 latent steps it slowly loses accuracy: flood 68.9% → 66.8% → 60.7%, gravity 95.8% → 92.3%. In round 0 the grid capped latent recursion at 16 steps, which forced TBPTT into re-masking above 16 Phi and made it look as if it collapsed; that was an artifact of the grid (review M3). The single-window `bwd_k` degrades faster (gravity 28.8% at 16 Phi, 5.8% at 64).
- **DRM does well whenever k_in ≥ 4**, its training window, at almost any S, and fails with k_in ≤ 2 (heatmaps).

## Train-test mismatch probe

Under the masking process, every **visible** cell a denoiser sees in training is correct. At test time, the visible cells are its own earlier guesses. The probe compares, at each outer step, accuracy along the model's own trajectory with accuracy when the same step's input is the training distribution (the clean target masked at the same level, the "oracle" input). It also measures how many wrong visible cells each model fixes.

![Three panels for flood with S=16, k_in=1 for every method. Left: cell accuracy per denoising step on the model's own trajectory (solid) versus a noised clean target (dashed), for fwd, DRM and bwd-TBPTT. Middle: fraction of visible input cells that are wrong. Right: fraction of those wrong cells fixed at the next step (fwd 0.1%, DRM 4.3%).](figures/mismatch.png)

*[figures/mismatch.png](figures/mismatch.png): data in [results/summary.json](results/summary.json) (`traces.flood.*.S16_k1`), recorded by [scripts/run_one.py](scripts/run_one.py); script [scripts/make_figures.py](scripts/make_figures.py).*

![Grouped bars per task: last-step cell accuracy on the model's own trajectory (solid) and on the noised clean target (hatched), for fwd, DRM and bwd-TBPTT, each at its own dev-chosen schedule. Flood: fwd 87.0% vs 100.0%, DRM 96.9% vs 99.9%.](figures/mismatch_own_schedule.png)

*[figures/mismatch_own_schedule.png](figures/mismatch_own_schedule.png): data in [results/summary.json](results/summary.json) (`traces_at_choice`); script [scripts/make_figures.py](scripts/make_figures.py).*

| at S = 16, k_in = 1, last step: own / oracle cell acc. (self-correction) | fwd | DRM | bwd-TBPTT |
|---|---|---|---|
| shift | 89.5 / 100.0 (0.0%) | 45.3 / 56.6 (0.0%) | 53.7 / 49.7 (16.1%) |
| symfill | 87.9 / 100.0 (0.0%) | 87.9 / 100.0 (0.0%) | 82.5 / 81.2 (19.3%) |
| gravity | 82.7 / 100.0 (0.0%) | 77.0 / 93.7 (0.1%) | 72.4 / 72.0 (16.7%) |
| flood | 87.0 / 100.0 (0.1%) | 70.3 / 96.8 (4.3%) | 69.8 / 70.3 (39.5%) |

How to read the probe:

- **At the same schedule (S = 16, k_in = 1, the table above)** both DRM and fwd do better on noised clean targets than on their own states, and the gap is about as large for DRM as for fwd (larger on flood: DRM 70.3% vs 96.8%, fwd 87.0% vs 100.0%). This schedule is fwd's own dev choice but a poor one for DRM, and near the end of it the oracle input is almost all clean visible cells, so a high oracle number partly measures copying. At this schedule the probe does not single out one-step denoising.
- **At each method's own dev schedule** ([mismatch_own_schedule.png](figures/mismatch_own_schedule.png)) the own-vs-oracle gap at the last step is +13.0 pp for fwd against +3.0 pp for DRM on flood, +17.3 vs +0.3 pp on gravity, and +10.5 vs +0.0 pp on shift. TBPTT, which never sees visible tokens, has gaps of +0.0 pp (flood) and +0.0 pp (gravity).
- **Self-correction** of wrong visible cells is low for every denoiser (fwd 0.1%, DRM 4.3% on flood at S = 16, k_in = 1). DRM's advantage therefore does not come from fixing visible tokens; it is consistent with the k latent steps inside each denoising step (paper §6), but this toy does not isolate that either.
- **Bottom line:** the symptom the paper describes (a one-step denoiser does worse on its own states than on training-distribution states) is present and is larger for fwd than for DRM at their own schedules. This is consistent with paper §1 and §3.1.1. It is a correlation within this probe, not a causal test.

![Denoising trajectory on one flood eval example for DRM and for one-step denoising, each at its dev-chosen schedule. Each column shows the masked input state (white = MASK) and the predicted grid at one outer step; a tick marks a fully correct prediction.](figures/trajectory.png)

*[figures/trajectory.png](figures/trajectory.png): data in [results/demo/trajectories.json](results/demo/trajectories.json), from [scripts/demo_trajectories.py](scripts/demo_trajectories.py), which retrains the seed-100 flood models and asserts that they reproduce the main-grid pass@1 to within 1e-6 (the tolerance was loosened from 1e-9 in round 0 because the main grid stores a float32 mean). The example is the first of 6 recorded eval examples where DRM ends correct and fwd ends wrong: illustrative, not typical.*

## Training timelines and the k sweep

![Dev pass@1 (S=4, k_in=4, 200 dev examples) against training Phi applications used, for each method on all five tasks.](figures/learning_curves.png)

*[figures/learning_curves.png](figures/learning_curves.png): data in [results/summary.json](results/summary.json) (`learning_curves`), logged on **dev** examples only during training ([drm/train.py](drm/train.py)).*

![Eval pass@1 on gravity and flood against the training recursion window k in {1, 2, 4, 8} at a fixed 12,000-Phi training budget, for DRM and bwd-TBPTT, at the native schedule (S=16/k, k_in=k, solid) and at latent S=1, k_in=16 (dotted). Flood DRM native: k=1 39.3% [37.8, 41.0], k=4 75.5% [73.4, 77.6], k=8 76.0% [74.0, 78.0].](figures/ksweep.png)

*[figures/ksweep.png](figures/ksweep.png): data in [results/summary.json](results/summary.json) (`ksweep`), 4 eval seeds (100-103); script [scripts/make_figures.py](scripts/make_figures.py).*

| k | flood DRM native | flood TBPTT latent | gravity DRM native | gravity TBPTT latent |
|---|---|---|---|---|
| 1 | 39.3% [37.8, 41.0] | 49.9% [44.5, 55.5] | 0.0% [0.0, 0.1] | 7.2% [2.6, 12.4] |
| 2 | 65.5% [64.0, 66.9] | 67.4% [62.1, 72.4] | 1.7% [1.6, 1.8] | 53.3% [35.1, 75.1] |
| 4 | 75.5% [73.4, 77.6] | 71.7% [68.1, 75.3] | 73.3% [72.8, 73.6] | 95.9% [95.5, 96.4] |
| 8 | 76.0% [74.0, 78.0] | 59.8% [54.6, 63.3] | 99.1% [98.8, 99.3] | 84.7% [80.1, 91.5] |

At a fixed 12,000-Phi training budget, DRM at its native schedule (S = 16/k, k_in = k) improves steeply from k = 1, which *is* one-step denoising, to k = 4. On flood it then plateaus (75.5% [73.4, 77.6] at k = 4, 76.0% [74.0, 78.0] at k = 8). On gravity it keeps rising to 99.1% [98.8, 99.3] at k = 8, above TBPTT's best latent value (95.9% [95.5, 96.4] at k = 4). TBPTT is native only in latent mode (S = 1, k_in = 16), where it peaks at k = 4 on both tasks and drops at k = 8. So DRM gains more from a larger k than TBPTT does here, in line with the paper's Table 5. The gravity k = 8 result suggests that the k = 4 DRM in the main grid is not DRM's best configuration on gravity, but k was fixed at 4 for every method before the run. The sweep is exploratory: 4 seeds, not in the Holm family, and **the k ≠ 4 runs reuse the lr tuned at k = 4**.

## How this maps to the paper (§3)

| paper claim (§3, §5, App. F) | this toy |
|---|---|
| Forward (one-step) training misaligns train and test, which causes compounding and greedy errors (§1, §3.1.1) | **Consistent: symptom observed, cause not isolated.** At their own schedules, fwd's own-vs-oracle gap is larger than DRM's (flood +13.0 vs +3.0 pp); at a shared S = 16, k_in = 1 schedule it is not. DRM beats fwd by +42.1 pp [+40.6, +43.5] pooled, from the three tasks that need depth. |
| Masked diffusion without recursion fails badly (Table 1: about 0%) | Consistent: `fwd` reaches 0.1% on gravity, 0.2% on symfill and 6.3% on shift. |
| DRM's noise curriculum helps compared with recursion from noise without it (§3.2, §6) | **Partly reproduced** against `bwd_k`: +10.7 pp [+8.5, +12.9] pooled, ahead on shift and gravity, n.s. on flood, **behind on symfill** (-16.4 pp [-20.1, -12.1]). |
| DRM beats TRM (Table 1) | **Not reproduced.** DRM wins flood and (by little) shift, but loses gravity, symfill and the pooled mean to TBPTT (-5.5 pp [-6.8, -4.0]). There is no scratchpad, warm-up loop or halting here, and TBPTT gets 4 windows per batch; the paper's App. F.8 also reports TRM ahead on Sudoku. |
| DRM gains more per recursive step k than TRM (Table 5) | See the k-sweep table above; exploratory, 4 seeds, lr not re-tuned per k. |
| DRM uses extra test-time compute well | Consistent on flood (DRM rises to 86.8% at 64 Phi while latent TBPTT drifts down to 60.7%); flat on gravity. |

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
pip install -r requirements.lock           # exact versions of every package (pip freeze of the run env)
python -m pytest -q tests                  # about 30 s; uses tiny configs
WORKERS=7 sh scripts/pipeline.sh            # whole chain incl. demo, figures, claims, README; skips up-to-date runs
python scripts/mutation_check.py --workers 2
```

`pipeline.sh` uses `$PY` (default `python`) and runs each stage to completion before the next. `tune.py --fail-on-edge` stops it if a chosen lr sits at an end of the grid; `tune.py` and `aggregate.py` exit 2 on an incomplete, stale, extra-file or mixed-code grid. Python 3.13.5, torch 2.14.1+cpu; [requirements.lock](requirements.lock) pins the transitive packages that PNG bytes depend on (pillow, fonttools, kiwisolver, contourpy).

**Compute used (CPU only, no GPU).** The single-threaded runs sum to **43.3 CPU-hours** of training plus evaluation: dev tuning 21.2 h, eval grid 18.6 h, k-sweep 3.5 h (from `train_seconds` + `eval_seconds` in the raw metas). With 7 workers on an 8-core box this was spread over 2026-10-28 15:07 to 2026-10-28 22:10 EDT, including interruptions. A DRM run trains in about 159 s on one core. Kaggle GPU access was available but not used, so every result in this repo comes from the same CPU environment.

## Verification

- **102 tests** cover the pipeline wiring end to end, not just helpers. Examples: training Phi counts and budgets per method; DRM and fwd starting from corrupted targets while bwd starts from a full mask; TBPTT detaching between windows; zero_grad before every backward; loss targets being clean labels; the remask schedule and the per-seed inference generator; eval metrics using the eval split while the learning curve uses dev; the main grid using tuned lrs, all 5 tasks and all 12 eval seeds; tuning choosing only from the matched-compute menu and flagging lr edges; collision-free run names and input validation; checkpoints that reproduce the stored pass@1; the 95% CI width; aggregation pairing by seed, applying one Holm family, and refusing partial, extra or stale grids; and no machine paths or token-like strings in the repo.
- **Mutation check** ([scripts/mutation_check.py](scripts/mutation_check.py), [results/mutation.csv](results/mutation.csv)): an unmutated control run must pass first, and a mutant counts as killed only when pytest exits 1. **92/92 killed**, including the reviewer's 15 round-1 mutants (15/15) and 14 mutants of the round-1 code (14/14). The README-sync test skips inside mutation copies (they have no `results/`), so README drift is checked only in the full tree.
- **Provenance:** every raw result stores the code hash (`91de577ed2a30aae`), its config hash, ISO start/end timestamps, load average and wall time. Dev outputs and `tuning.json` were committed and pushed (3d7f9c4) before the first eval run started. Training is deterministic per seed; the trajectory demo retrains two models and checks them against the grid.

## Limitations

- This is a toy: 6×6 grids, about 52k parameters, one model per task, no augmentation. Effects here say nothing direct about ARC-AGI.
- The lr is chosen from 2 dev seeds, and at lr ≥ 1.6e-2 some runs diverge, so the choice is noisy. Cells still at a grid edge after the last extension: flood:fwd_steps:0.064, gravity:bwd_k:0.064, gravity:fwd:0.0005, shift:bwd_k:0.064.
- The k-sweep reuses the k = 4 lr for every k and has 4 seeds.
- The seed is the replication unit. CIs resample seeds only, and the 1,000 eval examples are shared across seeds. The percentile bootstrap under-covers somewhat at 12 seeds; t-intervals are in the tables.
- Inference uses evenly spaced remask levels and uniform remasking, with no confidence-based remasking.
- No LICENSE file yet; that is the repository owner's decision.
