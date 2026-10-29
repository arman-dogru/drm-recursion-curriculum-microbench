# DRM recursion-curriculum microbench (CPU toy)

A small, CPU-only reproduction of the core idea of **Denoising Recursion Models** (DRM): corrupt the target, then train a shared recursive block to recover it over **k recursive steps** rather than in one jump. It compares DRM with **one-step denoising** (diffusion-style "forward" training) and with **backward recursion from a full mask** (TRM-style TBPTT), all on the same block Phi, at matched parameter count and matched training FLOPs.

> Credit: Cameron, Wang, Ivanov, Bhattacharyya, Chételat and Zhang, *One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models*, Huawei Technologies, University of Toronto and University of Waterloo, [arXiv:2604.18839](https://arxiv.org/html/2604.18839v1). Official code: [wwwwwwwwz/DenoisingRecursionModels](https://github.com/wwwwwwwwz/DenoisingRecursionModels). This repo is an independent didactic toy on 6×6 grids. It makes no ARC-AGI or SOTA claim.

## TL;DR

> **Round 2 (after the second review).** The headline now lets every method stop early: the schedule is chosen on dev among all schedules with **at most** 16 test Phi (S·k_in ≤ 16), not exactly 16. Before, the single-window `bwd_k`, trained on 4 steps, was forced to run 16. This was re-tuned from the committed dev runs; only the {{n_cells_lr_changed}} cells whose lr changed were re-run on eval. **The noise-curriculum result changed** (second bullet). See [Round 2 changes](#round-2-changes).
>
> **Round 3 (after the third review).** No new training runs and no change to any number in the Holm family. The wording changed: the gravity lead over `bwd_k` is **not robust** and the noise curriculum is **not reproduced** (second bullet). The tie-break rule is now documented as it is implemented, and its effect is reported. See [Round 3 changes](#round-3-changes).
>
> **Round 1.** All **5 tasks** × **{{n_eval_seeds}} eval seeds** (the round-0 task cut is reverted), an lr grid of 5e-4 … 6.4e-2 for **every** task and method, and pure latent inference up to 64 steps. Round-0 and round-1 artifacts are kept in [results/r0_932be94/](results/r0_932be94/) and [results/r1_c4532b5/](results/r1_c4532b5/).

- **DRM beats one-step denoising, but on 3 of the 5 tasks, not all.** Pooled over all five tasks, pass@1 is {{p1_pooled_drm}} for DRM against {{p1_pooled_fwd}} for the FLOP-matched one-step denoiser ({{d_pooled_drm_vs_fwd}}, {{wl_pooled_drm_vs_fwd}} seeds won/lost, Holm p = {{ph_pooled_drm_vs_fwd}}). The gain comes from shift ({{d_shift_drm_vs_fwd}}), gravity ({{d_gravity_drm_vs_fwd}}) and flood ({{d_flood_drm_vs_fwd}}). On colormap every method scores 100%, and on symfill DRM and fwd are both near 0 ({{p1m_symfill_drm}}% vs {{p1m_symfill_fwd}}%, {{d_symfill_drm_vs_fwd}}).
- **No pooled evidence that the noise curriculum helps.** DRM is the same k = 4 recursion as `bwd_k`, but it starts from a corrupted target instead of a full mask. Once `bwd_k` may stop early (dev picks a 4- or 8-step latent schedule) instead of being forced to 16 steps, the pooled difference is {{d_pooled_drm_vs_bwd_k}} ({{wl_pooled_drm_vs_bwd_k}} seeds won/lost, Holm p = {{ph_pooled_drm_vs_bwd_k}}). In round 1 it was {{r1_d_pooled_drm_vs_bwd_k}}. Scoring `bwd_k` at its native 4-step schedule (S = 1, k_in = 4, same models) instead gives {{snat_pooled_d}} pooled (raw p = {{snat_pooled_p}}). The only per-task difference that holds up is symfill, where DRM is **behind**:
  - **Behind on symfill:** {{d_symfill_drm_vs_bwd_k}}, Holm p = {{ph_symfill_drm_vs_bwd_k}} (dev picked the native schedule here, so the two analyses agree).
  - **Gravity, {{d_gravity_drm_vs_bwd_k}} (Holm p = {{ph_gravity_drm_vs_bwd_k}}), is not robust.** It is mostly dev-selection noise (a winner's curse from picking among 120 lr × schedule cells on 2 dev seeds). At the schedule dev picked for `bwd_k` (S = 1, k_in = 8) its 12 eval seeds range from {{range_gravity_bwd_k}}% (sd {{sd_gravity_bwd_k}} pp), so its mean is low and noisy. At its native S = 1, k_in = 4 the same models score {{snat_gravity_bwd_k}} (sd {{snat_gravity_bwd_k_sd}} pp), and DRM's lead shrinks to {{snat_gravity_d}} ({{snat_gravity_wl}} seeds, raw p = {{snat_gravity_p}}, not significant). DRM is itself noisy on gravity (seeds range {{range_gravity_drm}}%).
  - **Flood and shift: not significant after Holm.** Flood is {{d_flood_drm_vs_bwd_k}} (Holm p = {{ph_flood_drm_vs_bwd_k}}) and shift {{d_shift_drm_vs_bwd_k}}. At `bwd_k`'s native 4-step schedule flood is {{snat_flood_d}}, but that pits DRM at its 16-Phi dev pick ({{sched_flood_drm}}) against `bwd_k` at 4 Phi, so it is a compute mismatch, not curriculum evidence. Scored at the **same** schedule, the two are level on flood ({{ss_flood_bwd_k_pick_d}} at {{ss_flood_bwd_k_pick_sched}}, `bwd_k`'s dev pick, {{ss_flood_bwd_k_pick_wl}} seeds, raw p = {{ss_flood_bwd_k_pick_p}}); that holds at this one schedule only, since the other two flood same-schedule rows range from {{ss_flood_bwd_k_native_d}} at {{ss_flood_bwd_k_native_sched}} to {{ss_flood_drm_pick_d}} at {{ss_flood_drm_pick_sched}} (Round 4 table below). On gravity with both at S1/k4 DRM is ahead by only {{ss_gravity_bwd_k_native_d}} (raw p = {{ss_gravity_bwd_k_native_p}}). These same-schedule rows are descriptive and outside the Holm family ([results/same_schedule.json](results/same_schedule.json), [scripts/same_schedule.py](scripts/same_schedule.py)).
- **DRM loses to TRM-style backward TBPTT overall.** Pooled it is {{p1_pooled_drm}} against {{p1_pooled_bwd_tbptt}} ({{d_pooled_drm_vs_bwd_tbptt}}, Holm-significant). DRM is ahead on flood ({{d_flood_drm_vs_bwd_tbptt}}) and slightly on shift ({{d_shift_drm_vs_bwd_tbptt}}), and behind on gravity ({{d_gravity_drm_vs_bwd_tbptt}}) and symfill ({{d_symfill_drm_vs_bwd_tbptt}}). TBPTT here gets 4 windows (16 recursive steps) per batch for the same FLOPs.
- **With more test compute, DRM keeps improving on flood and TBPTT does not**, now that both can use their best inference mode at every budget: from 16 to 64 Phi, DRM goes {{cc_flood_drm_16}}% → {{cc_flood_drm_64}}% ({{ccs_flood_drm_64}}) and latent TBPTT {{cc_flood_bwd_tbptt_16}}% → {{cc_flood_bwd_tbptt_64}}% ({{ccs_flood_bwd_tbptt_64}}). On gravity both are flat (DRM {{cc_gravity_drm_64}}%, TBPTT {{cc_gravity_bwd_tbptt_64}}% at 64 Phi).
- **A train-test-mismatch symptom shows up for one-step denoising**, but this probe does not prove it is the cause. At each method's own dev-best 16-Phi schedule, flood cell accuracy on its own trajectory vs on noised clean targets is {{mmo_flood_fwd_own_last}}% vs {{mmo_flood_fwd_oracle_last}}% for fwd, and {{mmo_flood_drm_own_last}}% vs {{mmo_flood_drm_oracle_last}}% for DRM.

All numbers are eval-split pass@1 (whole-grid exact match) over **{{n_eval_seeds}} fresh training seeds** (100-111), at **at most** 16 test Phi applications (S·k_in ≤ 16), with the lr and the inference schedule chosen on dev for each (task, method). Intervals are 95% percentile seed-bootstrap CIs (they under-cover a little at n = {{n_eval_seeds}}; Student-t intervals are in [results/tables/](results/tables/)). p-values are exact paired sign-flip tests, Holm-corrected over all {{holm_m}} comparisons. The model has {{n_params}} parameters, the code hash is `{{code_hash}}` (results computed under `{{code_hash_parent}}`, verified output-identical; see [Verification](#verification)), and there are {{n_tests}} tests with {{mut_killed}}/{{mut_total}} planted bugs caught (see [Verification](#verification)). Everything ran on CPU; no GPU was used.

![Grouped bar chart of eval pass@1 for five trainers on all five tasks, the pooled mean over all five (primary) and the descriptive pooled mean over shift, gravity and flood. Pooled over all five: DRM {{p1m_pooled_drm}}%, backward TBPTT {{p1m_pooled_bwd_tbptt}}%, single-window backward {{p1m_pooled_bwd_k}}%, one-step denoising FLOP-matched {{p1m_pooled_fwd}}%, step-matched {{p1m_pooled_fwd_steps}}%.](figures/primary.png)

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

S = 1 is pure latent recursion, which is how a backward-trained model is natively used. Every run is scored on all 25 schedules with S ∈ {1,…,16}, k_in ∈ {1,…,64} and S·k_in ≤ 64, so pure latent recursion (S = 1) is available at every budget up to 64. The headline uses every schedule with **at most** matched compute, S·k_in ≤ 16 (15 schedules), picking the one that is best on dev for each (task, method). Exact dev ties go to the lowest lr first and only then to menu order, where the 16-Phi schedules come first (lr is the outer loop of the search). A method may therefore stop early, e.g. `bwd_k`, trained on one 4-step window, can run 4 latent steps instead of being forced to 16.

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

### Round 5 changes

No new runs, and no result, tuning or figure file changed (only `claim_numbers.json` and the mutation tables are regenerated).

- **Claims rebuild test (review r5 m1).** [tests/test_r5.py](tests/test_r5.py) rebuilds every claim from `results/` and requires it to equal the committed [results/claim_numbers.json](results/claim_numbers.json) (full tree only), and checks the same-schedule (`ss_*`) won/lost and S/k labels on a fabricated asymmetric `same_schedule.json`, so a won/lost or S/k swap fails inside the mutation copies too. Both swaps are in the harness, with the two pipeline mutants below ({{mut_rev5_killed}}/{{mut_rev5_total}} killed).
- **Wording (n1, n2).** The heatmap sentence below no longer says `bwd_k` always drops past its 4-step window (on flood it is best at S1/k8), and the TL;DR gives the range of the other flood same-schedule rows next to "level".
- **Pipeline order (n3).** A test now requires `pipeline.sh` to run `same_schedule.py` after `aggregate.py` and before `claims.py`.

### Round 4 changes

No new training runs. No existing result, tuning or figure file changed; the one new result file is the descriptive `results/same_schedule.json`. The Holm family and every headline number are unchanged.

- **Flood wording (review r4 m1).** The TL;DR no longer reads the native-schedule flood gap as an effect: `sensitivity.bwd_k_native` moves only `bwd_k`'s schedule, so on flood it compares DRM at 16 Phi with `bwd_k` at 4 Phi. A new descriptive check, [scripts/same_schedule.py](scripts/same_schedule.py) → [results/same_schedule.json](results/same_schedule.json), scores **both** methods at the same schedule with the same tuned lrs and eval runs:

  | task | schedule (whose pick) | DRM − `bwd_k` | seeds won/lost | raw p |
  |---|---|---|---|---|
  | flood | {{ss_flood_bwd_k_pick_sched}} (`bwd_k`'s dev pick) | {{ss_flood_bwd_k_pick_d}} | {{ss_flood_bwd_k_pick_wl}} | {{ss_flood_bwd_k_pick_p}} |
  | flood | {{ss_flood_bwd_k_native_sched}} (`bwd_k` native) | {{ss_flood_bwd_k_native_d}} | {{ss_flood_bwd_k_native_wl}} | {{ss_flood_bwd_k_native_p}} |
  | flood | {{ss_flood_drm_pick_sched}} (DRM's dev pick) | {{ss_flood_drm_pick_d}} | {{ss_flood_drm_pick_wl}} | {{ss_flood_drm_pick_p}} |
  | gravity | {{ss_gravity_bwd_k_native_sched}} (both native) | {{ss_gravity_bwd_k_native_d}} | {{ss_gravity_bwd_k_native_wl}} | {{ss_gravity_bwd_k_native_p}} |
  | gravity | {{ss_gravity_bwd_k_pick_sched}} (both dev picks) | {{ss_gravity_bwd_k_pick_d}} | {{ss_gravity_bwd_k_pick_wl}} | {{ss_gravity_bwd_k_pick_p}} |

  Which method "wins" depends mostly on which schedule both are scored at: each does best near its own pick. `bwd_k` drops sharply under re-masking (S ≥ 2) and at k_in ≥ 16, but not simply past its 4-step training window: on flood it is best just past it (S1/k8 {{hm_flood_bwd_k_S1_k8}} vs S1/k4 {{hm_flood_bwd_k_S1_k4}}; S1/k16 {{hm_flood_bwd_k_S1_k16}}, S2/k8 {{hm_flood_bwd_k_S2_k8}}), while on gravity it falls from {{hm_gravity_bwd_k_S1_k4}} at S1/k4 to {{hm_gravity_bwd_k_S1_k8}} at S1/k8. This supports the reading below ("not reproduced"). The same pattern is visible in [figures/schedule_heatmaps.png](figures/schedule_heatmaps.png). The other three tasks are in the JSON.
- **Tests for the round-3 code (m2).** [tests/test_r4.py](tests/test_r4.py) pins the sensitivity check's two-sided p, wins/losses, seed sd (ddof = 1) and 5-task pool; that `summary.json` `meta.computed_under` keeps an ancestor hash on regeneration; the claims tie list (colormap excluded) and superseded/round-0 CPU-h (train + eval seconds); that `verify_lineage` exits 1 on a failed check and leaves `code_lineage.json` unchanged; and the default full lineage set on a fabricated tree. The reviewer's P1-P9 are in the harness ({{mut_rev4_killed}}/{{mut_rev4_total}} killed).
- **Nits.** `tune.py`'s docstring lists all 6 tie cells (n2). If the dev runs ever carry more than one lineage-linked hash, `tuning.json` lists them all in `meta.computed_under` instead of silently keeping only the canonical one (n4; with one hash, as now, the file is byte-identical). "Total spent" now says it excludes the lineage checks and pilots (n5). n1 (squash vs plain merge) is left to the owner.

### Round 3 changes

- **Gravity vs `bwd_k` reworded (review r3 major 1).** The {{d_gravity_drm_vs_bwd_k}} lead comes from a seed-unstable schedule picked on dev for `bwd_k` (eval seeds range {{range_gravity_bwd_k}}%, sd {{sd_gravity_bwd_k}} pp). At `bwd_k`'s native 4-step schedule it is {{snat_gravity_d}} ({{snat_gravity_wl}}, raw p = {{snat_gravity_p}}). `aggregate.py` now writes this check to `summary.json` (`sensitivity.bwd_k_native`, every task and the pool, outside the Holm family). `tables/primary.csv` gains a seed-sd column.
- **Tie-break documented as implemented (m3).** The docs said ties keep the 16-Phi schedule. In fact lr is the outer loop with a strict `>`, so an exact dev tie goes to the **lowest lr** first and only then to menu order. The code is unchanged; the docs are fixed and a test pins the rule. Exact cross-lr ties on dev occur in {{n_tie_cells}} non-colormap cells ({{tie_cells}}). The one that mattered is symfill `bwd_tbptt`: round 1's 3.2e-2 at S1/k16 and 1.6e-2 at S1/k8 both score 0.288 on dev, so the lower lr won and 12 eval runs were redone. Keeping the round-1 runs instead would give symfill TBPTT {{tie_symfill_tbptt_r1}} instead of {{p1_symfill_bwd_tbptt}}, and pooled DRM − TBPTT {{tie_pooled_d_vs_tbptt}} instead of {{d_pooled_drm_vs_bwd_tbptt}} (`sensitivity.tie_symfill_tbptt_r1`).
- **Results keep the hash they were computed under (m2).** Regenerating `tuning.json`, `summary.json` or the README at a newer HEAD used to relabel the results with the current code hash. `tune.py` now records the raw runs' hash, and the README's "computed under" hash comes from the raw metas (`summary.json` `meta.computed_under`). A test checks that regenerating `tuning.json` at HEAD is byte-identical.
- **Tests (m4).** New tests in `tests/test_r3.py` kill all five survivors (W2, W3, W4, W7, W10) and W8 and W11. They cover cross-lr tie order, unit tests of the lineage comparison, a CLI collision exiting 2, a cyclic lineage file, and non-dev runs rejected by `tune.py`. W1-W11 and four mutants of my own (W12-W15) are in the harness.
- **Nits.** The round-1 lr-edge list now shows round-1 edges (n2). Total compute spent includes superseded and round-0 runs (n3). `pipeline.sh` runs the hygiene scan before it commits or pushes (n4). The lineage record now holds the static argument and re-runs one full committed run per task plus a k = 1 k-sweep run (n5). n1 (paths in old history) is left to the owner.

### Round 2 changes

- **Headline menu S·k_in ≤ 16 (review major 1).** Rounds 0-1 forced exactly 16 test Phi. That pushed `bwd_k` 4× past its 4-step training window. In round 1, at lr 6.4e-2, its gravity eval pass@1 was 72.9% at S=1, k_in=4 against 28.8% at the forced S=1, k_in=16. tune.py now chooses lr and schedule among all 15 schedules with S·k_in ≤ 16, using the committed dev runs; no new dev runs were needed. The chosen lr changed for {{n_cells_lr_changed}} cells ({{cells_lr_changed}}). Only those cells were re-run on eval ({{n_runs_r2}} runs); their round-1 runs are in [results/r1_c4532b5/](results/r1_c4532b5/). Every other cell reuses its round-1 eval runs, with only the scored schedule changing.
- **Commit-before-eval wording and enforcement (m2).** Round 1 dev outputs were committed at 18:18:47 EDT, before eval started, and pushed at 18:19:18, before any eval run finished. They were not "pushed before any eval run". `pipeline.sh` now fails if the commit fails, checks that `results/raw/dev` and `tuning.json` are fully committed, and pushes before launching eval.
- **k-sweep labels (m3).** TBPTT's training horizon is n_sup·k = 4k steps, so the fixed S=1, k_in=16 latent schedule is "native" only at k = 4. The table now has a horizon-matched column (S=1, k_in=4k).
- **Tests (m4).** All 6 surviving mutants of the reviewer's 13 are killed by new tests in `tests/test_r2.py`, and the reviewer's mutants were added to the harness.
- **Run names (m5).** Names now always include the training budget and attention radius (`_b12000_r1_`), and any other non-default field adds a config-hash suffix. `run_one.py` refuses to overwrite an output holding a different config unless `--force` is given, and a negative radius is rejected. The committed dev results were never overwritten in this repo (the reviewer's overwrite happened in a scratch clone). These changes touch hashed files, so the code hash changed. Instead of re-running everything, `scripts/verify_lineage.py` ran configs under both code versions and found every output identical, then recorded the link in [results/code_lineage.json](results/code_lineage.json).
- **Nits:** an `LR_GRID` pointer to the effective 8-point grid (n2), readable schedule labels in the own-schedule mismatch figure (n3), TBPTT's 0 gap noted as by construction (n4), the compute-curve caption saying lr is fixed (n5), and the lower lr edge noted as never extended (n6). n1 (paths in old history) is left to the owner's merge choice; no history was rewritten.

### Round 1 changes

- **All 5 tasks, 12 seeds (M1).** Round 0 dropped colormap and symfill from eval after looking at dev, and justified it with a "3.5× CPU slowdown". The logs show short bursts of slow jobs, not a sustained slowdown, and the two dropped tasks were the ones where DRM did not win on dev. The cut is reverted. The full story is in [docs/method.md](docs/method.md#round-0-scope-cut-reverted).
- **lr grid (M2).** Every (task, method) gets the same grid {5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2}. The last two points were added on dev evidence, for every cell, before any eval run; 6.4e-2 was declared the last extension. Round-1 cells whose dev optimum was still at an edge: {{r1_lr_edges}} (round-2 edges are under Limitations).
- **Inference (M3).** Pure latent recursion (S = 1) is now offered at 32 and 64 Phi, so at every budget each method uses its dev-best mode (latent or re-masking).
- **Provenance (m4).** Dev outputs and `tuning.json` were committed (3d7f9c4, 18:18:47 EDT) before the first eval run started, and pushed (18:19:18) before any eval run finished. Every raw result records start and end timestamps, load average and wall time.
- **Smaller items:** collision-free run names and input validation (m7), lock file and no hard-coded interpreter (m5), softened pilot claim (m6), DRM shown next to fwd in the mismatch probe (m1), corrected CPU-hours (m2), more mutants (m3), and the nits. See the PR thread for a per-item list.

## Results (eval split)

| task | DRM | bwd-TBPTT | bwd-k | fwd (FLOP-matched) | fwd (step-matched) |
|---|---|---|---|---|---|
| colormap | {{p1_colormap_drm}} | {{p1_colormap_bwd_tbptt}} | {{p1_colormap_bwd_k}} | {{p1_colormap_fwd}} | {{p1_colormap_fwd_steps}} |
| shift | {{p1_shift_drm}} | {{p1_shift_bwd_tbptt}} | {{p1_shift_bwd_k}} | {{p1_shift_fwd}} | {{p1_shift_fwd_steps}} |
| symfill | {{p1_symfill_drm}} | {{p1_symfill_bwd_tbptt}} | {{p1_symfill_bwd_k}} | {{p1_symfill_fwd}} | {{p1_symfill_fwd_steps}} |
| gravity | {{p1_gravity_drm}} | {{p1_gravity_bwd_tbptt}} | {{p1_gravity_bwd_k}} | {{p1_gravity_fwd}} | {{p1_gravity_fwd_steps}} |
| flood | {{p1_flood_drm}} | {{p1_flood_bwd_tbptt}} | {{p1_flood_bwd_k}} | {{p1_flood_fwd}} | {{p1_flood_fwd_steps}} |
| **pooled, all 5 (primary)** | **{{p1_pooled_drm}}** | {{p1_pooled_bwd_tbptt}} | {{p1_pooled_bwd_k}} | {{p1_pooled_fwd}} | {{p1_pooled_fwd_steps}} |
| pooled3: shift, gravity, flood (descriptive) | {{p1_pooled3_drm}} | {{p1_pooled3_bwd_tbptt}} | {{p1_pooled3_bwd_k}} | {{p1_pooled3_fwd}} | {{p1_pooled3_fwd_steps}} |

Dev-chosen settings (lr; schedule):

| task | DRM | bwd-TBPTT | bwd-k | fwd | fwd-steps |
|---|---|---|---|---|---|
| colormap | {{lr_colormap_drm}}; {{sched_colormap_drm}} | {{lr_colormap_bwd_tbptt}}; {{sched_colormap_bwd_tbptt}} | {{lr_colormap_bwd_k}}; {{sched_colormap_bwd_k}} | {{lr_colormap_fwd}}; {{sched_colormap_fwd}} | {{lr_colormap_fwd_steps}}; {{sched_colormap_fwd_steps}} |
| shift | {{lr_shift_drm}}; {{sched_shift_drm}} | {{lr_shift_bwd_tbptt}}; {{sched_shift_bwd_tbptt}} | {{lr_shift_bwd_k}}; {{sched_shift_bwd_k}} | {{lr_shift_fwd}}; {{sched_shift_fwd}} | {{lr_shift_fwd_steps}}; {{sched_shift_fwd_steps}} |
| symfill | {{lr_symfill_drm}}; {{sched_symfill_drm}} | {{lr_symfill_bwd_tbptt}}; {{sched_symfill_bwd_tbptt}} | {{lr_symfill_bwd_k}}; {{sched_symfill_bwd_k}} | {{lr_symfill_fwd}}; {{sched_symfill_fwd}} | {{lr_symfill_fwd_steps}}; {{sched_symfill_fwd_steps}} |
| gravity | {{lr_gravity_drm}}; {{sched_gravity_drm}} | {{lr_gravity_bwd_tbptt}}; {{sched_gravity_bwd_tbptt}} | {{lr_gravity_bwd_k}}; {{sched_gravity_bwd_k}} | {{lr_gravity_fwd}}; {{sched_gravity_fwd}} | {{lr_gravity_fwd_steps}}; {{sched_gravity_fwd_steps}} |
| flood | {{lr_flood_drm}}; {{sched_flood_drm}} | {{lr_flood_bwd_tbptt}}; {{sched_flood_bwd_tbptt}} | {{lr_flood_bwd_k}}; {{sched_flood_bwd_k}} | {{lr_flood_fwd}}; {{sched_flood_fwd}} | {{lr_flood_fwd_steps}}; {{sched_flood_fwd_steps}} |

On colormap every setting reaches 100% on dev, so the tie-break (first lr in grid order) picks 5e-4 for everyone.

### DRM minus each baseline (paired by seed)

| | vs fwd (FLOP) | vs fwd (steps) | vs bwd-k | vs bwd-TBPTT |
|---|---|---|---|---|
| colormap | {{d_colormap_drm_vs_fwd}} (p_H={{ph_colormap_drm_vs_fwd}}) | {{d_colormap_drm_vs_fwd_steps}} (p_H={{ph_colormap_drm_vs_fwd_steps}}) | {{d_colormap_drm_vs_bwd_k}} (p_H={{ph_colormap_drm_vs_bwd_k}}) | {{d_colormap_drm_vs_bwd_tbptt}} (p_H={{ph_colormap_drm_vs_bwd_tbptt}}) |
| shift | {{d_shift_drm_vs_fwd}} (p_H={{ph_shift_drm_vs_fwd}}) | {{d_shift_drm_vs_fwd_steps}} (p_H={{ph_shift_drm_vs_fwd_steps}}) | {{d_shift_drm_vs_bwd_k}} (p_H={{ph_shift_drm_vs_bwd_k}}) | {{d_shift_drm_vs_bwd_tbptt}} (p_H={{ph_shift_drm_vs_bwd_tbptt}}) |
| symfill | {{d_symfill_drm_vs_fwd}} (p_H={{ph_symfill_drm_vs_fwd}}) | {{d_symfill_drm_vs_fwd_steps}} (p_H={{ph_symfill_drm_vs_fwd_steps}}) | {{d_symfill_drm_vs_bwd_k}} (p_H={{ph_symfill_drm_vs_bwd_k}}) | {{d_symfill_drm_vs_bwd_tbptt}} (p_H={{ph_symfill_drm_vs_bwd_tbptt}}) |
| gravity | {{d_gravity_drm_vs_fwd}} (p_H={{ph_gravity_drm_vs_fwd}}) | {{d_gravity_drm_vs_fwd_steps}} (p_H={{ph_gravity_drm_vs_fwd_steps}}) | {{d_gravity_drm_vs_bwd_k}} (p_H={{ph_gravity_drm_vs_bwd_k}}) | {{d_gravity_drm_vs_bwd_tbptt}} (p_H={{ph_gravity_drm_vs_bwd_tbptt}}) |
| flood | {{d_flood_drm_vs_fwd}} (p_H={{ph_flood_drm_vs_fwd}}) | {{d_flood_drm_vs_fwd_steps}} (p_H={{ph_flood_drm_vs_fwd_steps}}) | {{d_flood_drm_vs_bwd_k}} (p_H={{ph_flood_drm_vs_bwd_k}}) | {{d_flood_drm_vs_bwd_tbptt}} (p_H={{ph_flood_drm_vs_bwd_tbptt}}) |
| **pooled** | {{d_pooled_drm_vs_fwd}} (p_H={{ph_pooled_drm_vs_fwd}}) | {{d_pooled_drm_vs_fwd_steps}} (p_H={{ph_pooled_drm_vs_fwd_steps}}) | {{d_pooled_drm_vs_bwd_k}} (p_H={{ph_pooled_drm_vs_bwd_k}}) | {{d_pooled_drm_vs_bwd_tbptt}} (p_H={{ph_pooled_drm_vs_bwd_tbptt}}) |

{{n_sig}} of {{holm_m}} comparisons are significant after Holm (p_H < 0.05). With {{n_eval_seeds}} seeds the smallest attainable raw p is 2/4096. On colormap every difference is exactly 0 (p = 1).

![Forest plot of DRM minus each baseline in eval pass@1, with 95% seed-bootstrap CIs for all five tasks and the pooled mean. Pooled: vs fwd {{d_pooled_drm_vs_fwd}}, vs bwd-k {{d_pooled_drm_vs_bwd_k}}, vs bwd-TBPTT {{d_pooled_drm_vs_bwd_tbptt}}. Filled markers are Holm-significant.](figures/forest.png)

*[figures/forest.png](figures/forest.png): data in [results/summary.json](results/summary.json) (`tests`) and [results/tables/tests.csv](results/tables/tests.csv); script [scripts/make_figures.py](scripts/make_figures.py).*

**Round 0 vs round 1 (pooled, DRM minus baseline).** Round 0 pooled only shift, gravity and flood with 10 seeds and a narrower lr grid: vs fwd {{r0_d_pooled_drm_vs_fwd}}, vs bwd-k {{r0_d_pooled_drm_vs_bwd_k}}, vs bwd-TBPTT {{r0_d_pooled_drm_vs_bwd_tbptt}}. The comparable round-1 number is pooled3 above; the primary round-1 number is the 5-task pool.

**Round 1 vs round 2 (≤ 16-Phi menu).** Pooled pass@1 for `bwd_k` went from {{r1_p1_pooled_bwd_k}} to {{p1_pooled_bwd_k}}, so DRM − bwd-k went from {{r1_d_pooled_drm_vs_bwd_k}} to {{d_pooled_drm_vs_bwd_k}}. The other comparisons barely moved: DRM − fwd went {{r1_d_pooled_drm_vs_fwd}} → {{d_pooled_drm_vs_fwd}}, and DRM − TBPTT went {{r1_d_pooled_drm_vs_bwd_tbptt}} → {{d_pooled_drm_vs_bwd_tbptt}}. Comparisons significant after Holm: {{r1_n_sig}} → {{n_sig}} of {{holm_m}}.

## pass@1 vs test-time compute

![Line plots of eval pass@1 against test-time Phi applications (1 to 64, log scale) for each method on all five tasks, using the dev-best schedule at each compute level among all schedules with that budget. Filled markers are latent schedules (S=1), hollow markers re-masking (S>1). On flood at 64 Phi: DRM {{cc_flood_drm_64}}% ({{ccs_flood_drm_64}}), bwd-TBPTT {{cc_flood_bwd_tbptt_64}}% ({{ccs_flood_bwd_tbptt_64}}), fwd {{cc_flood_fwd_64}}%. On gravity at 64 Phi: DRM {{cc_gravity_drm_64}}%, bwd-TBPTT {{cc_gravity_bwd_tbptt_64}}% ({{ccs_gravity_bwd_tbptt_64}}).](figures/compute_curves.png)

*[figures/compute_curves.png](figures/compute_curves.png): data in [results/summary.json](results/summary.json) (`compute_curves`, which also records the schedule used at each point). The schedule per compute level was chosen on dev ([results/tuning.json](results/tuning.json), `by_compute`); **the lr is fixed at each cell's headline (≤ 16 Phi) dev choice and is not re-tuned per budget**.*

![Heatmaps of eval pass@1 for all 25 inference schedules, outer steps S by inner steps k_in up to 64, for each method (columns) and task (rows). The red box marks the headline schedule chosen on dev among all schedules with at most 16 Phi.](figures/schedule_heatmaps.png)

*[figures/schedule_heatmaps.png](figures/schedule_heatmaps.png): data in [results/summary.json](results/summary.json) (`schedules`); script [scripts/make_figures.py](scripts/make_figures.py).*

What the compute curves and heatmaps show (each point uses the dev-best schedule at that budget, latent or re-masking):

- **One-step denoising does not use extra test compute.** On flood it peaks at {{cc_flood_fwd_16}}% at 16 Phi and falls to {{cc_flood_fwd_64}}% at 64; on gravity and symfill it is near 0 at every budget, and on shift it stays at about {{cc_shift_fwd_16}}%.
- **DRM keeps improving past the matched budget on flood**, {{cc_flood_drm_16}}% at 16 Phi to {{cc_flood_drm_32}}% at 32 and {{cc_flood_drm_64}}% at 64, using re-masking schedules. On gravity it is flat ({{cc_gravity_drm_16}}% → {{cc_gravity_drm_64}}%).
- **Backward-trained models are best with latent recursion near their training horizon.** TBPTT now gets pure latent recursion at every budget and the dev picks it every time (S = 1). Going past 16 latent steps it slowly loses accuracy: flood {{cc_flood_bwd_tbptt_16}}% → {{cc_flood_bwd_tbptt_32}}% → {{cc_flood_bwd_tbptt_64}}%, gravity {{cc_gravity_bwd_tbptt_16}}% → {{cc_gravity_bwd_tbptt_64}}%. In round 0 the grid capped latent recursion at 16 steps, which forced TBPTT into re-masking above 16 Phi and made it look as if it collapsed; that was an artifact of the grid (review M3). The single-window `bwd_k` degrades faster (gravity {{cc_gravity_bwd_k_16}}% at 16 Phi, {{cc_gravity_bwd_k_64}}% at 64).
- **DRM does well whenever k_in ≥ 4**, its training window, at almost any S, and fails with k_in ≤ 2 (heatmaps).

## Train-test mismatch probe

Under the masking process, every **visible** cell a denoiser sees in training is correct. At test time, the visible cells are its own earlier guesses. The probe compares, at each outer step, accuracy along the model's own trajectory with accuracy when the same step's input is the training distribution (the clean target masked at the same level, the "oracle" input). It also measures how many wrong visible cells each model fixes.

![Three panels for flood with S=16, k_in=1 for every method. Left: cell accuracy per denoising step on the model's own trajectory (solid) versus a noised clean target (dashed), for fwd, DRM and bwd-TBPTT. Middle: fraction of visible input cells that are wrong. Right: fraction of those wrong cells fixed at the next step (fwd {{mm_flood_fwd_selfcorr}}%, DRM {{mm_flood_drm_selfcorr}}%).](figures/mismatch.png)

*[figures/mismatch.png](figures/mismatch.png): data in [results/summary.json](results/summary.json) (`traces.flood.*.S16_k1`), recorded by [scripts/run_one.py](scripts/run_one.py); script [scripts/make_figures.py](scripts/make_figures.py).*

![Grouped bars per task: last-step cell accuracy on the model's own trajectory (solid) and on the noised clean target (hatched), for fwd, DRM and bwd-TBPTT, each at its own dev-best 16-Phi schedule (labelled above each pair). Flood: fwd {{mmo_flood_fwd_own_last}}% vs {{mmo_flood_fwd_oracle_last}}%, DRM {{mmo_flood_drm_own_last}}% vs {{mmo_flood_drm_oracle_last}}%.](figures/mismatch_own_schedule.png)

*[figures/mismatch_own_schedule.png](figures/mismatch_own_schedule.png): data in [results/summary.json](results/summary.json) (`traces_at_choice`); script [scripts/make_figures.py](scripts/make_figures.py).*

| at S = 16, k_in = 1, last step: own / oracle cell acc. (self-correction) | fwd | DRM | bwd-TBPTT |
|---|---|---|---|
| shift | {{mm_shift_fwd_own_last_cell}} / {{mm_shift_fwd_oracle_last_cell}} ({{mm_shift_fwd_selfcorr}}%) | {{mm_shift_drm_own_last_cell}} / {{mm_shift_drm_oracle_last_cell}} ({{mm_shift_drm_selfcorr}}%) | {{mm_shift_bwd_tbptt_own_last_cell}} / {{mm_shift_bwd_tbptt_oracle_last_cell}} ({{mm_shift_bwd_tbptt_selfcorr}}%) |
| symfill | {{mm_symfill_fwd_own_last_cell}} / {{mm_symfill_fwd_oracle_last_cell}} ({{mm_symfill_fwd_selfcorr}}%) | {{mm_symfill_drm_own_last_cell}} / {{mm_symfill_drm_oracle_last_cell}} ({{mm_symfill_drm_selfcorr}}%) | {{mm_symfill_bwd_tbptt_own_last_cell}} / {{mm_symfill_bwd_tbptt_oracle_last_cell}} ({{mm_symfill_bwd_tbptt_selfcorr}}%) |
| gravity | {{mm_gravity_fwd_own_last_cell}} / {{mm_gravity_fwd_oracle_last_cell}} ({{mm_gravity_fwd_selfcorr}}%) | {{mm_gravity_drm_own_last_cell}} / {{mm_gravity_drm_oracle_last_cell}} ({{mm_gravity_drm_selfcorr}}%) | {{mm_gravity_bwd_tbptt_own_last_cell}} / {{mm_gravity_bwd_tbptt_oracle_last_cell}} ({{mm_gravity_bwd_tbptt_selfcorr}}%) |
| flood | {{mm_flood_fwd_own_last_cell}} / {{mm_flood_fwd_oracle_last_cell}} ({{mm_flood_fwd_selfcorr}}%) | {{mm_flood_drm_own_last_cell}} / {{mm_flood_drm_oracle_last_cell}} ({{mm_flood_drm_selfcorr}}%) | {{mm_flood_bwd_tbptt_own_last_cell}} / {{mm_flood_bwd_tbptt_oracle_last_cell}} ({{mm_flood_bwd_tbptt_selfcorr}}%) |

How to read the probe:

- **At the same schedule (S = 16, k_in = 1, the table above)** both DRM and fwd do better on noised clean targets than on their own states, and the gap is about as large for DRM as for fwd (larger on flood: DRM {{mm_flood_drm_own_last_cell}}% vs {{mm_flood_drm_oracle_last_cell}}%, fwd {{mm_flood_fwd_own_last_cell}}% vs {{mm_flood_fwd_oracle_last_cell}}%). This schedule is fwd's own dev choice but a poor one for DRM, and near the end of it the oracle input is almost all clean visible cells, so a high oracle number partly measures copying. At this schedule the probe does not single out one-step denoising.
- **At each method's own dev-best 16-Phi schedule** ([mismatch_own_schedule.png](figures/mismatch_own_schedule.png); traces are recorded only for 16-Phi schedules, so this can differ from the headline schedule) the own-vs-oracle gap at the last step is {{mmo_flood_fwd_gap_last}} pp for fwd against {{mmo_flood_drm_gap_last}} pp for DRM on flood, {{mmo_gravity_fwd_gap_last}} vs {{mmo_gravity_drm_gap_last}} pp on gravity, and {{mmo_shift_fwd_gap_last}} vs {{mmo_shift_drm_gap_last}} pp on shift. TBPTT's own schedule is S = 1, which has no visible tokens and no re-masking, so its own and oracle inputs are the same full mask and its gap ({{mmo_flood_bwd_tbptt_gap_last}} pp) is **0 by construction**, not a finding. fwd and DRM are also compared at the last step of different schedules (fwd {{mmo_flood_fwd_sched}}, DRM {{mmo_flood_drm_sched}} on flood), which means different mask levels.
- **Self-correction** of wrong visible cells is low for every denoiser (fwd {{mm_flood_fwd_selfcorr}}%, DRM {{mm_flood_drm_selfcorr}}% on flood at S = 16, k_in = 1). DRM's advantage therefore does not come from fixing visible tokens; it is consistent with the k latent steps inside each denoising step (paper §6), but this toy does not isolate that either.
- **Bottom line:** the symptom the paper describes (a one-step denoiser does worse on its own states than on training-distribution states) is present and is larger for fwd than for DRM at their own schedules. This is consistent with paper §1 and §3.1.1. It is a correlation within this probe, not a causal test.

![Denoising trajectory on one flood eval example for DRM and for one-step denoising, each at its dev-chosen schedule. Each column shows the masked input state (white = MASK) and the predicted grid at one outer step; a tick marks a fully correct prediction.](figures/trajectory.png)

*[figures/trajectory.png](figures/trajectory.png): data in [results/demo/trajectories.json](results/demo/trajectories.json), from [scripts/demo_trajectories.py](scripts/demo_trajectories.py), which retrains the seed-100 flood models and asserts that they reproduce the main-grid pass@1 to within 1e-6 (the tolerance was loosened from 1e-9 in round 0 because the main grid stores a float32 mean). The example is the first of 6 recorded eval examples where DRM ends correct and fwd ends wrong: illustrative, not typical.*

## Training timelines and the k sweep

![Dev pass@1 (S=4, k_in=4, 200 dev examples) against training Phi applications used, for each method on all five tasks.](figures/learning_curves.png)

*[figures/learning_curves.png](figures/learning_curves.png): data in [results/summary.json](results/summary.json) (`learning_curves`), logged on **dev** examples only during training ([drm/train.py](drm/train.py)).*

![Eval pass@1 on gravity and flood against the training recursion window k in {1, 2, 4, 8} at a fixed 12,000-Phi training budget, for DRM and bwd-TBPTT, at S=16/k, k_in=k (solid), at fixed latent S=1, k_in=16 (dotted), and for TBPTT at its horizon-matched latent S=1, k_in=4k (dashed). Flood DRM native: k=1 {{ks_flood_drm_k1}}, k=4 {{ks_flood_drm_k4}}, k=8 {{ks_flood_drm_k8}}.](figures/ksweep.png)

*[figures/ksweep.png](figures/ksweep.png): data in [results/summary.json](results/summary.json) (`ksweep`), 4 eval seeds (100-103); script [scripts/make_figures.py](scripts/make_figures.py).*

| k | flood DRM native (S=16/k, k_in=k) | flood TBPTT horizon latent (S=1, k_in=4k) | flood TBPTT fixed latent (S=1, k_in=16) | gravity DRM native | gravity TBPTT horizon latent | gravity TBPTT fixed latent |
|---|---|---|---|---|---|---|
| 1 | {{ks_flood_drm_k1}} | {{ksh_flood_bwd_tbptt_k1}} | {{ksl_flood_bwd_tbptt_k1}} | {{ks_gravity_drm_k1}} | {{ksh_gravity_bwd_tbptt_k1}} | {{ksl_gravity_bwd_tbptt_k1}} |
| 2 | {{ks_flood_drm_k2}} | {{ksh_flood_bwd_tbptt_k2}} | {{ksl_flood_bwd_tbptt_k2}} | {{ks_gravity_drm_k2}} | {{ksh_gravity_bwd_tbptt_k2}} | {{ksl_gravity_bwd_tbptt_k2}} |
| 4 | {{ks_flood_drm_k4}} | {{ksh_flood_bwd_tbptt_k4}} | {{ksl_flood_bwd_tbptt_k4}} | {{ks_gravity_drm_k4}} | {{ksh_gravity_bwd_tbptt_k4}} | {{ksl_gravity_bwd_tbptt_k4}} |
| 8 | {{ks_flood_drm_k8}} | {{ksh_flood_bwd_tbptt_k8}} | {{ksl_flood_bwd_tbptt_k8}} | {{ks_gravity_drm_k8}} | {{ksh_gravity_bwd_tbptt_k8}} | {{ksl_gravity_bwd_tbptt_k8}} |

At a fixed 12,000-Phi training budget, DRM at its native schedule (S = 16/k, k_in = k) improves steeply from k = 1, which *is* one-step denoising, to k = 4. On flood it then plateaus ({{ks_flood_drm_k4}} at k = 4, {{ks_flood_drm_k8}} at k = 8). On gravity it keeps rising to {{ks_gravity_drm_k8}} at k = 8, above TBPTT's best value ({{ksh_gravity_bwd_tbptt_k4}} at k = 4). TBPTT is trained on n_sup·k = 4k latent steps from a full mask, so its schedule matched to that horizon is S = 1, k_in = 4k (fixed by the config, not chosen post hoc). The fixed S = 1, k_in = 16 schedule matches only at k = 4 and understates TBPTT at small k (gravity k = 2: {{ksh_gravity_bwd_tbptt_k2}} at its horizon vs {{ksl_gravity_bwd_tbptt_k2}} at 16). On both schedules TBPTT peaks at k = 4 and drops at k = 8. So DRM gains more from a larger k than TBPTT does here, in line with the paper's Table 5. The gravity k = 8 result suggests that the k = 4 DRM in the main grid is not DRM's best configuration on gravity, but k was fixed at 4 for every method before the run. The sweep is exploratory: 4 seeds, not in the Holm family, and **the k ≠ 4 runs reuse the lr tuned at k = 4**.

## How this maps to the paper (§3)

| paper claim (§3, §5, App. F) | this toy |
|---|---|
| Forward (one-step) training misaligns train and test, which causes compounding and greedy errors (§1, §3.1.1) | **Consistent: symptom observed, cause not isolated.** At their own schedules, fwd's own-vs-oracle gap is larger than DRM's (flood {{mmo_flood_fwd_gap_last}} vs {{mmo_flood_drm_gap_last}} pp); at a shared S = 16, k_in = 1 schedule it is not. DRM beats fwd by {{d_pooled_drm_vs_fwd}} pooled, from the three tasks that need depth. |
| Masked diffusion without recursion fails badly (Table 1: about 0%) | Consistent: `fwd` reaches {{p1m_gravity_fwd}}% on gravity, {{p1m_symfill_fwd}}% on symfill and {{p1m_shift_fwd}}% on shift. |
| DRM's noise curriculum helps compared with recursion from noise without it (§3.2, §6) | **Not reproduced.** Against `bwd_k` with its schedule chosen among ≤ 16 Phi: {{d_pooled_drm_vs_bwd_k}} pooled, n.s.; at `bwd_k`'s native 4-step schedule: {{snat_pooled_d}}. The only robust per-task difference is symfill, where DRM is behind ({{d_symfill_drm_vs_bwd_k}}). Gravity ({{d_gravity_drm_vs_bwd_k}} under the protocol) falls to {{snat_gravity_d}}, n.s., at the native schedule, so it reflects dev-selection noise and is not evidence for the curriculum. The round-1 "+10.7 pp" depended on forcing `bwd_k` 4× past its training window (review r2). |
| DRM beats TRM (Table 1) | **Not reproduced.** DRM wins flood and (by little) shift, but loses gravity, symfill and the pooled mean to TBPTT ({{d_pooled_drm_vs_bwd_tbptt}}). There is no scratchpad, warm-up loop or halting here, and TBPTT gets 4 windows per batch; the paper's App. F.8 also reports TRM ahead on Sudoku. |
| DRM gains more per recursive step k than TRM (Table 5) | See the k-sweep table above; exploratory, 4 seeds, lr not re-tuned per k. |
| DRM uses extra test-time compute well | Consistent on flood (DRM rises to {{cc_flood_drm_64}}% at 64 Phi while latent TBPTT drifts down to {{cc_flood_bwd_tbptt_64}}%); flat on gravity. |

See [docs/paper_mapping.md](docs/paper_mapping.md) for the equation-by-equation mapping and for the simplifications. None of the following are used, for any method: scratchpad z, warm-up loops, Q-head and halting, EMA, task embedding, augmentation and voting, or SPRM.

## Pipeline and reproduction

```mermaid
flowchart TD
  T["run_grid.py tune<br/>dev split, seeds 1-2, 8 lrs x 5 tasks x 5 methods"] --> U["tune.py --fail-on-edge<br/>lr + schedule (S x k_in <= 16) per task x method"]
  U --> G["git commit + push of dev outputs<br/>(fails hard; before any eval run)"]
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

**Compute used (CPU only, no GPU).** The single-threaded runs behind the current results sum to **{{cpu_h_total}} CPU-hours** of training plus evaluation: dev tuning {{cpu_h_dev}} h, eval grid {{cpu_h_eval}} h, k-sweep {{cpu_h_ksweep}} h (from `train_seconds` + `eval_seconds` in the raw metas). With 7 workers on an 8-core box this was spread over {{run_first_start}} to {{run_last_finish}} EDT, including interruptions. A DRM run trains in about {{train_s_drm}} s on one core. Kaggle GPU access was available but not used, so every result in this repo comes from the same CPU environment. **Total spent** also includes the {{n_runs_superseded}} round-1 eval runs that round 2 superseded ({{cpu_h_superseded}} h, [results/r1_c4532b5/](results/r1_c4532b5/)): ≈{{cpu_h_spent_r12}} CPU-hours for rounds 1-2, plus {{cpu_h_r0}} h for round 0's {{n_runs_r0}} runs ([results/r0_932be94/](results/r0_932be94/)). These totals **exclude** the lineage-verification runs (short configs run under both code versions plus a few full re-runs, in rounds 2 and 3; their outputs were not kept) and the pilots in [docs/pilots.md](docs/pilots.md).

## Verification

- **{{n_tests}} tests** cover the pipeline wiring end to end, not just helpers. Examples: training Phi counts and budgets per method; DRM and fwd starting from corrupted targets while bwd starts from a full mask; TBPTT detaching between windows; zero_grad before every backward; loss targets being clean labels; the remask schedule and the per-seed inference generator; eval metrics using the eval split while the learning curve uses dev; the main grid using tuned lrs, all 5 tasks and all 12 eval seeds; tuning choosing only among schedules with at most 16 Phi, sending exact ties to the lowest lr and then to menu order, and flagging both lr edges; tuning.json and the README keeping the code hash the results were computed under after a regeneration at a newer HEAD; the lineage comparison detecting any output change; collision-free run names that include budget and radius, refusal to overwrite a different config, and input validation; checkpoints that reproduce the stored pass@1; the 95% CI width; aggregation pairing by seed, applying one Holm family, and refusing partial, extra or stale grids; significance decided by Holm-adjusted p; CPU-hours from train + eval seconds; start time taken before training; a distinct inference generator per schedule; `pipeline.sh` committing and pushing all dev outputs before eval with no ignored failure; a k-sweep horizon fixed by the config; the code-hash lineage; and no machine paths or token-like strings in the repo.
- **Mutation check** ([scripts/mutation_check.py](scripts/mutation_check.py), [results/mutation.csv](results/mutation.csv)): an unmutated control run must pass first, and a mutant counts as killed only when pytest exits 1. **{{mut_killed}}/{{mut_total}} killed**, including the reviewer's round-1 mutants ({{mut_rev_killed}}/{{mut_rev_total}}), round-2 mutants ({{mut_rev2_killed}}/{{mut_rev2_total}}) round-3 mutants plus mine for the round-3 code ({{mut_rev3_killed}}/{{mut_rev3_total}}) round-4 mutants of the round-3 code ({{mut_rev4_killed}}/{{mut_rev4_total}}) and round-5 mutants of the claims formatting and pipeline order ({{mut_rev5_killed}}/{{mut_rev5_total}}), {{mut_r1_total}} mutants of the round-1 code ({{mut_r1_killed}}/{{mut_r1_total}}) and {{mut_r2_total}} of the round-2 code ({{mut_r2_killed}}/{{mut_r2_total}}). The README-sync test skips inside mutation copies (they have no `results/`), so README drift is checked only in the full tree.
- **Provenance:** every raw result stores the code hash, its config hash, ISO start/end timestamps, load average and wall time. Round 1: dev outputs and `tuning.json` were committed (3d7f9c4) at 18:18:47 EDT, before the first eval run started, and pushed at 18:19:18, before any eval run finished. Round 2: the re-tuned `tuning.json` was committed and pushed (26afcd8, 22:51:08) before the 96 new eval runs started.
- **Code-hash lineage.** All results were computed under code hash `{{code_hash_parent}}`. The round-2 m5 fix (run names, overwrite guard, radius check, a comment) changed hashed files, so the current hash is `{{code_hash}}`. [scripts/verify_lineage.py](scripts/verify_lineage.py) checked out the parent commit with `git worktree` and ran the same short configs under both versions: every method × 2 tasks × radius {1, 0} × k {2, 4}, scored on all 25 schedules. It also re-ran full committed runs under the new code: one DRM eval run per task, a k = 1 k-sweep run and dev runs. Every output was identical except timing fields (evidence in [results/code_lineage.json](results/code_lineage.json)). The empirical check is a sample, not a proof. It is enough here because of a static argument, recorded in the lineage file and [docs/method.md](docs/method.md): the diff touches only naming, validation, the overwrite guard and hash acceptance; every RNG derives from config fields (seed, data_seed, S, k), never from run names; and the pipeline never reads checkpoints back. `tune.py`, `aggregate.py` and `run_one.py` accept a hash only through such a verified link, and the results keep the hash they were computed under (`tuning.json` `meta.code_hash`, `summary.json` `meta.computed_under`) even when regenerated at a newer HEAD. Any further code change breaks the chain until it is verified again. Training is deterministic per seed; the trajectory demo retrains two models and checks them against the grid.

## Limitations

- This is a toy: 6×6 grids, about 52k parameters, one model per task, no augmentation. Effects here say nothing direct about ARC-AGI.
- The lr is chosen from 2 dev seeds, and at lr ≥ 1.6e-2 some runs diverge, so the choice is noisy. Cells still at a grid edge after the last extension: {{lr_edges}}. These are one-step denoisers, near 0% on gravity and about 5% on shift at every lr. The grid was only ever extended upward; round 1's one lower-edge cell (gravity fwd at 5e-4) was never extended downward.
- The k-sweep reuses the k = 4 lr for every k and has 4 seeds.
- The seed is the replication unit. CIs resample seeds only, and the 1,000 eval examples are shared across seeds. The percentile bootstrap under-covers somewhat at 12 seeds; t-intervals are in the tables.
- Inference uses evenly spaced remask levels and uniform remasking, with no confidence-based remasking.
- No LICENSE file yet; that is the repository owner's decision.
