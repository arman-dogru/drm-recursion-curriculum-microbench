# Method

This is a didactic CPU microbench of the core idea in *One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models* (Cameron, Wang, Ivanov, Bhattacharyya, Chételat and Zhang; Huawei Technologies, University of Toronto and University of Waterloo; [arXiv:2604.18839](https://arxiv.org/html/2604.18839v1); code at [wwwwwwwwz/DenoisingRecursionModels](https://github.com/wwwwwwwwz/DenoisingRecursionModels)). It is not an ARC-AGI result and makes no claim about ARC.

## Tasks ([`drm/tasks.py`](../drm/tasks.py))

Each task is a 6×6 grid with 8 colours. Every input has one exact target, and pass@1 means the whole grid matches.

| task | rule | min. Phi steps under radius-1 attention |
|---|---|---|
| `colormap` | fixed per-cell colour permutation | 1 (local) |
| `shift` | translate every object by (+1 row, +2 cols), no wrap-around | 2 |
| `symfill` | left-right symmetric grid with an occluder inside one half; fill it from the mirror | up to 5 |
| `gravity` | coloured cells fall to the bottom of their column, keeping their order | up to 5 |
| `flood` | mark every empty cell 4-connected to the source through gaps in the walls | path length (often > 5) |

The train, dev and eval splits (2000/500/1000 examples) come from different RNG streams. They are de-duplicated against each other on the input grid, and a test checks this on the data itself.

## Model ([`drm/model.py`](../drm/model.py))

One shared transition operator Phi, which is one pre-norm transformer layer (d = 64, 4 heads, MLP ×4):

`H_t = Phi(H_{t-1}, X) = Block(H_{t-1} + E_in(X))`

Attention is **local**: a cell attends to cells within Chebyshev distance 1. Information therefore moves at most one cell per Phi step, and each task has a known minimum recursion depth ([`test_receptive_field_grows_one_cell_per_phi`](../tests/test_model.py)). This is a design choice: with global attention a single Phi application can in principle move information anywhere (symfill, for example, is a single copy), so the tasks would not *require* recursion. I make no empirical claim about global attention. The only global-attention pilot I ran (8×8 grids, before `radius` existed; [logs/pilot/](../logs/pilot/), configs in [docs/pilots.md](pilots.md)) does not show one-step solutions, and an earlier version of this text that said it did was wrong. The output stream has its own embedding table with colours plus MASK, separate from the input embedding, as in paper §4.3. The model has about 52k parameters, far under the 2M cap.

## Trainers ([`drm/train.py`](../drm/train.py)): paper §3.1.1, §3.1.2 and §3.2

All five trainers use the same model, initialisation (paired by seed), optimiser (AdamW, wd 0.1, 100-step warm-up, then cosine decay to 10%), batch size 64 and grad-clip 1.0.

| id | objective | start state H_0 | gradient-bearing Phi per opt step | opt steps |
|---|---|---|---|---|
| `fwd` | one-step denoising, Eq. 3 | E_y(q_tau(Y)) | 1 | 12000 (FLOP-matched) |
| `fwd_steps` | same, step-matched | E_y(q_tau(Y)) | 1 | 3000 (¼ of the FLOPs) |
| `bwd_k` | backward recursion, one window | E_y(all MASK) | 4 | 3000 |
| `bwd_tbptt` | TRM-style TBPTT, Eq. 4-5: 4 windows of k = 4, stop-gradient between windows, CE plus an opt step per window | E_y(all MASK) | 4 | 3000 (750 batches × 4) |
| `drm` | DRM, Eq. 6: k = 4 steps from the corrupted target, CE to the clean target at H_k | E_y(q_tau(Y)) | 4 | 3000 |

q_tau is the paper's masking process (§4.3, Eq. 8): tau ~ U(0,1), and floor((1 − cos²(πtau/2))·M) uniformly random cells are replaced by MASK. Every trainer except `fwd_steps` spends exactly 12000 gradient-bearing Phi applications per batch row. The tests count these by instrumenting Phi.

`bwd_k` is DRM with tau fixed at 1, so the `drm` vs `bwd_k` comparison isolates the noise curriculum. `fwd` vs `drm` isolates multi-step backprop at matched FLOPs.

Simplifications versus the paper, the same for every method: no TRM scratchpad z, no warm-up (no-grad) loops, no Q-head or early halting, no EMA, no task embedding (one model per task), and no augmentation.

## Inference ([`drm/infer.py`](../drm/infer.py)): paper §3.2 and §4.3 "generate-and-remask"

Every method starts from a fully masked output grid. At each of **S** outer steps, the model embeds the current token grid, applies Phi **k_in** times, decodes the argmax grid, and re-masks floor(r(s_i)·M) uniformly random cells, with s_i = 1 − i/S. Tokens are not frozen. S = 1 is pure latent recursion, which is how a backward-trained model is natively used. One difference from the paper: s_i is evenly spaced, not sorted uniform samples.

Every run is evaluated on **every** schedule with S ∈ {1, 2, 4, 8, 16}, k_in ∈ {1, 2, 4, 8, 16, 32, 64} and S·k_in ≤ 64 (25 schedules). In round 0, k_in stopped at 16, so pure latent recursion (S = 1) was not offered above 16 Phi; that was fixed after review (M3).

## Protocol: dev/eval separation ([`scripts/run_grid.py`](../scripts/run_grid.py), [`scripts/tune.py`](../scripts/tune.py))

1. **Tune (dev only).** Every (task, method, lr ∈ {5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2}) is trained with dev seeds {1, 2} and scored on the dev split: 5 tasks × 5 methods × 8 lrs × 2 seeds = 400 runs. The grid was first {5e-4, …, 1.6e-2}; when flood `bwd_k` peaked at the 1.6e-2 edge on dev (0.47 at 8e-3, 0.79 at 1.6e-2), 3.2e-2 was added for **every** task and method before any eval run (`LR_EXT` in [`scripts/run_grid.py`](../scripts/run_grid.py)). At 3.2e-2 five cells were again at the edge (gravity `drm`, shift `bwd_k` and `fwd`, symfill `drm` and `bwd_tbptt`), so 6.4e-2 was added, again for every cell, and declared in advance to be the **last** extension. Any edges that remain are listed in `results/tuning.json` (`meta.lr_edges`) and in the README. Dev pass@1 at lr ≥ 1.6e-2 is noisy (some runs diverge; e.g. shift `bwd_tbptt` 0.98 at 8e-3, 0.06 at 1.6e-2, 0.46 at 3.2e-2), so 2 dev seeds pick the lr with real noise. For each (task, method), `tune.py` picks the lr **and** the inference schedule that maximise mean dev pass@1 among all 15 schedules with **at most** matched test compute, S × k_in ≤ 16 (`HEADLINE_MENU`). **Tie-break:** lr is the outer loop and the comparison is strict, so an exact tie in mean dev pass@1 goes to the **lowest lr** first and only then to menu order, where the S × k_in = 16 schedules are listed first. (Rounds 0-2 documented this as "ties keep the 16-Phi schedule", which is true only within one lr; review r3 m3.) In round 2 this rule decided symfill `bwd_tbptt`: 3.2e-2 at S1/k16 and 1.6e-2 at S1/k8 both score 0.288 on dev, so 1.6e-2 won and its 12 eval runs were redone. The README reports what the round-1 configuration would have given (`summary.json` `sensitivity.tie_symfill_tbptt_r1`). This gives every baseline its best schedule, so for example the one-step denoiser can choose 16 small steps, and a backward model trained on a 4-step window can stop after 4 latent steps. **Round 2 change (review major 1):** rounds 0 and 1 used only the five S × k_in = 16 schedules {(1,16), (2,8), (4,4), (8,2), (16,1)}, which forced the single-window `bwd_k` (trained on 4 steps) to run 4× past its training horizon. Re-tuning on the committed dev runs, with no new dev runs, changed the lr of 8 cells; only those 8 cells × 12 seeds = 96 eval runs were re-run. Their round-1 eval runs are kept in [`results/r1_c4532b5/`](../results/r1_c4532b5/). For the compute curves it also records, for every test-compute level c, the dev-best schedule among **all** schedules with S·k_in = c, so each method can use its best inference mode (latent or re-masking) at every budget. `tune.py --fail-on-edge` stops the pipeline if any chosen lr is at an end of the grid and strictly better than its neighbour; the list is stored in `results/tuning.json` (`meta.lr_edges`). The grid was only ever extended **upward**: the one lower-edge cell in round 1 (gravity `fwd` at 5e-4) was not extended downward, because `fwd` is at about 0% on gravity at every lr (review r2 n6).
2. **Commit and push, then eval.** `scripts/pipeline.sh` (with `COMMIT=1`) commits `results/raw/dev`, `tuning.json` and the logs, checks that nothing under them is left uncommitted, and pushes, before it launches any eval run. Any of these steps failing stops the pipeline. Every raw result and log line carries an ISO timestamp. Round 1 timeline (EDT, from metas, git and the GitHub activity API): the last dev run finished at 18:18:45; 3d7f9c4 was **committed at 18:18:47, before the first eval run started** (18:18:47); it was **pushed at 18:19:18, before any eval run finished** (first at 18:22:19). Round-1 `pipeline.sh` did not push and ignored a failed commit (`|| true`); both are fixed. Round 2: tuning commit 26afcd8 at 22:51:08, pushed before the first round-2 eval run at 22:51:10.
3. **Main (eval only).** Every (task, method), all five tasks, is retrained at its tuned lr with the 12 reserved eval seeds {100, …, 111} and scored on the held-out eval split at its dev-chosen schedule: 300 runs.
4. **k-sweep (exploratory).** `drm` and `bwd_tbptt` are trained at k ∈ {1, 2, 8} (k = 4 comes from the main grid) on gravity and flood with eval seeds 100-103, at a fixed 12000-Phi budget. Each is scored at two pre-registered schedules: native (S = 16/k, k_in = k) and a **fixed** 16-step latent schedule (S = 1, k_in = 16). For TBPTT the 16-step latent schedule matches its training horizon (n_sup·k = 4k steps unrolled from a full mask) only at k = 4, so round 2 adds a third schedule fixed by the config: horizon-matched latent, S = 1, k_in = 4k for TBPTT (review r2 m3). **The k ≠ 4 runs reuse the lr tuned at k = 4**; lr is not re-tuned per k.

### Round-0 scope cut (reverted)

The round-0 version (PR head 932be94, kept in [`results/r0_932be94/`](../results/r0_932be94/)) cut colormap and symfill from the eval grid, used 10 of the 12 eval seeds, and only gave the 4e-3 lr to the three remaining tasks. The cut was made before any eval run and from dev results only, but it was a mistake:

- The stated reason, "the box CPU slowed about 3.5×", is not what the logs show. In `logs/tune.log` the slowdown came in short bursts: a wave of 7 jobs (jobs 29-35) took about 390 s each and a few later waves (jobs 106-112, 117-119, 130) about 500 s, while every other job took 42-217 s, and the later eval runs had a median of about 160 s.
- The two dropped tasks were exactly the ones where DRM did not win on dev (colormap: every method 100%; symfill: DRM ≈ fwd, both < 1%, and DRM behind `bwd_k`). Dropping them made the headline look better.

Round 1 therefore restores the full suite (5 tasks × 12 seeds), and gives every (task, method) the same 8-point lr grid.

The learning curve logged during training is measured on 200 **dev** examples, never on eval, and a test checks this.

## Statistics ([`drm/stats.py`](../drm/stats.py), [`scripts/aggregate.py`](../scripts/aggregate.py))

- The unit is the training seed. Methods with the same seed share the init RNG, batch-order RNG and inference RNG, so comparisons are paired by seed.
- **CI:** a percentile bootstrap over seeds (10,000 resamples), seeds only. Examples within a seed are not resampled. With n = 12 seeds the percentile bootstrap **under-covers** (it is roughly a z-interval with the ddof=0 SD, about 10% narrower than a t-interval), so `results/tables/*.csv` and `summary.json` also report a Student-t interval (`t_lo`, `t_hi`). The README quotes the bootstrap interval, as pre-registered.
- **Test:** an exact two-sided paired sign-flip permutation test on the per-seed difference. With 12 seeds the smallest attainable p is 2/4096 ≈ 0.0005.
- **Multiplicity:** one Holm family over all 24 primary comparisons (DRM vs 4 baselines × (5 tasks + the pooled mean over all 5)). The pooled mean over the three tasks that need depth (shift, gravity, flood) is reported descriptively, for continuity with round 0, and is not tested.
- **Provenance:** every raw result records a hash of the code and of its config. `tune.py` and `aggregate.py` refuse (exit 2) a grid that is incomplete or has extra, stale or mixed-code files. `--allow-partial` must write somewhere other than `results/`.

## Train-test mismatch probe

Each eval run records, at every schedule with S·k_in = 16, per outer step:

- cell accuracy along the model's **own** trajectory;
- the same accuracy when the step's input is the **training distribution**: the clean target masked at the same level (the "oracle state");
- the fraction of visible (unmasked) input cells that are wrong. Under q_tau this is always 0 in training, but at test time it is whatever the model got wrong earlier;
- the fraction of those wrong visible cells the next prediction fixes (self-correction).

The probe is reported in two ways: at S = 16, k_in = 1 for every method (same schedule), and at each method's own dev-best schedule with exactly 16 Phi (traces exist only for the 16-Phi schedules, so since round 2 this is not always the headline schedule). For TBPTT at S = 1 there are no visible tokens, so its own-vs-oracle gap is 0 by construction. Comparing fwd and DRM at their own schedules compares the last step of different schedules (e.g. S = 16 vs S = 2), i.e. different mask levels. Near the end of a schedule the oracle input is almost entirely clean visible cells, so a high oracle accuracy there partly measures copying. The probe shows a symptom (accuracy on own states is lower than on training-distribution states). It does not, on its own, isolate that symptom as the cause of the one-step denoiser's loss.

## Code-hash lineage: why the link is trusted (review r3 n5)

Round 2's m5 fix changed hashed files (`drm/config.py` run names, `drm/lineage.py`, `scripts/run_one.py`), so the current code hash differs from the one every result was computed under. Rather than re-run everything, [`scripts/verify_lineage.py`](../scripts/verify_lineage.py) records a link in [`results/code_lineage.json`](../results/code_lineage.json). The link rests on two legs:

1. **Static.** The diff between the parent commit and the current code touches only run naming (`Config.run_name`), input validation (radius ≥ 0, CLI checks), the overwrite guard, hash acceptance and comments. No training, inference, data or RNG code changed. Every RNG (data generation, initialisation, corruption, the per-schedule eval generator) is seeded from config fields (`seed`, `data_seed`, `S`, `k`), never from run names or paths. Checkpoints are written but never read back by the pipeline, so a renamed checkpoint cannot change a result.
2. **Empirical.** Short configs (every method × 2 tasks × radius {1, 0} × k {2, 4}, all 25 schedules) are run under both versions, and full committed runs are re-run under the current code. By default that is one DRM eval run per task, a k = 1 TBPTT k-sweep run and a dev run (`default_full()`). All outputs (eval, traces, learning curve, losses and the npz arrays) must be identical except timing fields.

Leg 2 alone is a sample, not a proof; it suffices only because of leg 1. A future diff that touches numerics needs a rerun, not a lineage link. `aggregate.py` records in `meta.computed_under` the hash stored in the raw runs, and `tune.py` records it in `tuning.json`, so a regeneration at a newer HEAD never relabels the results (review r3 m2).

