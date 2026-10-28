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

Attention is **local**: a cell attends to cells within Chebyshev distance 1. Information therefore moves at most one cell per Phi step, and each task has a known minimum recursion depth ([`test_receptive_field_grows_one_cell_per_phi`](../tests/test_model.py)). I chose this deliberately. In a first dev-only pilot with global attention, one Phi application could solve most tasks (symfill is a single copy), so the experiment could not separate recursive training from one-step training. The output stream has its own embedding table with colours plus MASK, separate from the input embedding, as in paper §4.3. The model has about 52k parameters, far under the 2M cap.

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

Every run is evaluated on **every** schedule with S, k_in ∈ {1, 2, 4, 8, 16} and S·k_in ≤ 64.

## Protocol: dev/eval separation ([`scripts/run_grid.py`](../scripts/run_grid.py), [`scripts/tune.py`](../scripts/tune.py))

1. **Tune (dev only).** Every (task, method, lr ∈ {5e-4, 1e-3, 2e-3}) is trained, plus lr = 4e-3 on the three eval tasks. I added 4e-3 on dev evidence, because the recursive methods peaked at the 2e-3 edge. Each run is trained with dev seeds {1, 2} and scored on the dev split. For each (task, method), `tune.py` picks the lr **and** the inference schedule from the matched-compute menu S × k_in = 16, {(1,16), (2,8), (4,4), (8,2), (16,1)}, that maximise mean dev pass@1. This gives every baseline its best schedule, so for example the one-step denoiser can choose 16 small steps. For the compute curves it also records the dev-best schedule at every test-compute level.
2. **Main (eval only).** Every (method, task ∈ {shift, gravity, flood}) is retrained at its tuned lr with 10 fresh eval seeds {100, …, 109} and scored on the held-out eval split at its dev-chosen schedule.
3. **k-sweep.** `drm` and `bwd_tbptt` are trained at k ∈ {1, 2, 8} (k = 4 comes from the main grid) on flood with eval seeds 100-103, at a fixed 12000-Phi budget.

**Scope cut, made before any eval run.** At about 10:00 EDT the box CPU slowed about 3.5× (per-run training went from about 140 s to about 510 s). Using **dev results only**, I cut the eval grid. Colormap is solved by every method at every lr (dev pass@1 = 1.00), so it cannot separate them. Symfill is solved by none (dev pass@1 < 0.05 for every method and lr). Both are reported from the dev tuning grid only. The eval grid uses 10 of the 12 reserved eval seeds, and the k-sweep uses flood only. The cut is recorded in [`scripts/run_grid.py`](../scripts/run_grid.py). Each is scored at two pre-registered schedules: native (S = 16/k, k_in = k) and latent (S = 1, k_in = 16).

The learning curve logged during training is measured on 200 **dev** examples, never on eval, and a test checks this.

## Statistics ([`drm/stats.py`](../drm/stats.py), [`scripts/aggregate.py`](../scripts/aggregate.py))

- The unit is the training seed. Methods with the same seed share the init RNG, batch-order RNG and inference RNG, so comparisons are paired by seed.
- **CI:** a percentile bootstrap over seeds (10,000 resamples), seeds only. Examples within a seed are not resampled.
- **Test:** an exact two-sided paired sign-flip permutation test on the per-seed difference. With 10 seeds the smallest attainable p is 2/1024 ≈ 0.002.
- **Multiplicity:** one Holm family over all 16 primary comparisons (DRM vs 4 baselines × (3 eval tasks + the pooled mean over them)). A comparison where all 10 seeds agree in sign survives Holm (16 × 0.002 = 0.031).
- **Provenance:** every raw result records a hash of the code and of its config. `tune.py` and `aggregate.py` refuse (exit 2) a grid that is incomplete or has extra, stale or mixed-code files. `--allow-partial` must write somewhere other than `results/`.

## Train-test mismatch probe

For the S = 16, k_in = 1 schedule, each eval run records per outer step:

- cell accuracy along the model's **own** trajectory;
- the same accuracy when the step's input is the **training distribution**: the clean target masked at the same level (the "oracle state");
- the fraction of visible (unmasked) input cells that are wrong. Under q_tau this is always 0 in training, but at test time it is whatever the model got wrong earlier;
- the fraction of those wrong visible cells the next prediction fixes (self-correction).
