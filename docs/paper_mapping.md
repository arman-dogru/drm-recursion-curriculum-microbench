# Mapping to the DRM paper

Paper: Cameron et al., *One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models*, [arXiv:2604.18839](https://arxiv.org/html/2604.18839v1). Official code: [wwwwwwwwz/DenoisingRecursionModels](https://github.com/wwwwwwwwz/DenoisingRecursionModels). All credit for the method belongs to the authors. This repo is an independent CPU toy that does not use their code.

| paper | here | notes |
|---|---|---|
| §3.1, Eq. 1-2: looped model `H_t = Phi(H_{t-1}, X)`, decoder softmax(dec(H)) | `RecursiveModel.phi`, `.logits` ([`drm/model.py`](../drm/model.py)) | one shared pre-norm transformer layer, input re-injected every step |
| §3.1.1, Eq. 3: forward training (one-step denoising to the clean target) | `fwd`, `fwd_steps` ([`drm/train.py`](../drm/train.py)) | reconstruction target Y, no tau conditioning (same for all methods) |
| §3.1.2, Eq. 4-5: backward training with TBPTT and deep supervision | `bwd_tbptt` | 4 windows × k = 4, stop-gradient at window boundaries, CE plus an optimiser step per window, as in TRM |
| (no-TBPTT backward, single window) | `bwd_k` | DRM with tau = 1: isolates the curriculum |
| §3.2, Eq. 6: DRM, corrupt the target and do k recursive steps, CE at H_k | `drm` | k = 4 as in the 7M paper config; no warm-up loops |
| §4.3, Eq. 8: cosine masking schedule, uniform "salt and pepper" masks, separate label embedding | [`drm/corruption.py`](../drm/corruption.py), `lab_emb` | identical |
| §4.3: generate-and-remask inference, tokens not frozen | [`drm/infer.py`](../drm/infer.py) | evenly spaced s_i instead of sorted uniform samples |
| §3.2 "(d) all methods map from maximal corruption at inference" | every method is evaluated from a full mask on the same schedule grid | |
| App. F.3, Table 5: k sweep | k-sweep stage (k ∈ {1, 2, 4, 8} at a fixed FLOP budget) | |
| App. F.2: permutation tests | paired sign-flip tests over training seeds, with Holm | |

Not reproduced, and the same for every method here: the TRM scratchpad z (the paper's App. F.7 finds it unnecessary), no-grad warm-up loops (App. F.3 Table 6 finds they matter a lot for DRM at k = 1), the Q-head and halting, EMA, the task embedding, augmentation and voting, and SPRM (§3.3).
