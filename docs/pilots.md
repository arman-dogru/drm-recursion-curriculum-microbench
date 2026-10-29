# Pilot runs (dev split only, not used for any reported number)

All pilots were single-seed (seed 1) dev runs made while choosing the setup. Their logs are in `logs/pilot*/`. The run JSONs (with `meta.config` and `meta.code_hash`) are in `logs/pilot_json/`. Fields that did not exist yet when a pilot ran (`radius`, grid size) are inferred from the code history and marked as such.

| logs | code hash | grid | attention | layers | Phi budget | lr | result (dev pass@1 at S=4, k_in=4) |
|---|---|---|---|---|---|---|---|
| `logs/pilot/` | 6e1b326315953000 | 8×8 (inferred) | **global** (inferred: no `radius` field yet) | 1 | 8000 | 1e-3 | flood: fwd 0.11, DRM 0.10, TBPTT 0.08, bwd_k 0.02; gravity: all 0.00; symfill: all ≤ 0.03 |
| `logs/pilot2/` | 327f97575bb1a828 | 6×6 | radius 1 | 1 | 12000 | 2e-3 | flood DRM 0.42 vs fwd 0.00; gravity DRM 0.05; symfill all ≤ 0.004 |
| `logs/pilot3/` | 2cc5ba49c314464d | 6×6 | radius 1 | 1 | 12000 | 1e-3 | flood DRM 0.58, fwd 0.33, TBPTT 0.29; gravity DRM 0.66, others 0.00 |
| `logs/pilot4/` `_r1` / `_r2` | 846c8a4f00a5e9c8 | 6×6 | radius 1 / radius 2 | 1 | 12000 | 1e-3 | colormap all 1.00 (r1); shift DRM 0.12 (r1); gravity DRM 0.79 (r2) |

Round 0 of the README said that "in a dev-only pilot with global attention, one Phi step solved most tasks". **No committed pilot supports that**: the only global-attention pilot (`logs/pilot/`) shows no method solving gravity or symfill. The sentence was removed in round 1. The choice of radius-1 attention is a design decision (it makes recursion necessary), not an empirical finding.
