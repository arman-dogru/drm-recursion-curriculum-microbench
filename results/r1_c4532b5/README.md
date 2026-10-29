# Round-1 artifacts superseded in round 2 (c4532b5)

Round 2 review (major 1) changed the headline inference menu from S x k_in = 16 to S x k_in <= 16.
Re-tuning on the committed dev runs changed the chosen lr of 8 (task, method) cells. Their round-1 eval
runs (12 seeds each, json + npz) are moved here unchanged, so the live eval grid holds exactly one lr per cell.
`tuning.json`, `summary.json`, `claim_numbers.json` and `tables/` are the round-1 versions at c4532b5.
All other round-1 eval runs are reused as-is, since only their scored schedule changed.
