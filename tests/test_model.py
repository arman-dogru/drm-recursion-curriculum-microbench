import torch

from drm.corruption import corrupt, mask_fraction, n_masked
from drm.model import RecursiveModel, full_mask, local_mask
from drm.tasks import H, W, M, MASK


def test_mask_counts_follow_cosine_schedule():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 8, (5, M))
    tau = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    out = corrupt(y, tau, g)
    expect = [int(mask_fraction(float(t)) * M + 1e-6) for t in tau]
    assert (out == MASK).sum(1).tolist() == expect
    assert expect[0] == 0 and expect[-1] == M
    keep = out != MASK
    assert torch.equal(out[keep], y[keep])          # visible cells are the clean target
    assert n_masked(torch.tensor([0.5])).item() == expect[2]


def test_local_mask_radius():
    m = local_mask(1)
    assert m.shape == (M, M) and m[0, 0] and m[0, 1] and m[0, W + 1] and not m[0, 2]
    assert local_mask(0) is None


def test_receptive_field_grows_one_cell_per_phi():
    torch.manual_seed(0)
    model = RecursiveModel(16, 2, 1, radius=1).eval()
    x = torch.zeros(1, M, dtype=torch.long)
    x2 = x.clone()
    x2[0, 0] = 5                                      # perturb top-left cell
    h0 = model.embed_y(full_mask(1))
    for k in (1, 2, 3):
        a = model.unroll(h0, model.embed_x(x), k)
        b = model.unroll(h0, model.embed_x(x2), k)
        changed = ((a - b).abs().amax(-1) > 1e-6)[0].reshape(H, W)
        # cell (r, c) can only change if Chebyshev distance to (0,0) <= k-1+... :
        # one Phi uses the input at the cell itself plus one attention hop.
        r, c = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        assert not changed[torch.maximum(r, c) > k].any()
        assert changed[torch.maximum(r, c) == k].any()


def test_global_attention_reaches_everything():
    torch.manual_seed(0)
    model = RecursiveModel(16, 2, 1, radius=0).eval()
    x = torch.zeros(1, M, dtype=torch.long)
    x2 = x.clone()
    x2[0, 0] = 5
    h0 = model.embed_y(full_mask(1))
    a = model.unroll(h0, model.embed_x(x), 1)
    b = model.unroll(h0, model.embed_x(x2), 1)
    assert ((a - b).abs().amax(-1) > 1e-6).all()


def test_label_and_input_embeddings_are_separate():
    model = RecursiveModel(16, 2, 1)
    assert model.in_emb.weight.data_ptr() != model.lab_emb.weight.data_ptr()
    assert model.lab_emb.num_embeddings == MASK + 1
