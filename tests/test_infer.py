import torch

from drm.corruption import n_masked
from drm.infer import infer, oracle_state_acc
from drm.model import RecursiveModel
from drm.tasks import M, MASK


class Spy(RecursiveModel):
    def __init__(self):
        super().__init__(16, 2, 1)
        self.seen, self.ks = [], []

    def embed_y(self, tok):
        self.seen.append(tok.clone())
        return super().embed_y(tok)

    def unroll(self, h, ex, k):
        self.ks.append(k)
        return super().unroll(h, ex, k)


def test_remask_schedule_and_start_from_full_mask():
    torch.manual_seed(0)
    m = Spy()
    X = torch.randint(0, 8, (5, M))
    S = 4
    infer(m, X, S, 2, torch.Generator().manual_seed(0))
    assert len(m.seen) == S and m.ks == [2] * S
    assert (m.seen[0] == MASK).all()
    for i in range(1, S):
        expect = n_masked(torch.tensor([1.0 - i / S])).item()
        assert ((m.seen[i] == MASK).sum(1) == expect).all()
    assert expect > 0


def test_latent_only_inference_is_one_outer_step():
    m = Spy()
    X = torch.randint(0, 8, (3, M))
    infer(m, X, 1, 16, torch.Generator().manual_seed(0))
    assert len(m.seen) == 1 and m.ks == [16]


def test_visible_tokens_come_from_previous_prediction_not_frozen():
    torch.manual_seed(0)
    m = Spy()
    X = torch.randint(0, 8, (4, M))
    g = torch.Generator().manual_seed(0)
    infer(m, X, 3, 1, g)
    # recompute step-1 prediction and check visible cells of step-2 input equal it
    with torch.no_grad():
        p1 = m.logits(RecursiveModel.unroll(m, RecursiveModel.embed_y(m, m.seen[0]), m.embed_x(X), 1)).argmax(-1)
    vis = m.seen[1] != MASK
    assert torch.equal(m.seen[1][vis], p1[vis])


def test_trace_fields_and_oracle():
    m = RecursiveModel(16, 2, 1)
    X = torch.randint(0, 8, (6, M))
    Y = torch.randint(0, 8, (6, M))
    tr = []
    infer(m, X, 4, 1, torch.Generator().manual_seed(0), trace=tr, Y=Y)
    assert [t["step"] for t in tr] == [1, 2, 3, 4]
    assert tr[0]["visible_frac"] == 0.0
    o = oracle_state_acc(m, X, Y, 4, 1, torch.Generator().manual_seed(0))
    assert o[0]["mask_frac"] == 1.0 and o[-1]["mask_frac"] < 1.0


def test_inference_deterministic_given_generator():
    m = RecursiveModel(16, 2, 1)
    X = torch.randint(0, 8, (6, M))
    a = infer(m, X, 8, 1, torch.Generator().manual_seed(3))
    b = infer(m, X, 8, 1, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
