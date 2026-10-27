import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from _impl import load

ex = load(__file__)
from helpers import make_blobs  # noqa: E402


# ---- Part A ----
@pytest.mark.parametrize("a,b", [((8, 1, 6), (7, 1)), ((3,), (4, 3)), ((), (2, 2)), ((5, 4), (1,)),
                                 ((1, 3, 1), (2, 1, 5)), ((256, 256, 3), (3,))])
def test_broadcast_shape_matches_torch(a, b):
    assert ex.broadcast_shape(a, b) == tuple(torch.broadcast_shapes(a, b))
    assert ex.broadcast_shape(b, a) == tuple(torch.broadcast_shapes(a, b))


def test_broadcast_shape_rejects():
    for a, b in [((3,), (4,)), ((2, 3), (3, 2)), ((8, 4, 3), (2, 1))]:
        with pytest.raises(ValueError):
            ex.broadcast_shape(a, b)


def test_grad_of():
    x = np.array([1.0, -2.0, 0.5])
    g = ex.grad_of(lambda t: (t ** 3).sum(), x)
    assert isinstance(g, np.ndarray) and g.shape == (3,)
    assert np.allclose(g, 3 * x ** 2)
    # gradient of logsumexp is softmax (lesson 14)
    g2 = ex.grad_of(lambda t: torch.logsumexp(t, dim=0), x)
    assert np.allclose(g2, np.exp(x) / np.exp(x).sum())
    # matrix input: d/dW sum(W @ v) = 1 v^T
    W = np.arange(6.0).reshape(2, 3)
    v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert np.allclose(ex.grad_of(lambda T: (T @ v).sum(), W), np.ones((2, 1)) * v.numpy())


# ---- Part B ----
def test_my_linear_matches_nn_linear():
    torch.manual_seed(0)
    m = ex.MyLinear(5, 3)
    names = dict(m.named_parameters())
    assert set(names) == {"weight", "bias"}
    assert m.weight.shape == (3, 5) and m.bias.shape == (3,)
    assert isinstance(m.weight, nn.Parameter)
    k = 1 / math.sqrt(5)
    assert m.weight.abs().max() <= k and m.bias.abs().max() <= k
    assert m.weight.std() > 0.1 * k                     # actually random, not zeros
    x = torch.randn(7, 5)
    assert torch.allclose(m(x), F.linear(x, m.weight, m.bias), atol=1e-6)
    ref = nn.Linear(5, 3)
    ref.load_state_dict(m.state_dict())                 # same names & shapes as torch's own
    assert torch.allclose(ref(x), m(x), atol=1e-6)
    nb = ex.MyLinear(5, 3, bias=False)
    assert len(list(nb.parameters())) == 1
    assert torch.allclose(nb(x), x @ nb.weight.T)
    # works on extra leading dims, like nn.Linear: (B, T, in) -> (B, T, out)
    assert m(torch.randn(2, 4, 5)).shape == (2, 4, 3)


def test_mlp_and_param_count():
    torch.manual_seed(0)
    m = ex.MLP([4, 16, 8, 3])
    assert ex.count_parameters(m) == (4 * 16 + 16) + (16 * 8 + 8) + (8 * 3 + 3)
    assert len(list(m.parameters())) == 6               # registered via ModuleList
    out = m(torch.randn(10, 4))
    assert out.shape == (10, 3)
    assert (out < 0).any()                              # last layer has no ReLU
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    assert ex.count_parameters(nn.Linear(10, 2)) == 22


def test_blob_dataset_and_loader():
    ds = ex.BlobDataset(50, seed=1)
    assert len(ds) == 150
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (2,) and x.dtype == torch.float32
    assert isinstance(y, torch.Tensor) and y.dtype == torch.int64 and y.dim() == 0
    Xn, yn = make_blobs(50, seed=1)
    assert np.allclose(ds[149][0].numpy(), Xn[149]) and int(ds[149][1]) == yn[149]
    xb, yb = next(iter(DataLoader(ds, batch_size=32, shuffle=False)))
    assert xb.shape == (32, 2) and yb.shape == (32,)


def test_training_loop_on_blobs():
    torch.manual_seed(0)
    ds = ex.BlobDataset(100, seed=0)
    loader = DataLoader(ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
    model = ex.MLP([2, 32, 3])
    hist = ex.train(model, loader, epochs=15, lr=1e-2)
    assert len(hist) == 15 and all(isinstance(h, float) for h in hist)
    assert hist[-1] < 0.5 * hist[0]
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])
    acc = ex.accuracy(model, X, y)
    assert isinstance(acc, float) and acc > 0.95


def test_accuracy_no_grad():
    lin = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(2))
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.5], [0.1, 0.0]])
    assert ex.accuracy(lin, X, torch.tensor([0, 1, 0, 1])) == pytest.approx(0.75)


def test_spiral_classifier():
    model, acc = ex.train_spiral_classifier(seed=0)
    assert isinstance(model, nn.Module)
    assert acc > 0.95


# ---- Part C ----
def test_scaled_tanh_forward_and_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    y = ex.scaled_tanh(x, a)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.tanh(0.7 * x))
    assert y.grad_fn is not None                         # it's a node in the graph
    assert torch.autograd.gradcheck(ex.ScaledTanh.apply, (x, a), eps=1e-6, atol=1e-6)


def test_scaled_tanh_backward_values_and_shapes():
    x = torch.tensor([[0.5, -1.0], [2.0, 0.0]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    ex.scaled_tanh(x, a).sum().backward()
    t = torch.tanh(1.5 * x.detach())
    assert torch.allclose(x.grad, 1.5 * (1 - t ** 2))
    assert a.grad.shape == () and torch.allclose(a.grad, (x.detach() * (1 - t ** 2)).sum())
    # usable inside a model: train the gain a to fit y = tanh(2 x)
    torch.manual_seed(1)
    xs = torch.linspace(-2, 2, 50, dtype=torch.float64)
    target = torch.tanh(2 * xs)
    gain = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([gain], lr=0.5)
    for _ in range(300):
        loss = ((ex.scaled_tanh(xs, gain) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(gain.item() - 2.0) < 0.05
