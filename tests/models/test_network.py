import numpy as np

from wildfire.models import network

PARAMS = dict(width=32, depth=2, dropout=0.1, learning_rate=1e-3, weight_decay=1e-4)


def data(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 1.2).astype(np.int8)
    return x, y, np.where(y == 1, 1.0, 4.0)


def test_build_net_has_the_requested_shape():
    net = network.build_net(6, 32, 3, 0.1)
    assert sum(isinstance(m, network.nn.Linear) for m in net) == 4       # 3 hidden layers + output


def standardised(x, n=45000):
    std = network.Standardiser(x[:n])
    return std.apply(x.copy())


def test_network_learns_and_reports_the_weighted_fit_prevalence():
    x, y, w = data()
    x = standardised(x)
    m = network.Network(PARAMS, seed=0).fit(x[:45000], y[:45000], w[:45000], x[45000:], y[45000:], w[45000:])
    p = m.predict(x[45000:])
    assert p[y[45000:] == 1].mean() > p[y[45000:] == 0].mean() + 0.1 and ((p > 0) & (p < 1)).all()
    assert abs(m.fit_prevalence - np.average(y[:45000], weights=w[:45000])) < 1e-12


def test_the_same_seed_gives_the_same_network():
    x, y, w = data()
    x = standardised(x)
    fit = lambda s: network.Network(PARAMS, seed=s).fit(x[:45000], y[:45000], w[:45000], x[45000:], y[45000:], w[45000:]).predict(x[45000:])
    assert np.array_equal(fit(1), fit(1)) and not np.array_equal(fit(1), fit(2))


def test_weighted_log_loss_matches_a_hand_value():
    import torch
    loss = network.weighted_log_loss(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 3.0]))
    assert abs(float(loss) - np.log(2)) < 1e-6


def test_standardiser_uses_the_given_rows_and_works_in_place():
    x = np.random.default_rng(3).normal(5.0, 2.0, size=(1000, 3)).astype(np.float32)
    std = network.Standardiser(x)
    out = std.apply(x)
    assert out is x and abs(float(x.mean())) < 1e-3 and abs(float(x.std()) - 1) < 1e-2
