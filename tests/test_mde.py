import numpy as np
import pytest

from mde_optimizer import minimize


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin_vec(X):
    return np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)


def test_sphere_converges():
    res = minimize(sphere, lb=-100, ub=100, dim=10, max_iter=1000, seed=1)
    assert res.fun < 1e-12
    assert res.x.shape == (10,)


def test_vectorized_matches_scalar():
    a = minimize(sphere, lb=-5, ub=5, dim=5, max_iter=200, seed=3)
    b = minimize(lambda X: np.sum(X ** 2, axis=1), lb=-5, ub=5, dim=5, max_iter=200, seed=3, vectorized=True)
    assert a.fun == b.fun
    np.testing.assert_array_equal(a.x, b.x)


def test_reproducible_with_seed():
    a = minimize(rastrigin_vec, lb=-5.12, ub=5.12, dim=10, max_iter=300, seed=7, vectorized=True)
    b = minimize(rastrigin_vec, lb=-5.12, ub=5.12, dim=10, max_iter=300, seed=7, vectorized=True)
    c = minimize(rastrigin_vec, lb=-5.12, ub=5.12, dim=10, max_iter=300, seed=8, vectorized=True)
    assert a.fun == b.fun
    assert not np.array_equal(a.x, c.x)


def test_evaluation_budget():
    calls = []

    def f(x):
        calls.append(1)
        return sphere(x)

    res = minimize(f, lb=-1, ub=1, dim=3, pop_size=10, max_iter=50, seed=0)
    assert res.nfev == len(calls) == 3 * 10 + 50 * 10
    assert res.nit == 50
    assert len(res.history) == 50
    assert np.all(np.diff(res.history) <= 0)  # best value never gets worse

    res = minimize(sphere, lb=-1, ub=1, dim=3, pop_size=10, max_iter=1000, max_nfev=205, seed=0)
    assert res.nfev <= 205


@pytest.mark.parametrize("boundary", ["resample", "clip"])
def test_solutions_stay_in_bounds(boundary):
    lb, ub = np.array([1.0, -2.0, 10.0]), np.array([5.0, -1.0, 20.0])  # same-sign bounds included
    seen = []

    def f(x):
        seen.append(x.copy())
        return float(np.sum((x - ub) ** 2))  # optimum on the upper corner

    res = minimize(f, lb, ub, max_iter=200, seed=2, boundary=boundary)
    seen = np.array(seen)
    assert np.all(seen >= lb) and np.all(seen <= ub)
    np.testing.assert_allclose(res.x, ub, atol=1e-3)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        minimize(sphere, lb=[0, 0], ub=[1], seed=0)
    with pytest.raises(ValueError):
        minimize(sphere, lb=1, ub=0, dim=2)
    with pytest.raises(ValueError):
        minimize(sphere, lb=-1, ub=1, dim=2, pop_size=2)
    with pytest.raises(ValueError):
        minimize(sphere, lb=-1, ub=1, dim=2, boundary="wrap")
