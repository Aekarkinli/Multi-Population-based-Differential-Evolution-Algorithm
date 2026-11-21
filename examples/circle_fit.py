"""Fit a circle to a noisy point cloud with MDE (Python version of example_circle_fit.m)."""
import numpy as np

from mde_optimizer import minimize

rng = np.random.default_rng(100)
r = 10.0
t = np.linspace(-np.pi, np.pi, 5000)
x = r * (np.sin(t) + 0.05 * (rng.random(t.size) - 0.5))
y = r * (np.cos(t) + 0.05 * (rng.random(t.size) - 0.5))


def fit_circle(P):
    """P: (n, 3) candidates [x0, y0, r] -> (n,) sum of absolute algebraic residuals."""
    x0, y0, rr = P[:, :1], P[:, 1:2], np.abs(P[:, 2:3])
    return np.sum(np.abs((x - x0) ** 2 + (y - y0) ** 2 - rr ** 2), axis=1)


res = minimize(fit_circle, lb=-100, ub=100, dim=3, pop_size=30, max_iter=1000, seed=100, vectorized=True)
x0, y0, rr = res.x[0], res.x[1], abs(res.x[2])
print(f"x0 = {x0:.4f}, y0 = {y0:.4f}, r = {rr:.4f}  (objective {res.fun:.2f}, {res.nfev} evaluations)")
