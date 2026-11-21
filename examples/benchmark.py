"""Run MDE on a few classic benchmark functions (D = 10, 25 runs each)."""
import numpy as np

from mde_optimizer import minimize

FUNCS = {
    "Sphere": (lambda X: np.sum(X ** 2, axis=1), 100),
    "Rastrigin": (lambda X: np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1), 5.12),
    "Rosenbrock": (lambda X: np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1), 30),
    "Ackley": (lambda X: -20 * np.exp(-0.2 * np.sqrt(np.mean(X ** 2, axis=1)))
               - np.exp(np.mean(np.cos(2 * np.pi * X), axis=1)) + 20 + np.e, 32),
}

for name, (f, bound) in FUNCS.items():
    vals = [minimize(f, -bound, bound, dim=10, pop_size=30, max_iter=1000, seed=s, vectorized=True).fun
            for s in range(25)]
    print(f"{name:<11} median = {np.median(vals):.3e}   mean = {np.mean(vals):.3e}   std = {np.std(vals):.3e}")
