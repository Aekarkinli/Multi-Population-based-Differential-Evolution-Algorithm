"""
Multi-population based Differential Evolution (MDE).

Karkinli, A.E. (2023). Detection of object boundary from point cloud by using
multi-population based differential evolution algorithm.
Neural Computing and Applications, 35, 5193-5206.
https://doi.org/10.1007/s00521-022-07969-w

This module is a line-by-line port of ``algo_MDE.m`` (same repository) and has
been checked statistically against the MATLAB implementation
(see ``verification/comparison.md``).

Copyright (c) 2022-2026 Ahmet Emin Karkinli. MIT License.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray, float]

__all__ = ["MDEResult", "minimize"]


@dataclass
class MDEResult:
    """Result of :func:`minimize`.

    Attributes
    ----------
    x : ndarray
        Best solution found.
    fun : float
        Objective value of ``x``.
    nit : int
        Number of iterations performed.
    nfev : int
        Number of objective function evaluations (3*N + nit*N).
    history : ndarray
        Best objective value after each iteration.
    """

    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    history: np.ndarray = field(repr=False)


def _scale(rng: np.random.Generator) -> float:
    # scale = |a - rand^b| * randn^d,   a ~ U{0,1}, b ~ U{1..10}, d ~ U{1..5}
    return abs(rng.integers(0, 2) - rng.random() ** rng.integers(1, 11)) * rng.standard_normal() ** rng.integers(1, 6)


def _trial(B: np.ndarray, best: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mutation + parameter-free crossover (Eqs. 7-14)."""
    N, D = B.shape
    temp = np.empty_like(B)
    for i in range(N):
        scale = _scale(rng)
        while True:  # r = permute(N, 2) such that r(1) != i and r(2) != i
            r1, r2 = rng.choice(N, 2, replace=False)
            if r1 != i and r2 != i:
                break
        dx = np.where(rng.random(D) < 0.5, B[r1], best)   # elitist / random component
        dy = np.where(rng.random(D) < 0.5, B[i], B[r2])
        temp[i] = B[r2] + scale * (dx - dy)
    c = 1 if rng.random() ** rng.integers(1, 6) < 0.5 else D
    cmap = np.abs(rng.integers(0, 2, (N, c)) - rng.random((N, D)) ** rng.integers(1, 6, (N, c))) < 0.5
    return B + cmap * (temp + noise - B)


def _noise(B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # noise = B .* 10^phi .* (rand(N,1) - 0.5),   phi ~ U{-12..-9}   (Eq. 13)
    N, D = B.shape
    return B * 10.0 ** rng.integers(-12, -8, (N, D)) * (rng.random((N, 1)) - 0.5)


def _bound(X: np.ndarray, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator, boundary: str) -> np.ndarray:
    if boundary == "clip":
        return np.clip(X, lb, ub)
    out = (X < lb) | (X > ub)          # Eq. 15: resample out-of-bounds variables
    if out.any():
        X = np.where(out, lb + rng.random(X.shape) * (ub - lb), X)
    return X


def minimize(
    func: Callable,
    lb: ArrayLike,
    ub: ArrayLike,
    dim: Optional[int] = None,
    pop_size: int = 30,
    max_iter: int = 1000,
    max_nfev: Optional[int] = None,
    seed: Optional[int] = None,
    vectorized: bool = False,
    boundary: str = "resample",
    verbose: bool = False,
) -> MDEResult:
    """Minimise ``func`` with the Multi-population based Differential Evolution algorithm.

    MDE has no algorithm-specific control parameters (no F, no CR).

    Parameters
    ----------
    func : callable
        Objective function. ``func(x) -> float`` for ``x`` of shape ``(D,)``, or,
        with ``vectorized=True``, ``func(X) -> (n,)`` for ``X`` of shape ``(n, D)``.
    lb, ub : float or array_like
        Lower and upper bounds. Scalars are broadcast to ``dim`` variables.
    dim : int, optional
        Problem dimension; required only when ``lb`` and ``ub`` are scalars.
    pop_size : int, default 30
        N, the size of the sub-population evolved in each iteration.
        The full population holds ``3 * N`` vectors.
    max_iter : int, default 1000
        Maximum number of iterations (MaxCycle). Each iteration costs N evaluations.
    max_nfev : int, optional
        Stop when this number of function evaluations has been reached.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`.
    vectorized : bool, default False
        Whether ``func`` evaluates a whole population at once.
    boundary : {"resample", "clip"}, default "resample"
        Out-of-bounds handling. ``"resample"`` follows Eq. 15 of the paper.
    verbose : bool, default False
        Print the best value after each iteration.

    Returns
    -------
    MDEResult

    Examples
    --------
    >>> import numpy as np
    >>> from mde_optimizer import minimize
    >>> res = minimize(lambda x: np.sum(x**2), lb=-100, ub=100, dim=10, max_iter=500, seed=1)
    >>> res.fun < 1e-6
    True
    """
    lb_arr = np.atleast_1d(np.asarray(lb, dtype=float))
    ub_arr = np.atleast_1d(np.asarray(ub, dtype=float))
    if dim is not None:
        lb_arr = np.broadcast_to(lb_arr, (dim,)).copy()
        ub_arr = np.broadcast_to(ub_arr, (dim,)).copy()
    if lb_arr.shape != ub_arr.shape:
        raise ValueError("lb and ub must have the same length (or pass dim with scalar bounds)")
    if np.any(lb_arr >= ub_arr):
        raise ValueError("every lower bound must be smaller than the upper bound")
    if boundary not in ("resample", "clip"):
        raise ValueError("boundary must be 'resample' or 'clip'")
    N = int(pop_size)
    if N < 3:
        raise ValueError("pop_size must be >= 3")

    rng = np.random.default_rng(seed)
    D = lb_arr.size
    t = 3

    def evaluate(X: np.ndarray) -> np.ndarray:
        if vectorized:
            return np.asarray(func(X), dtype=float).reshape(-1)
        return np.array([func(x) for x in X], dtype=float)

    A = lb_arr + rng.random((t * N, D)) * (ub_arr - lb_arr)
    fitA = evaluate(A)
    nfev = t * N
    k = int(np.argmin(fitA))
    best, best_f = A[k].copy(), float(fitA[k])
    noise = np.zeros((N, D))
    history = []
    nit = 0
    for nit in range(1, max_iter + 1):
        if max_nfev is not None and nfev + N > max_nfev:
            nit -= 1
            break
        j0 = rng.permutation(t * N)[:N]
        B, fitB = A[j0].copy(), fitA[j0].copy()
        T = _bound(_trial(B, best, noise, rng), lb_arr, ub_arr, rng, boundary)
        fitT = evaluate(T)
        nfev += N
        better = fitT < fitB
        B[better], fitB[better] = T[better], fitT[better]
        A[j0], fitA[j0] = B, fitB
        k = int(np.argmin(fitA))
        best, best_f = A[k].copy(), float(fitA[k])
        history.append(best_f)
        if verbose:
            print(f"{nit:5d} --> {best_f:.16f}")
        noise = _noise(B, rng)
    return MDEResult(x=best, fun=best_f, nit=nit, nfev=nfev, history=np.asarray(history))
