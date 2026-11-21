"""Run the Python MDE port with the same settings as run_matlab.m and compare."""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from mde_optimizer import minimize  # noqa: E402


def sphere(X): return np.sum(X ** 2, axis=1)
def rastrigin(X): return np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)
def rosenbrock(X): return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)
def ackley(X):
    D = X.shape[1]
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, axis=1) / D))
            - np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / D) + 20 + np.e)
def shifted_sphere(X): return np.sum((X - 3) ** 2, axis=1)


FUNCS = {"sphere": (sphere, -100, 100), "rastrigin": (rastrigin, -5.12, 5.12),
         "rosenbrock": (rosenbrock, -30, 30), "ackley": (ackley, -32, 32),
         "shifted_sphere": (shifted_sphere, 1, 5)}
D, N, MAX_ITER, RUNS = 10, 30, 1000, 25


def main():
    py = {}
    for variant, boundary in (("eq15", "resample"), ("clip", "clip")):
        for name, (f, lo, hi) in FUNCS.items():
            if variant == "clip" and name == "shifted_sphere":
                continue
            py[(variant, name)] = [minimize(f, [lo] * D, [hi] * D, pop_size=N, max_iter=MAX_ITER, seed=r,
                                                vectorized=True, boundary=boundary).fun
                                   for r in range(1, RUNS + 1)]
    ml = {}
    with open(HERE / "matlab_results.csv") as fh:
        for row in csv.DictReader(fh):
            ml.setdefault((row["variant"], row["func"]), []).append(float(row["best"]))

    lines = ["| variant | function | MATLAB median | Python median | MATLAB mean | Python mean | Mann-Whitney p |",
             "|---|---|---|---|---|---|---|"]
    for key in py:
        a, b = np.array(ml[key]), np.array(py[key])
        p = 1.0 if (a.max() == a.min() == b.max() == b.min()) else mannwhitneyu(a, b).pvalue
        lines.append(f"| {key[0]} | {key[1]} | {np.median(a):.3e} | {np.median(b):.3e} | "
                     f"{a.mean():.3e} | {b.mean():.3e} | {p:.3f} |")
    table = "\n".join(lines)
    print(table)
    (HERE / "comparison.md").write_text(
        f"# MATLAB vs Python MDE\n\nD={D}, N={N}, MaxCycle={MAX_ITER}, {RUNS} independent runs per cell.\n"
        "p > 0.05 means no statistically significant difference between the two implementations.\n\n"
        + table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
