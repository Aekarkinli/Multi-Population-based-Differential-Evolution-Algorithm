# MDE: Multi-population based Differential Evolution

[![View on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/119988)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00521--022--07969--w-blue)](https://doi.org/10.1007/s00521-022-07969-w)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm/actions/workflows/tests.yml/badge.svg)](https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm/actions/workflows/tests.yml)
![MATLAB](https://img.shields.io/badge/MATLAB-R2022a%2B-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**MDE** is a population-based, non-recursive, real-valued global optimizer for single-objective
problems. It was introduced in:

> Karkinli, A.E. (2023). Detection of object boundary from point cloud by using multi-population based
> differential evolution algorithm. *Neural Computing and Applications*, 35, 5193–5206.
> https://doi.org/10.1007/s00521-022-07969-w

This repository holds the official **MATLAB** implementation (also on
[MATLAB File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/119988)), the official **Python** implementation,
and links to MDE in popular Python optimization libraries.

## Why MDE?

- **Parameter-free in practice.** No scale factor *F*, no crossover rate *CR*, no strategy to choose.
  You set only the population size and the budget.
- **Partial-elitist mutation.** Each trial vector mixes an elitist component (the global best), a
  random component (other sub-population members) and a tiny noise term that keeps diversity alive.
- **Scale-invariant step sizes.** A power-law scale value produces both very small and very large
  steps, which helps it escape local optima.
- **Multi-population structure.** In every iteration, a random sub-population of *N* vectors leaves the
  main population of *3N*, evolves, and returns its improvements.
- **Non-recursive.** All *N* trial vectors of an iteration are independent, so they can be evaluated
  in parallel or vectorised.
- **Global convergence proof.** The paper includes one.
- **Benchmarked.** On CEC2013 (28 problems, D = 20), MDE was statistically better than ABC, CS, SOS and
  GWO (Wilcoxon signed-rank test, p = 0.05).

## Algorithm in brief

```
A  <- 3N random vectors in [low, up];  Best <- argmin F(A)
repeat MaxCycle times:
    B     <- N vectors drawn at random from A                       (sub-population)
    for each b_i in B:
        scale   = |a - rand^b| * randn^d,  a~U{0,1}, b~U{1..10}, d~U{1..5}
        r1, r2  distinct random indices, both != i
        dx      = b_r1 or Best      (per variable, prob. 0.5)
        dy      = b_i  or b_r2      (per variable, prob. 0.5)
        temp_i  = b_r2 + scale * (dx - dy)
    map   = |alpha - beta.^delta| < 0.5                             (parameter-free crossover)
    T     = B + map .* (temp + noise - B);  out-of-bounds -> U(low, up)
    B_i   <- T_i if F(T_i) < F(b_i);   A <- B;   Best <- argmin F(A)
    noise <- B .* 10^phi .* (rand - 0.5),  phi~U{-12..-9}
```

Each iteration costs *N* function evaluations (plus *3N* for initialisation).

## MATLAB

```matlab
%   out = algo_MDE(objfun, mydata, N, D, low, up, MaxCycle, seed)
out = algo_MDE('fitCircle', mydata, 30, 3, -100, 100, 1000, 100);
out.bestsol   % best solution
out.bestval   % best objective value
```

`objfun` receives an `N x D` matrix (one candidate per row) plus `mydata`, and returns an `N x 1`
vector. Run [`example_circle_fit.m`](example_circle_fit.m) for the full circle-fitting example from the
paper. [`fitCircle.m`](fitCircle.m) is its objective function.

## Python

```bash
pip install git+https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm.git
```

```python
import numpy as np
from mde_optimizer import minimize

def rastrigin(x):
    return 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

res = minimize(rastrigin, lb=-5.12, ub=5.12, dim=10, pop_size=30, max_iter=1000, seed=1)
print(res.x, res.fun, res.nfev)
```

- `pop_size` is *N*; the full population holds *3N* vectors.
- With `vectorized=True`, `func` receives an `(n, D)` array and returns `n` values.
- `max_nfev` caps the number of function evaluations.
- The result also carries `history`, the best value after each iteration.

See [`examples/`](examples) for the circle-fitting example and a benchmark script.

## MDE in Python optimization libraries

| Library | How to use MDE | Status |
|---|---|---|
| [mealpy](https://github.com/thieu1995/mealpy) | `from mealpy import MDE` → `MDE.OriginalMDE(epoch=1000, pop_size=90)` | Submitted: [thieu1995/mealpy#286](https://github.com/thieu1995/mealpy/pull/286) |
| [NiaPy](https://github.com/NiaOrg/NiaPy) | `from niapy.algorithms.basic import MultiPopulationDifferentialEvolution` | Submitted: [NiaOrg/NiaPy#770](https://github.com/NiaOrg/NiaPy/pull/770) |
| [OptunaHub](https://hub.optuna.org) | `optunahub.load_module("samplers/mde").MDESampler(population_size=30)` | Submitted: [optuna/optunahub-registry#457](https://github.com/optuna/optunahub-registry/pull/457) |
| [EvoloPy](https://github.com/7ossam81/EvoloPy) | `run(["MDE"], ["F1"], ...)` | Submitted: [7ossam81/EvoloPy#85](https://github.com/7ossam81/EvoloPy/pull/85) |
| [pyMetaheuristic](https://github.com/Valdecy/pyMetaheuristic) | `pymetaheuristic.optimize(algorithm="mde", ...)` | Submitted: [Valdecy/pyMetaheuristic#14](https://github.com/Valdecy/pyMetaheuristic/pull/14) |

Until a pull request is merged, you can install that library from the `add-mde` branch of its fork
(for example `pip install git+https://github.com/Aekarkinli/mealpy.git@add-mde`).

In mealpy, NiaPy, EvoloPy and pyMetaheuristic, the library's population size means the full population
(*3N*). So `pop_size=90` is the same as `N = 30` here.

<details>
<summary>Usage examples for each library</summary>

**mealpy**

```python
import numpy as np
from mealpy import FloatVar, MDE

problem = {"bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30),
           "minmax": "min", "obj_func": lambda x: np.sum(x ** 2)}
g_best = MDE.OriginalMDE(epoch=1000, pop_size=90).solve(problem, seed=42)
print(g_best.solution, g_best.target.fitness)
```

**NiaPy**

```python
from niapy.algorithms.basic import MultiPopulationDifferentialEvolution
from niapy.problems import Rastrigin
from niapy.task import Task

task = Task(problem=Rastrigin(dimension=10), max_evals=30090)
best_x, best_f = MultiPopulationDifferentialEvolution(population_size=90, seed=1).run(task)
```

**OptunaHub**

```python
import optuna, optunahub

sampler = optunahub.load_module(package="samplers/mde").MDESampler(population_size=30, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=3000)
```

**EvoloPy**

```python
from EvoloPy.optimizer import run
run(["DE", "MDE"], ["F1", "F9"], NumOfRuns=10, params={"PopulationSize": 30, "Iterations": 500})
```

**pyMetaheuristic**

```python
import pymetaheuristic
result = pymetaheuristic.optimize(algorithm="mde", target_function=f,
                                  min_values=[-5.12] * 10, max_values=[5.12] * 10,
                                  population_size=90, max_evaluations=30_090, seed=42)
```

</details>

## Verification

The Python implementation was checked against the MATLAB code (R2023a). The setup was D = 10, N = 30,
1000 iterations and 25 independent runs per cell. No difference was statistically significant
(Mann–Whitney U, p > 0.05 in all 9 cells):

| Function | MATLAB median | Python median | p |
|---|---|---|---|
| Sphere | 3.10e-19 | 2.29e-19 | 0.79 |
| Rastrigin | 0 | 0 | 0.92 |
| Rosenbrock | 2.99 | 5.23 | 0.32 |
| Ackley | 1.52e-10 | 1.54e-10 | 0.42 |
| Shifted sphere, box [1, 5] | 2.55e-20 | 2.75e-20 | 0.80 |

The full table and the scripts to reproduce it are in [`verification/`](verification).

The library versions were checked against this Python implementation. NiaPy and OptunaHub give
bit-identical results for the same seed. mealpy, EvoloPy and pyMetaheuristic show no statistically
significant difference.

## Changelog

- **v2.0 (2026)**
  - Added the official Python implementation (`mde_optimizer`) with tests and continuous integration.
  - Added the library integrations listed above.
  - `borderControl` in `algo_MDE.m` now follows Eq. 15 of the paper: an out-of-bounds variable is
    re-sampled uniformly in `[low, up]`. The previous version could return values outside the box
    when `low` and `up` had the same sign. For boxes such as `[-100, 100]`, the old version clipped to
    the bound instead. Both rules give statistically similar results (see `verification/`).
  - License changed from GPL-3.0 to MIT.
- **v1.0 (2022)**: MATLAB implementation accompanying the paper.

## Citation

If you use MDE in your research, please cite:

```bibtex
@article{karkinli2023mde,
  author  = {Karkinli, Ahmet Emin},
  title   = {Detection of object boundary from point cloud by using multi-population based differential evolution algorithm},
  journal = {Neural Computing and Applications},
  volume  = {35},
  pages   = {5193--5206},
  year    = {2023},
  doi     = {10.1007/s00521-022-07969-w}
}
```

GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) © 2022–2026 Ahmet Emin Karkinli. Department of Geomatics Engineering, Niğde Ömer Halisdemir University.
