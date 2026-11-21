# MATLAB vs Python MDE

D=10, N=30, MaxCycle=1000, 25 independent runs per cell.
p > 0.05 means no statistically significant difference between the two implementations.

| variant | function | MATLAB median | Python median | MATLAB mean | Python mean | Mann-Whitney p |
|---|---|---|---|---|---|---|
| eq15 | sphere | 3.103e-19 | 2.291e-19 | 4.063e-19 | 3.576e-19 | 0.786 |
| eq15 | rastrigin | 0.000e+00 | 0.000e+00 | 4.776e-01 | 4.378e-01 | 0.919 |
| eq15 | rosenbrock | 2.985e+00 | 5.232e+00 | 3.487e+00 | 4.119e+00 | 0.322 |
| eq15 | ackley | 1.524e-10 | 1.541e-10 | 1.628e-10 | 2.160e-10 | 0.415 |
| eq15 | shifted_sphere | 2.554e-20 | 2.747e-20 | 2.935e-20 | 2.933e-20 | 0.801 |
| clip | sphere | 2.870e-19 | 2.703e-19 | 5.068e-19 | 7.824e-19 | 0.641 |
| clip | rastrigin | 9.950e-01 | 0.000e+00 | 5.970e-01 | 6.766e-01 | 0.771 |
| clip | rosenbrock | 4.871e+00 | 5.458e+00 | 4.294e+00 | 4.734e+00 | 0.497 |
| clip | ackley | 2.579e-10 | 1.837e-10 | 2.916e-10 | 2.640e-10 | 0.269 |
