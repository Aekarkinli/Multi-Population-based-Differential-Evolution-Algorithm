# Verification: MATLAB vs Python

| File | Purpose |
|---|---|
| `run_matlab.m` | Runs MDE in MATLAB on 5 functions (25 runs each) and writes `matlab_results.csv` |
| `algo_MDE_eq15.m` | `algo_MDE.m` with Eq. 15 boundary handling (same as v2.0) and no screen output |
| `algo_MDE_clip.m` | The v1.0 `algo_MDE.m` boundary handling (clip), for comparison |
| `bench.m` | Row-wise benchmark functions |
| `run_python.py` | Runs `mde_optimizer` with identical settings and compares the results (Mann–Whitney U) |
| `comparison.md` | Result table |

To reproduce, run `run_matlab` in MATLAB inside this folder, then `python run_python.py`.
