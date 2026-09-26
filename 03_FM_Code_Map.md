# 03 — Code Map

File and function map for the Week 3 exercises. Teaching scripts stay in the local exercise folders. The official package is imported from the `flow` conda environment (`flow_matching` 1.0.10); this note does not vendor [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).

A local clone of that repository was used for reading (`AffineProbPath`, `CondOTScheduler`, `ODESolver`). The API check, solver study, and conditional model **ran** the installed wheel, not an editable install of the clone. Byte identity between the wheel and the clone was not re-checked for this note.

---

## 1. Formula check

```
day2_formula_check.py
  xt = (1-t) x0 + t x1
  velocity_target = x1 - x0
  assert velocity_target == (x1 - xt) / (1-t)
  tiny MLP, MSE, loss.backward()
```

No dataset, no optimizer step, no saved checkpoint.

---

## 2. Handwritten training and Euler sampling

Script: `Week3_Day3/day3_flow_matching.py`.

```
main
  → run_experiment
       ├── sample_two_moons / make_training_batch
       ├── VelocityMLP.forward(xt, t)
       ├── train_model          # Adam, 5000 steps
       ├── sample_euler         # K constant steps, no x1
       └── save_outputs         # loss.csv, metrics.json, plots, .pt
```

| Function | Role |
|---|---|
| `sample_two_moons` | two semicircles plus `data_noise * N(0, I)` |
| `make_training_batch` | independent $x_0$, uniform $\tau$, linear $x_t$, label $x_1-x_0$ |
| `VelocityMLP` | 3-wide SiLU MLP, input `[xt, t]` |
| `train_model` | `optimizer.step` updates $\theta$; checks that parameters change |
| `evaluate_cfm` | MSE on one frozen validation batch (4096) |
| `sample_euler` | integrates $X$; stack of states is `[K+1, N, 2]` |

The notebook `flow_matching_from_scratch.ipynb` is the same exercise with a saved CPU preview. The cited two-moons metrics are the local CUDA script run in `outputs/day3/`, not `reference_cpu_run/`.

---

## 3. Official API on the two-moons network

Script: `Week3_Day4/day4_official_api.py`.

```
check_path
  AffineProbPath(CondOTScheduler()).sample
  vs handwritten xt and x1-x0

check_training
  one Adam step on disposable copies
  original state_dict untouched

check_sampling
  manual_euler(time grid)
  vs ODESolver.sample(method="euler", time_grid=grid)
  vs day3_constant_dt_euler
```

| Official symbol | File in the library |
|---|---|
| `CondOTScheduler.__call__` | `flow_matching/path/scheduler/scheduler.py` |
| `AffineProbPath.sample` | `flow_matching/path/affine.py` |
| `ModelWrapper` | `flow_matching/utils/model_wrapper.py` |
| `ODESolver.sample` | `flow_matching/solver/ode_solver.py` |

`--checks-only` uses an untrained network and does not load the two-moons checkpoint. The numbers in [`results/day4_checks.json`](results/day4_checks.json) are the checkpoint-reuse run.

Reading order used this week: scheduler → `AffineProbPath.sample` → model wrapper → `ODESolver.sample`. `compute_likelihood` was not part of the exercise. `examples/2d_flow_matching.ipynb` was read for imports and call order, not executed end to end.

---

## 4. Equal-NFE solver benchmark

Script: `Week3_Day5/day5_solver_benchmark.py`. Depends on the two-moons `.pt` and on `ODESolver`. It does not import the training or API-check Python files.

```
load_model(checkpoint)          # same VelocityMLP layout
CountingVelocity                # nfe += 1 per batch forward
fixed_sample                    # explicit K+1 grid, step_size=None
adaptive_reference              # float64 Dopri5, two tolerances
endpoint_rmse                   # paired with the reference, same x0
swd2                            # sort after projection; index i is not a pair
```

`--self-test` checks an ODE with a known solution and the NFE counter. It does not write the experiment directory. The benchmark refuses to swap in a handwritten solver if the official import fails.

---

## 5. Conditional chunk

Script: `Week3_Day6/day6_conditional_action_flow.py`. **Does not** load the two-moons checkpoint. Input and output ranks differ.

```
make_demonstrations(goals)      # scaled x1 and positions; bend sign stays off the graph
PathBackend.sample              # official AffineProbPath, or an explicit native fallback
ConditionalVelocityMLP.forward(x, t, condition)
train_model                     # Adam + CosineAnnealingLR
sample_actions                  # ODESolver(..., condition=condition); no x1
decode_positions                # divide by H, cumsum; no goal argument
evaluate                        # correct goal vs cyclic shift; mode histogram at (1, 0)
```

Default `--backend official` raises if `flow_matching` cannot be imported. `--backend native` is an explicit offline path with the same $x_t$ and $x_1-x_0$ formulas. The run cited here is `backend: official`.

`sample_actions` passes `condition` through `ODESolver` model extras, so every velocity evaluation sees the same goal. `make_demonstrations` is not called from `sample_actions`.

---

## 6. What is intentionally absent

| Item | Reason |
|---|---|
| Vendored `flow_matching/` tree | note-only repo, same pattern as Week 2 vs Diffusion Policy |
| `*.pt` checkpoints | local weights; identified by SHA-256 |
| `*.npz` sample dumps | large; plots and CSV summaries are the note |
| Two-moons notebook and HTML preview | teaching bundle; metrics cited from the CUDA script run |
| OpenPI source reading | deferred; not part of this note |
