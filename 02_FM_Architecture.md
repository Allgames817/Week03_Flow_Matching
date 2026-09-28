# 02 — Flow Matching Architecture (this week)

Maps the Week 3 mental model to the handwritten MLPs and to `flow_matching` 1.0.10. Two networks appear. They are not checkpoints of each other: two-moons is a 2-D unconditional velocity field; the action-chunk model is a goal-conditioned field trained from scratch.

---

## 1. Unconditional two-moons

```
Training
────────
  x1 = two_moons(n, noise=0.08)     # [B, 2]
  x0 ~ N(0, I)                       # [B, 2], independent
  τ  ~ U(0, 1)                       # [B, 1] in the handwritten loop
  xt = (1-τ) x0 + τ x1
  target = x1 - x0
  pred = VelocityMLP(xt, τ)
  loss = mean((pred - target)^2)
  Adam step on θ

Inference
─────────
  X ~ N(0, I)                        # [N, 2]
  for k in 0 .. K-1:
      τ = k / K
      X ← X + (1/K) * VelocityMLP(X, τ)
  # K = 100 for the reported samples; the solver study varies K
```

`VelocityMLP` maps `concat(xt, τ)` (3 inputs) through three SiLU hidden layers of width 128, then a linear map to $\mathbb{R}^{2}$. The clean endpoint is not an input.

`data_noise = 0.08` is the thickness of the moon point cloud. It is not a `sigma_min` on the probability path. The path variance is entirely the interpolated Gaussian endpoint.

---

## 2. Official affine path

```
path = AffineProbPath(scheduler=CondOTScheduler())
sample = path.sample(x_0=x0, x_1=x1, t=t)   # t shape [B], not [B, 1]
# sample.x_t  = (1-t) x0 + t x1
# sample.dx_t = x1 - x0          # label, not a network prediction
pred = model(sample.x_t, sample.t[:, None])
loss = MSE(pred, sample.dx_t)
```

`Day3Adapter` only renames arguments so `ODESolver` can call `velocity(x=..., t=...)` while the handwritten module still expects `forward(xt, t)` with a `[B, 1]` time. The adapter adds no new weights and is not trained.

For the checkpoint comparison, both Eulers see the same 101-node grid `linspace(0, 1, 101)` (100 intervals). `step_size=None` on that grid means “use the grid spacing.” It does not turn Euler into an adaptive solver. A separate check against the handwritten constant `dt = 1/K` is allowed a float32 rounding gap; the measured endpoint max abs diff is **1.5497208e-6**.

---

## 3. Solvers and NFE

NFE counts **batch** forwards of the velocity net. It does not multiply by the number of points, and it is not a training step.

| Method | Evaluations per interval | Intervals at NFE $= 24$ |
|---|---:|---:|
| `euler` | 1 | 24 |
| `midpoint` | 2 | 12 |
| `heun3` | 3 | 8 |

The reference trajectory is the same weights cast to float64 and integrated with Dopri5 at `rtol = atol = 1e-9`. Passing time grid `[0, 1]` to Dopri5 asks for outputs at the endpoints; the solver still chooses its own internal steps. Those internal steps are excluded from the latency ranking.

---

## 4. Goal-conditioned action chunk

Demonstrations are synthetic. For goal $g$ and progress $s = j/H$:

$$
p(s) = s\,g + m\,b\,\sin(\pi s)\,n(g) + w\,\sin(2\pi s)
$$

$$
n(g) = (-g_y, g_x) / \|g\|,\quad
m \in \{-1,+1\}\ \text{with equal probability},\quad
b \in [0.35, 0.50],\quad
w \sim \mathcal{N}(0, 0.03^2 I)
$$

Endpoints of the demonstration are set to $(0,0)$ and $g$. Relative actions are $a_j = p_{j+1} - p_j$. The model never receives $m$, $b$, or $w$.

```
Training
────────
  c = goal                                          # [B, 2]
  raw actions                                       # [B, 16, 2]
  x1 = 16 * raw actions                             # fixed scale, not a learned normalizer
  x0 ~ N(0, I)                                      # [B, 16, 2]
  τ ~ U(0, 1)                                       # [B]
  xt, target = AffineProbPath.sample(x0, x1, τ)    # target = x1 - x0
  pred = ConditionalVelocityMLP(xt, τ, c)          # [B, 16, 2]
  loss = MSE(pred, target)
  Adam + CosineAnnealingLR (1e-3 → 1e-4 over 6000 steps)

Inference
─────────
  X ~ N(0, I) shaped [B, 16, 2]
  ODESolver midpoint, K = 32 intervals, condition=c on every evaluation
  NFE = 64
  positions = cumsum(X / 16) with a leading (0, 0)     # [B, 17, 2]
```

`ConditionalVelocityMLP` flattens the chunk, then concatenates time and goal: $32 + 1 + 2 = 35$ inputs, 32 outputs, three SiLU layers of width 128. One forward predicts the whole chunk. The goal is not noised and does not change with $\tau$.

Two different “times” show up in the same tensor:

| Name | What it indexes |
|---|---|
| Flow time $\tau$ | where the chunk sits between noise and data |
| Action index $j = 0..15$ | which relative displacement inside the chunk |
| ODE interval count $K = 32$ | how the sampler walks in $\tau$ |
| NFE $= 64$ | Midpoint evaluates the net twice per interval |

`decode_positions` has no goal argument. It does not snap the endpoint to $g$. Summing displacements is kinematics of this toy, not the flow ODE.

---

## 5. What is shared with ACT and Diffusion Policy

| Piece | Week 1 ACT | Week 2 DP | Week 3 FM |
|---|---|---|---|
| Chunk | $k$ joint targets | horizon 16, execute 8 | horizon 16, open-loop sum |
| Condition | image + qpos | flattened keypoints | goal only |
| Generator | CVAE decoder | CondUnet1D noise predictor | velocity MLP |
| Test-time iteration | optional temporal aggregation | $K$ denoising steps | $K$ ODE steps |

The action-chunk horizon 16 matches Week 2’s prediction window only as a number. The observation, action semantics, dataset, and simulator are different, so the horizons are not a matched ablation.

A later image-conditioned PushT interface uses a different condition: a 512-D image feature plus 2-D agent position, and it executes 8 of the 16 predicted targets. That wiring, and a separate minibatch-OT pairing study, are in [07_Coupling_and_Observation.md](07_Coupling_and_Observation.md). The table above describes the synthetic goal-chunk model.
