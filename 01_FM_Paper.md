# 01 — Flow Matching Paper Concepts

Companion to [README.md](README.md). Focus: ideas from *Flow Matching for Generative Modeling* (Lipman et al., [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) as they map to this week’s two-moons run and goal-conditioned action chunks, and to leftovers from [Week 1 ACT](https://github.com/Allgames817/Week01_ACT) and [Week 2 Diffusion Policy](https://github.com/Allgames817/Week02_Diffusion_Policy).

The installed library is `flow_matching` 1.0.10 from the [Flow Matching codebase](https://github.com/facebookresearch/flow_matching) ([guide paper, arXiv:2412.06264](https://arxiv.org/abs/2412.06264)). Using that API is not a reproduction of the paper’s image or text experiments.

---

## 1. From ACT and diffusion leftovers to a velocity field

Week 1 modeled an action chunk with a CVAE and, at inference, set the latent to the prior mean $z = 0$. Week 2 replaced that head with conditional noise prediction and a DDPM/DDIM sampler. Both already predict a **chunk**, not only $a_t$.

Flow matching is a training framework for a generative model, not a Transformer architecture. This week uses it as:

$$
p(x_1 \mid c) \quad\text{via}\quad \frac{dX_\tau}{d\tau} = v_\theta(X_\tau, \tau, c)
$$

There is no encoder that is dropped at test time, and the regression target is not $\epsilon$.

---

## 2. Probability path and conditional velocity

Fix a coupling of noise $x_0$ and data $x_1$. This week uses an **independent** coupling: $x_0 \sim \mathcal{N}(0, I)$ drawn separately from $x_1$. That is not a solved optimal-transport pairing.

The conditional OT (CondOT) path used everywhere this week is linear in flow time $\tau \in [0, 1]$:

$$
x_\tau = (1-\tau)\,x_0 + \tau\,x_1
$$

Differentiate in $\tau$ with the endpoints held fixed:

$$
u_\tau = \frac{d}{d\tau}x_\tau = x_1 - x_0
$$

The same vector is $(x_1 - x_\tau) / (1-\tau)$ for $\tau < 1$. Both expressions were checked on two hand-built pairs (`day2_formula_check.py`):

| | sample 0 | sample 1 |
|---|---|---|
| $x_0$ | $(-1, 1)$ | $(2, -1)$ |
| $x_1$ | $(3, 5)$ | $(6, 3)$ |
| $\tau$ | $0.25$ | $0.50$ |
| $x_\tau$ | $(0, 2)$ | $(4, 1)$ |
| $x_1 - x_0$ | $(4, 4)$ | $(4, 4)$ |

Official `CondOTScheduler` encodes the same path as $\alpha(\tau) = \tau$, $\sigma(\tau) = 1-\tau$, so

$$
x_\tau = \sigma(\tau)\,x_0 + \alpha(\tau)\,x_1, \qquad
\dot x_\tau = \dot\sigma(\tau)\,x_0 + \dot\alpha(\tau)\,x_1 = x_1 - x_0
$$

`CondOTScheduler` is the path schedule. It is not `torch.optim.lr_scheduler`.

---

## 3. Conditional flow matching loss

The network sees $(x_\tau, \tau, c)$, not the hidden pair $(x_0, x_1)$. The per-sample regression target is still that pair’s velocity:

$$
\mathcal{L}(\theta) = \mathbb{E}\,\big\|v_\theta(x_\tau, \tau, c) - (x_1 - x_0)\big\|^2
$$

Under standard conditions the minimizer is the conditional mean velocity

$$
v^*(x, \tau, c) = \mathbb{E}[x_1 - x_0 \mid x_\tau = x,\ \tau,\ c]
$$

and the conditional-flow-matching objective differs from the marginal flow-matching objective by a $\theta$-independent constant (Lipman et al., Sections 2–4). The average is over velocities that pass through a given intermediate state. It is not “replace every demonstration by one mean action.”

Because many $(x_0, x_1)$ pairs can land on the same $x_\tau$, the best CFM loss on a given batch can stay away from zero. The two-moons validation MSE falling only to about **1.002** is consistent with that fact. It does not by itself measure sliced Wasserstein distance or a robot success rate.

---

## 4. Sampling is an ODE, not another training step

After $\theta$ is frozen:

$$
X_0 \sim \mathcal{N}(0, I), \qquad
X_{\tau+\Delta\tau} = X_\tau + \Delta\tau\, v_\theta(X_\tau, \tau, c)
\quad\text{(Euler)}
$$

| Operation | What is held fixed | What changes |
|---|---|---|
| `loss.backward()` + `optimizer.step()` | the current batch’s $x_0, x_1, \tau$ | parameters $\theta$ |
| Euler / Midpoint / Heun | $\theta$ and this sample’s $c$ | the state $X$ |

Training draws one $\tau$ per update. It does not roll out the ODE inside the loss. Sampling never receives the clean $x_1$; `sample_euler` and `sample_actions` have no clean-endpoint argument. Target points drawn for a figure are for display only.

Given a fixed $X_0$ and $c$, the ODE used this week is deterministic. A new $X_0$ is still a new sample. That is the same distinction Week 2 needed for DDIM with $\eta = 0$: deterministic given the noise draw, stochastic across noise draws.

---

## 5. What this week does not take from the paper

- No image or text Flow Matching run, and no claim that the two-moons MLP matches the official example’s hyperparameters.
- No optimal-transport coupling, no Riemannian path, no discrete flow.
- No proof, from these toy losses, that a flow action head outperforms Week 2’s noise-prediction head on PushT or Transfer Cube. Those tasks were not rerun.
- π₀’s vision-language action flow is a later reading. It is not an experiment in this note.

---

## 6. Primary references

1. Lipman et al., *Flow Matching for Generative Modeling*, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
2. Meta `AffineProbPath`: [API](https://facebookresearch.github.io/flow_matching/generated/flow_matching.path.AffineProbPath.html). Source: `flow_matching/path/affine.py`.
3. Meta `CondOTScheduler`: `flow_matching/path/scheduler/scheduler.py`.
4. Meta `ODESolver`: [API](https://facebookresearch.github.io/flow_matching/generated/flow_matching.solver.ODESolver.html).
