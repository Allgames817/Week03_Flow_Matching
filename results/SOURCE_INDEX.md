# Source Index

Maps every cited number to a local artifact. Checkpoint identity is the SHA-256. Folder names below are on the author machine and are not required to read the tables in this note.

Method paper: [Lipman et al., arXiv:2210.02747](https://arxiv.org/abs/2210.02747).  
Library: [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching), installed **1.0.10**.  
Week 1: [Allgames817/Week01_ACT](https://github.com/Allgames817/Week01_ACT).  
Week 2: [Allgames817/Week02_Diffusion_Policy](https://github.com/Allgames817/Week02_Diffusion_Policy).

The evidence-pack CSV copies match the local run files byte for byte (SHA-256 below). This note cites the local runs, which also carry `run_metadata.json`.

---

## 1. What is **not** in this GitHub repo

| Item | Reason |
|---|---|
| `flow_matching_day3.pt`, `conditional_flow.pt` | local weights; identified by hash |
| `samples.npz` and other `.npz` dumps | large sample tensors |
| Official `flow_matching` source tree | note-only repo; package 1.0.10 was imported at runtime |
| Two-moons notebook / HTML preview | teaching bundle; CUDA metrics are the cited run |
| Full per-step `loss.csv` | reduced to metrics JSON and the loss figures |
| Solver-run `timings.csv` | medians already in `day5_summary.csv` |
| OT-CFM `*.pt`, PushT `hri_pusht.pth` (~335 MB) | local weights |
| OpenPI sources and notes | deferred |

---

## 2. Checkpoints (local)

| Role | SHA-256 | Local role path |
|---|---|---|
| Two-moons MLP; also the API-check and solver weights | `9da3ab2bf6a2aa623da6d2d5ff056b2b0cd4de332df581c1d6d55f648ce6f189` | `Week3_Day3/outputs/day3/flow_matching_day3.pt` |
| Conditional chunk MLP | `c690234b069e1d514d64641ca1342da0e9a191ace350c59c04d9358492f09f8c` | `Week3_Day6/outputs/day6/run_20260927_004031_559009/conditional_flow.pt` |
| Official image PushT weights (rollout only) | `a4e16aebb2bcddc32846109dd7bd61a88d9f31bd0791357e76268958b5bd1096` | `Week3_Day8/data/hri_pusht.pth` |

Do not point the API check or the solver study at `Week3_Day3/reference_cpu_run/`. That directory is a different CPU process.

---

## 3. Copied results in this note

| File | Source | SHA-256 |
|---|---|---|
| [`day3_metrics.json`](day3_metrics.json) | `Week3_Day3/outputs/day3/metrics.json` | copy of the CUDA run |
| [`day3_config.json`](day3_config.json) | same folder, `config.json` | |
| [`day4_checks.json`](day4_checks.json) | numeric fields from `Week3_Day4/outputs/day4/comparison.json` | paths to site-packages removed |
| [`day5_summary.csv`](day5_summary.csv) | `Week3_Day5/.../run_20260926_205038_448401/summary.csv` | `7da47ca8802d19f018d75a89ce39a7bf2809bc3914a2a6616f4dd09b8c75d1b3` |
| [`day5_reference_checks.csv`](day5_reference_checks.csv) | same run | `e63b1e3edc70d0f9795a209c4d050dd29ace036301915910400d790ab4e13085` |
| [`day5_per_seed.csv`](day5_per_seed.csv) | same run | |
| [`day5_identity.json`](day5_identity.json) | fields from that run’s `run_metadata.json` | local checkpoint path removed |
| [`day6_summary.csv`](day6_summary.csv) | `Week3_Day6/.../run_20260927_004031_559009/summary.csv` | `3e3cdf8a73cfde7db96e2f871b073b3f8c5a0ef8b4aa25f078987bdf901d6dc0` |
| [`day6_mode_coverage.csv`](day6_mode_coverage.csv) | same run | `e4686f895be33180a434d019dfe98f940c2b76198905546d12e4ea4ce564798a` |
| [`day6_per_seed_metrics.csv`](day6_per_seed_metrics.csv) | same run | |
| [`day6_identity.json`](day6_identity.json) | fields from that run’s `run_metadata.json` | |
| [`otcfm_aggregate.csv`](otcfm_aggregate.csv) | `Week3_Day8/outputs/otcfm_full/aggregate.csv` | |
| [`otcfm_summary.csv`](otcfm_summary.csv) | same run, per seed | |
| [`otcfm_pairing.json`](otcfm_pairing.json) | same run | |
| [`otcfm_identity.json`](otcfm_identity.json) | config fields; site-packages path removed | |
| [`pusht_overfit.json`](pusht_overfit.json) | `Week3_Day8/outputs/robot_train_smoke_fix2/robot_smoke.json` | |
| [`pusht_overfit_loss.csv`](pusht_overfit_loss.csv) | same run, `loss.csv` | |
| [`pusht_rollout.csv`](pusht_rollout.csv) | `Week3_Day8/outputs/robot_rollout_nfe4/rollout.csv` | |
| [`pusht_identity.json`](pusht_identity.json) | dataset counts, overfit losses, checkpoint SHA-256 | |
| [`pusht_rollouts/`](pusht_rollouts/) | three GIFs from that rollout | |

The solver and conditional-model summary hashes match the evidence pack (`evidence_manifest.json` on the author machine). The pairing and PushT files are copies of the local `Week3_Day8` runs. Smoke directories are not the cited results.

---

## 4. Derived numbers (not stored as their own measurement)

| Claim | How computed |
|---|---|
| Reference-vs-target SWD2 **0.0657432369302124** | mean of three `reference_swd2` rows |
| Target-vs-target SWD2 **0.046907074700664055** | mean of three `target_vs_target_swd2` rows |
| Condition-shift ratio **22.179** | `0.8067576090494791 / 0.036374459664026894` |
| Mode fractions in doc 06 | equal-weight mean of the three seeds in `day6_mode_coverage.csv` |
| Conditional-model SD | sample SD of three seed means, `ddof=1`, as written by the script |

Eval seeds are not training seeds.

---

## 5. Figures

All figures are copies of the local run PNGs, not redraws.

| Figure | Local source |
|---|---|
| `figures/day3_loss_curve.png` | `Week3_Day3/outputs/day3/loss_curve.png` |
| `figures/day3_data_vs_generated.png` | same folder |
| `figures/day3_trajectories.png` | same folder |
| `figures/day4_manual_vs_official.png` | `Week3_Day4/outputs/day4/manual_vs_official.png` |
| `figures/day5_ode_error_vs_nfe.png` | solver run folder |
| `figures/day5_swd_vs_nfe.png` | solver run folder |
| `figures/day5_latency_vs_nfe.png` | solver run folder |
| `figures/day5_ode_reference.png` | solver run `samples/ode_reference.png` |
| `figures/day6_loss_curve.png` | conditional-model run folder |
| `figures/day6_condition_ablation.png` | conditional-model run folder |
| `figures/day6_same_goal_diversity.png` | conditional-model run folder |
| `figures/day6_same_noise_different_goals.png` | conditional-model run folder |
| `figures/day6_mode_histogram.png` | conditional-model run folder |
| `figures/otcfm_swd2_vs_nfe.png` | `Week3_Day8/outputs/otcfm_full/` |
| `figures/otcfm_pairing_random.png` | same run |
| `figures/otcfm_pairing_minibatch_ot.png` | same run |
| `figures/pusht_overfit_loss.png` | `Week3_Day8/outputs/robot_train_smoke_fix2/loss_curve.png` |

---

## 6. Rounding used in README tables

| Displayed | Exact | Where exact lives |
|---|---|---|
| 1.516425 → 1.001853 | full floats in metrics JSON | [`day3_metrics.json`](day3_metrics.json) |
| 1.250594 → 0.995415 | full floats | same |
| Solver table, 6 decimals / 1 decimal ms | CSV floats | [`day5_summary.csv`](day5_summary.csv) |
| 0.036374, 0.806758, 0.036224 | full floats in the CSV | [`day6_summary.csv`](day6_summary.csv) |
| 22.18× | 22.17923… | ratio in Section 4 |
| 47.46% / 49.87% | fractions in doc 06 | mode CSV means |
| Pairing SWD 0.603 / 0.077 / 0.187 / 0.062 / 0.091 / 0.066 | 3-decimal means | [`otcfm_aggregate.csv`](otcfm_aggregate.csv) |
| 0.772, 3.355 | pairing costs | [`otcfm_pairing.json`](otcfm_pairing.json) |
| 1.347 → 0.001208, eval 0.001017 | overfit losses | [`pusht_overfit.json`](pusht_overfit.json), [`pusht_overfit_loss.csv`](pusht_overfit_loss.csv) |
| max reward 0.303, 0.000, 0.185 | 3 decimals | [`pusht_rollout.csv`](pusht_rollout.csv) |

---

## 7. Script hashes recorded by the runs

| Script | SHA-256 |
|---|---|
| `day5_solver_benchmark.py` | `2db2f656383bae02e561126f5a325d7dd11f24d919ac0f6259cc78b01fc6ccec` |
| `day6_conditional_action_flow.py` | `6003cb3d252949b9ffab448411e0e9dba0bded78056832ef54d97e0a7077ea12` |

---

## 8. Local directories (author machine)

| Folder | Role |
|---|---|
| `Week3_Day2/` | formula check script only |
| `Week3_Day3/outputs/day3/` | cited CUDA training |
| `Week3_Day3/reference_cpu_run/` | author-of-the-lesson CPU preview; not the cited checkpoint |
| `Week3_Day4/outputs/day4/` | checkpoint API comparison |
| `Week3_Day4/outputs/api_checks/` | untrained `--checks-only` run |
| `Week3_Day5/outputs/day5/run_20260926_205038_448401/` | cited solver run |
| `Week3_Day6/outputs/day6/run_20260927_004031_559009/` | cited conditional run |
| `flow_matching/` | local clone for reading; runtime was the installed 1.0.10 wheel |
| `Week3_Day7/` | week summary and evidence pack; OpenPI notes not imported |
| `Week3_Day8/outputs/otcfm_full/` | cited CFM vs minibatch OT-CFM run |
| `Week3_Day8/outputs/robot_train_smoke_fix2/` | cited fixed-minibatch overfit |
| `Week3_Day8/outputs/robot_rollout_nfe4/` | cited official-weight rollout |
| `Week3_Day8/data/hri_pusht.pth` | official weights; not copied here |
