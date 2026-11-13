# Does putting the wall in the graph help?

**The wall-aware graph improves predictions when the wall weakens field dependence and hurts when dependence persists across it.** This follow-up to the [matched-feature ridge experiment](barrier-ridge-study.md) holds the graph basis dimension, diffusion scale, ridge penalty, observations, and walking geometry fixed when comparing the two graph models.

## Controlled comparison

- Wendland ridge uses the same coordinates and 234 Wendland basis features as before.
- Open graph ridge adds 144 heat features computed on the complete four-neighbor grid, with edges restored across the wall.
- Wall graph ridge adds 144 heat features computed after removing cross-wall edges except at the doorway.

For both graph models, the heat basis uses length 0.22 and the same row normalization. The open graph has a permeability of 1 and the wall graph has a permeability of 0. This changes the covariance through the basis and does not alter legal movement. Both models use the same Gaussian ridge prior scale and intercept ratio selected by pooling their development RMSEs; Wendland ridge gets its own development choice. The separate-settings sensitivity check independently selects the ridge prior for each model. All other input columns and the per-map estimator are identical across the two graph variants.

There are 32 fields per development layout in each of the independent tuning and calibration phases, and 128 fresh fields per test layout and regime. Development door rows are 3 and 8; test door rows are 2 and 9 (zero indexed). The test layout family is the same as before, but the test seeds are new. A paired field has identical Gaussian driving noise and sensor noise under both regimes. The true field uses a graph resolvent with hidden permeability 0.02 or 1 and one of two correlation scales; the fitted heat kernels do not contain the exact field generator. The distinct phases and parameter grid are frozen before test field generation. Scalar interval adjustments use the separate calibration fields.

The three mappers share one target-blind measurement route and readings for each map/count. The two caps are 8 readings within 1 distance unit and 16 readings within 2 units. Movement always goes through the door. Each layout has 28 reserved test sites, including eight in the columns next to the wall. The two count conditions can have different routes. This is a fixed-acquisition prediction comparison.

## Results

Mean held-out latent RMSE over 256 test fields in each row:

| True dependence | Readings | Wendland ridge | Open graph ridge | Wall graph ridge |
| --- | ---: | ---: | ---: | ---: |
| Weak across the wall | 8 | 0.8993 | 0.8938 | 0.8865 |
| Weak across the wall | 16 | 0.8470 | 0.8343 | 0.8207 |
| Persists across the wall | 8 | 0.8816 | 0.8699 | 0.8741 |
| Persists across the wall | 16 | 0.8099 | 0.7861 | 0.7941 |

With 16 readings and weak cross-wall dependence, adding the open graph already lowers mean RMSE by 0.0127. Cutting the wall edges gives a further 0.0136 reduction (1.63% relative to open graph; paired 95% interval for wall minus open: [−0.0178, −0.0096]). At the sites next to the wall, RMSE is 0.8506 → 0.8221, a difference of −0.0284 [−0.0376, −0.0194].

When dependence persists across the wall, cutting its graph edges instead raises RMSE by 0.0080 [0.0048, 0.0113] and near-wall RMSE by 0.0236 [0.0159, 0.0314]. The paired difference of wall effects between regimes at 16 readings is −0.0217 [−0.0243, −0.0191]. Both test door positions and the 8-reading cap show the same direction of reversal. The conclusion is unchanged with separate ridge settings: at 16 readings wall-minus-open RMSE is −0.0137 in the barrier regime and +0.0067 in the transparent regime. Intervals resample paired fields within each fixed layout (8,000 draws) and are unadjusted. They do not represent a population of floor plans.

The effect is specific to the known wall matching the simulated dependence structure. The contrast with open graph isolates the modeled graph edges; it does not show that room geometry is always an adequate proxy for a sensor field. The open-vs-Wendland comparison also changes the feature count and fitted prior and is a comparison of full estimators.

With 16 barrier readings, open and wall graph calibrated coverage is 94.6% and 94.9%, with mean widths 3.28 and 3.23. Their raw coverage is 77.8% and 79.4%, so the near-nominal result requires calibration. The wall graph's calibrated interval score improves from 4.18 to 3.98 (lower is better). All models spend the same average 0.693 and 1.636 distance units at the two caps. Fits cost roughly 0.02–0.06 ms per map in batched CPU throughput, excluding layout/basis construction, route selection, and sensor costs. These results do not validate joint posterior covariance for route selection.

## Reproduce

The [closed-loop follow-up](wall-closed-loop-study.md) gives the two graph mappers their own routes under matched budgets and replays those routes with both mappers to separate prediction from acquisition.

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.wall_topology_study --output outputs/wall-topology-study
```

Use a fresh output directory. Artifacts include protocol and frozen settings, separate observation and evaluator archives, per-field metrics for both tuning strategies, posterior predictions, paired comparisons, and `wall-effect.png`. Verification checks all 1,024 test acquisition sequences, equal graph settings in the primary arm, and independent posterior replay and scoring of all 6,144 mapper records.
