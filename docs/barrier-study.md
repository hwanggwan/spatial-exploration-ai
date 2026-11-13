# Geometry-aware spatial exploration

**With a wall that weakens field dependence, a geometry-aware GP improves final mapping error at the larger matched resource cap. The benefit is conditional: when the field remains connected across the wall, the geometry model loses at the smaller cap.**

This is a synthetic floor-plan experiment, not a reproduction of a barrier-SPDE paper or a real air-quality or robotics validation. The question is whether known geometry helps prediction and measurement selection under a budget. The [graph DeepKriging follow-up](barrier-deepkriging-study.md) uses fresh fields and new door positions to compare three mappers on identical acquisitions.

## Design

- **Domain and movement.** A 12-by-12 unit-square grid has a vertical wall and one door, with two door placements (rows 3 and 8, zero-indexed). Both models know the same layout for movement. Costs are shortest four-neighbor walking distances; the wall can only be crossed at the door. Passing through a site does not collect its value. The starting reading counts; no return to the start is required.
- **Hidden field.** The simulator generates zero-mean, unit-marginal-variance Gaussian fields using a normalized graph covariance proportional to `(κ²I + L)^−2`. In the barrier regime, non-door cross-wall field edges have weight 0.02. In the transparent regime, their weight is 1, even though robot movement is still blocked. Two hidden κ values vary spatial scale. The two regimes share latent standard-normal draws and sensor noise (SD 0.12) within each field pair. There are 48 draws per layout and 96 fields in each regime, balanced over four starts and the two κ values.
- **Mappers.** The coordinate GP uses an RBF kernel. The geometry GP searches a blend of that same RBF with a unit-diagonal graph heat kernel from the door-connected layout, using weights 0, 0.5, and 1. Both use the same five length candidates (0.09, 0.14, 0.22, 0.34, 0.50), known zero mean, unit variance, and sensor noise. In both families, from four readings onward, a discrete marginal likelihood fit uses only the acquired readings. Before four, length is 0.22 and graph weight is 0 or 0.5, respectively. The geometry family can fall back to RBF; it has additional flexibility. Its heat-kernel family does not include the simulator's exact covariance, permeability, or κ.
- **Same observations.** A fixed stationary-prior greedy route is computed from geometry and covariance without reading field values. Both models receive exactly its sites and sensor readings, and fit once at the endpoint.
- **Closed loop.** Each GP gets the same start and caps, refits at each reading, and uses the same one-step rule: predicted whole-grid latent variance reduction divided by legal travel distance. After committing to a site, the planner receives that reading. The tighter cap is 8 readings and 1 distance unit; the larger cap is 16 and 2. A budget is an upper limit, so the runs can spend different amounts.
- **Evaluation.** Twenty-eight reserved sites per layout, including eight near the wall, are never sensed by either planner. The primary outcome is mean per-field latent RMSE at these sites. Pairwise 95% percentile bootstrap intervals (4,000 draws, resampling fields within each layout) describe variation across simulated fields on these layouts. They are unadjusted across comparisons and are not intervals for a population of floor plans. Coverage and width use nominal 95% plug-in GP intervals for latent values, with the covariance choice treated as fixed.

## Prediction and exploration results

Each entry is stationary GP → geometry-aware GP. Negative paired differences favor geometry.

| Field dependence | Reading cap | Same observations RMSE | Closed-loop RMSE | Closed-loop paired difference (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Wall weakens dependence | 8 | 0.9082 → 0.9051 | 0.9137 → 0.9006 | −0.0131 [−0.0365, +0.0094] |
| Wall weakens dependence | 16 | 0.8282 → 0.8054 | 0.8528 → 0.7965 | −0.0564 [−0.0847, −0.0282] |
| Dependence crosses wall | 8 | 0.8717 → 0.8862 | 0.8672 → 0.8972 | +0.0300 [+0.0091, +0.0520] |
| Dependence crosses wall | 16 | 0.7736 → 0.7785 | 0.8009 → 0.8010 | +0.0001 [−0.0244, +0.0249] |

At the larger barrier cap, error fell by **2.75% with identical observations** (paired difference −0.0228 [−0.0368, −0.0099]) and **6.61% in the closed loop**. The closed-loop error fell on 70 of 96 fields and on average in both door layouts. Near-wall RMSE fell from 0.8786 to 0.8211, latent coverage rose from 88.4% to 91.4%, and mean interval widths were nearly equal (2.976 vs. 2.985). Coverage remained below the nominal 95%. Mean readings were 15.97 vs. 15.92, distance 1.55 vs. 1.61, and rates of sensing on both sides were 32.3% vs. 41.7%. The improvement is at equal caps, with slightly higher realized movement for the geometry arm. On this small grid, mean accumulated fitting plus planning was about 4–5 ms per run at that cap, excluding construction of shared layout/kernel tables, sensing, and physical movement.

The 8-reading barrier result is inconclusive. Under the same movement constraints but transparent field dependence, the geometry model's 8-reading error was **3.46% higher** in closed loop. The movement wall is not sufficient evidence that the field also has a barrier. The two resource settings change both caps, so differences between them cannot be attributed to readings alone.

## Which part changed the result?

For the larger barrier cap, replay both planners' acquired readings with both mappers. This uses existing readings, with no additional measurement or route optimization.

| Route whose readings are supplied | Stationary mapper RMSE | Geometry mapper RMSE |
| --- | ---: | ---: |
| Stationary-driven route | 0.8528 | 0.8322 |
| Geometry-driven route | 0.8239 | 0.7965 |

Holding the stationary-driven route fixed, geometry lowers RMSE by 0.0207 (95% paired interval [−0.0309, −0.0113]); holding the geometry-driven route fixed, it lowers RMSE by 0.0274 [−0.0374, −0.0174]. Holding the geometry mapper fixed, swapping in the geometry-driven route lowers RMSE by 0.0357 [−0.0630, −0.0084]; with the stationary mapper fixed, the route contrast is −0.0289 [−0.0582, +0.0003]. Thus both mapper and acquired-data differences contribute on these fields. The route contrast includes realized distance differences and does not establish route optimality or an equal-distance advantage.

The route figure illustrates the first barrier case where the geometry planner senses in both rooms and the stationary planner does not. It is chosen by route behavior, not accuracy: in that particular case, the geometry-driven model actually has higher held-out RMSE (0.887 vs. 0.703). Aggregate results are in the paired interval figure.

## Reproduce and verify

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m spatial_exploration.barrier_study --output outputs/barrier-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Use a fresh output directory. `protocol.json` records the settings before field generation. `closed-observations.csv` and `shared-observations.csv` contain measurements and cumulative legal travel distance. `evaluation-only.npz` contains the hidden fields and reserved masks. `metrics.csv`, `summary.json`, and `crossover.json` record endpoint scores and paired comparisons. `routes.png` shows legal walking paths and sampling sites; `paired-rmse.png` shows paired changes with intervals.

The completed run verified all 1,152 archived routes, 12,645 measurement legs, and 245 door crossings: all respect both caps and omit reserved sites. Every common site reading matches exactly within its field. Replaying all 2,304 score records from archived acquisitions yielded zero score discrepancy; a second run matched all non-timing fields. Unit checks cover legal door crossings, covariance validity, posterior agreement with the existing sequential GP, and the closed loop's read-only-at-acquisition interface.

Two fixed layouts and a synthetic static generator are enough for this conditional comparison, not generalization. Next, validate transfer on new layouts and weaker or mismatched barriers, then compare a richer model against the same geometry GP and sensor/travel constraints.

## References

- Kondor, R. I., and Lafferty, J. (2002). [Diffusion Kernels on Graphs and Other Discrete Input Spaces](https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf). ICML. Sections 2.3 and 3 introduce the graph diffusion kernel and its interpretation.
- Borovitskiy, V., et al. (2021). [Matérn Gaussian Processes on Graphs](https://proceedings.mlr.press/v130/borovitskiy21a.html). AISTATS. Gives the graph diffusion kernel as the covariance of a graph squared exponential GP and develops graph Matérn covariances.

The unit-diagonal normalization and blend with a coordinate RBF in this study are our modeling choices.
