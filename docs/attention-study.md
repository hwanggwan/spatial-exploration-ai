# Query-dependent attention for spatial context conditioning

## Change and hypothesis

The [pooled CNP](cnp-study.md) learned across maps but responded weakly to the observations on a new map. This experiment replaces its shared mean-pooled representation with a query-specific weighted sum of the encoded observed pairs. The question is whether direct access to relevant observations improves reconstruction and budget-constrained exploration.

The idea is motivated by [Kim et al., Attentive Neural Processes](https://arxiv.org/abs/1901.05761), with the authors' [implementation](https://github.com/google-deepmind/neural-processes/blob/master/attentive_neural_process.ipynb) as a reference. This implementation is **deterministic attention-CNP**, not a reproduction of the full latent ANP. It has no latent variational path, KL objective, context self-attention, or joint posterior over field realizations.

## Architecture and capacity control

Encode each observed `(x,y,value)` as a 64-dimensional value vector. A separate shared coordinate MLP maps context and query positions to keys/queries: `2 → 22 → 64`, with ReLU between layers. Normalize coordinates to [-1,1]. For query s and acquired sites i:

\[
a_i(s)=\operatorname{softmax}_i\left(\frac{g(s)^Tg(x_i)}{\sqrt{64}}\right),\qquad
r(s)=\sum_i a_i(s)h(x_i,y_i).
\]

The decoder receives r(s), query coordinates, and the existing normalized log context-count channel. It outputs mean and positive latent standard deviation. Keep the known-noise Gaussian target likelihood and 0.02 latent std floor. The attention weights depend on positions; context values enter through the encoded value vectors. There is no fixed distance kernel or imposed nearest-neighbor rule.

Reduce encoder/decoder hidden widths from 64 to 60 to allocate parameters to the positional network. The pooled baseline has **17,218 parameters**, versus **17,204** for attention (0.081% fewer). This controls total parameter count closely, but changes widths slightly and does not equalize FLOPs or every inductive bias. Attention weights are normalized aggregation coefficients, not causal feature importance.

If the positional network is zeroed, the weights become uniform and the forward pass equals a pooled CNP of the same width. Tests verify that limiting case, context-order invariance, query-specific weights, query independence, sensitivity to acquired values, and nonzero gradients through the positional network.

## Data, selection, and comparison

Reuse exactly the existing 4,096 noisy offline training maps (921,600 readings), 32 validation maps, validation routes, learned covariance GP, and pooled CNP checkpoints. Archive source hashes and copy the required artifacts without changing them. Previous benchmark targets are never loaded for attention training or selection. The two retained attention training seeds use the same initialization seed rule, mini-batch RNG rule, sampler, noisy labels, optimizer, clipping, and maximum 6,000 updates as the original CNP. Geometry sampling does not depend on network width, so the per-seed context/target batches match.

Keep three checkpoint candidates per architecture: 1,000 / 3,000 / 6,000 updates. Select one attention checkpoint count by noisy held-out likelihood averaged across the same reference-route prefixes and both seeds. Retain both seeds rather than choosing the winner. Pooled CNP retains its independently selected 3,000-update checkpoint.

| Updates | Pooled CNP validation NLL | Attention-CNP validation NLL |
| ---: | ---: | ---: |
| 1,000 | 1.35858 | 1.33632 |
| 3,000 | **1.34646** | 1.31686 |
| 6,000 | 1.34764 | **1.31516** |

Attention selects **6,000 updates**. Selection at different checkpoints means the primary deployed comparison is between selected pipelines, not an isolated fixed-duration intervention. As a secondary diagnostic, compare both architectures at 6,000 updates on the same fitted-GP observations. In the recorded run, this diagnostic was specified after checkpoint selection but **before examining benchmark outcomes**, and is separately recorded in `fixed_step_diagnostic_protocol.json`. It does not select or change a deployed model. The current runner includes it in the initial design.

Write the primary design before attention training. Use 16 **new** calibration maps (seed base 7,200,000) and 64 **new** benchmark maps (7,300,000). Freeze checkpoints, covariance and global interval scales before benchmark generation. Keep the gradual nonstationary field law, geometry, known mean/noise assumptions, start, evaluation mask, and physical caps. The benchmark is independent of earlier exposed test maps, but it is not a spatial-law distribution-shift test.

The seven systems are fitted stationary-RBF GP, frozen DeepKriging, the GP prior learned from the same offline maps, two pooled CNP seeds, and two attention seeds. All use native latent marginal variance divided by distance, restricted to affordable unvisited sensing sites. Cross caps of 8/16 measurements and travel distance 1/2. The starting reading counts, and the 45 evaluation sites cannot be sensed. No obstacles, robot dynamics, or return-to-base costs are added.

Predeclare a seven-by-seven final-route crossover: every mapper receives every route family's exact acquired positions and noisy values. Reuse native fits, save fixed predictions before reading evaluator targets, and preserve source files. Bootstrap paired differences over independent fields. The intervals condition on the fixed training runs; they do not cover all training-seed variability or adjust for multiple comparisons.

## Same observations and equal training duration

At caps (16,2), give both architectures the observations collected on the fitted-GP routes, and fix training to **6,000 updates** for both:

| Training seed | Pooled CNP RMSE | Attention-CNP RMSE | Attention minus pooled, paired 95% interval |
| ---: | ---: | ---: | --- |
| 0 | 0.8397 | **0.7644** | -0.07534 [-0.09613, -0.05467] |
| 1 | 0.8230 | **0.7667** | -0.05627 [-0.07542, -0.03825] |

Attention reduces mean error by approximately **9.0% / 6.8%** on these common acquired observations. This supports a reconstruction benefit for this controlled architecture change, rather than explaining the result solely by more updates or more parameters. Width reallocation and only two training runs still limit causal/general claims.

On these same contexts, mean RMSE against acquired noisy targets falls from 0.6343/0.6338 to **0.2248/0.2559**. Native mean latent standard deviation at those sites falls from 0.7807/0.7494 to **0.4689/0.4415**. This is evidence of stronger local conditioning, although substantial underconditioning remains and acquired-target fit is not held-out accuracy. Perfect interpolation of noisy values is not the goal.

## Selected-pipeline crossover

At caps (16,2), the final fitted-GP route has 8.50 readings and distance 1.965 on average. Keep those exact observations fixed:

| Frozen mapper | Held-out latent RMSE |
| --- | ---: |
| Fitted GP | 0.7878 |
| DeepKriging | 0.8237 |
| **Training-map GP** | **0.7336** |
| Pooled CNP seed 0, selected 3,000 updates | 0.8425 |
| Pooled CNP seed 1, selected 3,000 updates | 0.8407 |
| Attention seed 0, selected 6,000 updates | 0.7644 |
| Attention seed 1, selected 6,000 updates | 0.7667 |

Attention is clearly better than the pooled neural baselines in this condition. Its mean error is also lower than fitted GP, but paired attention-minus-GP intervals **include zero**: -0.02342 [-0.04965, 0.00142] and -0.02112 [-0.04668, 0.00378]. Do not claim an established GP advantage reversal. The same-data learned-prior GP still has lower error than both attention models, with positive paired attention-minus-learned-GP intervals.

## Each system chooses its own route

| Count cap | Travel cap | Fitted GP | Pooled CNP 0 | Pooled CNP 1 | Attention 0 | Attention 1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.9473 | 0.8991 | 0.9451 | 0.9243 | 0.9023 |
| 8 | 2 | **0.8086** | 0.8991 | 0.9451 | 0.9243 | 0.8948 |
| 16 | 1 | 0.9473 | 0.9018 | 0.9598 | 0.9251 | 0.9018 |
| 16 | 2 | **0.7878** | 0.9013 | 0.9611 | 0.9035 | 0.8585 |

At caps (16,2), attention seed 0 is effectively unchanged versus pooled seed 0: paired difference +0.00219 [-0.01984, 0.02465]. Attention seed 1 improves versus pooled seed 1 by -0.10267 [-0.13222, -0.07574]. Neither attention system beats fitted GP: differences are +0.11568 [0.07699, 0.15608] and +0.07063 [0.02912, 0.11130]. The training-map GP has own-system error 0.8332 and DeepKriging 0.8981. Preserve all resource conditions rather than choosing the favorable one.

| System, caps (16,2) | Actual readings | Actual travel | Mean cumulative online prefix computation |
| --- | ---: | ---: | ---: |
| Fitted GP | 8.50 | 1.965 | 63.0 ms |
| Pooled CNP 0 / 1 | 16.00 / 16.00 | 1.143 / 1.071 | 4.0 / 2.6 ms |
| Attention 0 / 1 | 16.00 / 15.50 | 1.425 / 1.845 | 5.3 / 4.1 ms |

Equal upper bounds do not imply equal consumption. More neural readings still fail to establish better exploration. New attention training, including checkpoint validation/saving, costs 51.63 s; the original pooled runs cost 41.60 s. Both ran through all 6,000 updates before checkpoint selection. These one-thread CPU records exclude data simulation and do not establish isolated deployment latency. Per-prefix timings include prediction and amortized DeepKriging fitting, while offline work is separate.

All **1,792 episodes** obey both resource caps and avoid evaluation sites. There are 6,756 unique prefix states. Calibration collection costs 12.11 s and benchmark collection 42.56 s. The crossover scores 12,544 endpoint/model pairs from 8,428 unique requested fits; native diagonal reproduction error is exactly zero. Source data, deployed weights, and prior remain unchanged.

## Remaining uncertainty and routing problem

At caps (16,2), native 95% marginal coverage is 96.67%/95.38% for attention, with mean held-out latent standard deviation 0.9646/0.8676. Global variance multipliers 1.2706/1.3354 produce calibrated coverage 98.33%/97.74%. Coverage is descriptive, spatial sites are correlated, and positive scalar scaling cannot change action rankings. Broad coverage does not certify an informative spatial uncertainty map or a correct hypothetical-conditioning rule.

Attention reconstruction on GP-selected data is 0.7644/0.7667, versus 0.9035/0.8585 on its own data. Conversely, fitted GP on attention-selected data has errors 0.9110/0.8720, versus 0.7878 on its own route. Thus poor sensing geometry remains visible even when changing the reconstruction model. These route contrasts include actual count differences and remain conditional descriptive comparisons, not causal allocations.

The predefined field-0 example shows query-specific attention weights and also a route concentrated near a narrow region. Attention's outputs can use nearby observations more effectively without the marginal-variance/distance score becoming a strong final-map planner. A local variance score is not the same objective as expected reduction in whole-map error. The network's factorized marginals still do not provide a coherent joint covariance for exact GP information-gain updates.

## Next useful test

The evidence supports improved context-conditioned reconstruction, not a complete exploration advantage. The next priority is to hold the mapper fixed while testing sensing decisions, rather than expanding network size immediately. The learned-covariance GP already supplies a deployable joint covariance without true-kernel access, so it can support the existing integrated-variance planner as a common route-design control for neural and GP reconstructions. Any claimed neural information-gain alternative needs its own validated hypothetical-measurement update.

Select future settings on development data and generate another frozen test set. Do not tune on these now-exposed fields, promote a single training seed, describe this as full ANP, or attribute the entire change to attention without acknowledging the width and checkpoint controls.

## Reproduction and validation

```sh
# First create the original development data and pooled CNP checkpoints if absent.
PYTHONPATH=src .venv/bin/python -m spatial_exploration.cnp_study --output outputs/cnp-study
PYTHONPATH=src .venv/bin/python -m spatial_exploration.attention_study --reference outputs/cnp-study --output outputs/attention-study
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Use fresh output folders. The attention command preserves its reference and creates a sibling `outputs/attention-study-crossover`. Outputs include copied development inputs, model checkpoints, validation traces, selection/freeze records, acquired/evaluator archives, step/final errors and resources, all crossover predictions, fixed-step diagnostic predictions and paired results, context diagnostics, query-attention weights, figures and hashes. `example_attention.png` uses seed 0, field 0, maximum resource caps and four fixed query positions, without selecting a favorable case.

The full environment passes **81 tests**. System Python runs 81 with 37 optional tests skipped. New checks cover parameter matching, uniform-attention equivalence, context permutation, value sensitivity, query-specific normalized weights, query independence, attention gradients, reuse of exact development data, retained seeds, checkpoint selection, freeze-before-new-test ordering, both resource caps, fixed-step artifacts, source preservation, and exact native crossover reproduction. Figures were rendered and visually inspected.
