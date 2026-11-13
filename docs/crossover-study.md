# Frozen-route crossover diagnosis

This page records the original two-model post-hoc diagnosis. The later [same-basis Bayesian control](basis-study.md) uses a new independently tuned neural setting, fresh fields, and a prespecified three-model crossover. Its balance of reconstruction and sampling contrasts differs; keep the two experiments distinct.

## Purpose and limits

The [uncertainty-driven experiment](uncertainty-study.md) changes both the observations collected and the model used to reconstruct the field. To investigate the source of its accuracy differences, hold each final acquired route fixed and apply both frozen mappers to its measurements. This produces a two-by-two table of route origin and reconstruction model under each common resource pair.

This is a **post-hoc diagnostic on already exposed fields**, not a new final test, a new policy, or a causal experiment that independently randomizes routes. No route, architecture, optimizer setting, epoch count, or regularization value is selected or changed. The sampling-design contrast includes both locations and actual reading counts; it is not a comparison of locations at identical count. Different policies may collect different counts under the same physical upper bounds.

## Procedure

Load the exact acquired positions/values from the saved closed-loop traces. Fit the counterpart mapper using only those measurements; do not give it hidden field targets. Retain GP noise assumptions and bounded snapshot fitting, and neural 100-epoch/0.0001-decay three-member fitting. Use the original member seed rule for each field/count. Reuse any fit on an identical original prefix rather than changing the predictions of a native result.

Write all fixed crossover predictions before loading evaluator-only field targets. Score the same 45 withheld sites. Hash-check source data before and after; record route/configuration hashes too. Original native diagonal outcomes are retained exactly by prediction reuse. The archive contains 1,024 endpoint evaluations, representing both mappers on 512 fixed final routes. Deduplication yields 678 requested fits; 189 new GP fits and 150 new neural ensembles are needed. The remaining predictions reuse original fits.

Within a fixed route, paired model differences isolate reconstruction under identical acquired values. Within a fixed mapper, paired route differences compare the observations selected by the two original systems. These contrasts remain conditional on the original adaptive selection process.

## Results

At caps of 16 readings and distance 2, mean held-out latent RMSE across 64 paired fields is:

| Original route selector | Reconstruct with fitted GP | Reconstruct with DeepKriging-style |
| --- | ---: | ---: |
| GP | 0.8363 | 0.8688 |
| DeepKriging-style | 0.8424 | 0.8806 |

On GP-selected data, changing the mapper to DeepKriging raises mean RMSE by 0.03250 (paired field-bootstrap 95% interval [-0.00185, 0.06444]). On DeepKriging-selected data it raises error by 0.03823 (interval [0.00435, 0.07253]). The first interval includes zero; do not claim uniform superiority on every conditional comparison.

Changing from GP-selected to neural-selected observations with the GP mapper changes mean RMSE by +0.00602 (interval [-0.04619, 0.05472]). With the neural mapper the change is +0.01174 (interval [-0.02341, 0.04662]). Thus the mean reconstruction contrast is larger than the route contrast in this particular resource condition, and route contrasts are imprecise. This is more informative than attributing the complete observed difference to path appearance alone.

A symmetric descriptive decomposition averages the two possible change orders. Let GG/GD/DG/DD denote the table's four per-field errors (route letter first). Define:

- Sampling-design component: `0.5*((DG-GG)+(DD-GD))`.
- Reconstruction-model component: `0.5*((GD-GG)+(DD-DG))`.

Their sum equals `DD-GG` for each field. At caps (16,2), the total mean difference +0.04424 splits into +0.00888 sampling (interval [-0.02746, 0.04324]) and +0.03536 reconstruction (interval [0.01203, 0.05873]). This arithmetic is not a percentage of proven causal attribution. Other budget pairs have different balances; all four matrices and paired contrasts are archived. For example, at caps (8,2), the mean sampling component is 0.03167 and reconstruction component 0.02445, with both component intervals crossing zero.

### Mechanism interpretation

Several features of the implementation plausibly explain a reconstruction disadvantage, but this crossover does not separately test them:

1. **Informative GP structure and assumptions.** The fields are Gaussian-process mixtures. The fitted GP is misspecified as stationary single-RBF, but its smooth covariance structure, known zero mean, and explicit known sensor noise remain useful priors with few observations. Neural spatial features encode geometry without defining the same probabilistic conditioning rule or explicit noise model.
2. **Small-data interpolation and extrapolation.** The adapted network has 8,673 parameters and is fitted fresh on very few readings. In the caps-(16,2) crossover, mean RMSE against acquired noisy targets is about 0.00100 on GP routes and 0.00109 on neural routes. Corresponding GP acquired-target RMSE is 0.01734/0.04331. The neural predictor nearly interpolates acquired noise but has larger unseen-site error. This is consistent with weak generalization/regularization in this setup; parameter count alone and these training errors do not prove that noise overfitting is the sole cause.
3. **Different meanings of uncertainty.** GP marginal variance comes from an explicit spatial covariance/noise model; neural spread measures disagreement among just three initializations. Agreement can coexist with wrong predictions, and fresh fitting/seeds can change the spread pattern after each reading. Global interval scaling cannot repair spatial ranking. The source run's poor native neural coverage and higher evaluator-only route variance are evidence of uncertainty limitations, not a complete explanation of every accuracy difference. GP plug-in uncertainty also undercovers, and parameter refits can increase its reported uncertainty.

The data support examining reconstruction regularization/inductive bias alongside uncertainty quality, rather than assuming the visible route difference is the primary cause. They do not establish that DeepKriging generally fails, that a GP is optimal under realistic unknown mean/noise, or that increasing the ensemble or training duration would necessarily help. Such changes need separate development maps and a fresh frozen test.

## Reproduction

```sh
PYTHONPATH=src .venv/bin/python -m spatial_exploration.crossover_study --reference outputs/uncertainty-study --output outputs/uncertainty-crossover
```

Use a fresh output folder. Save source hashes/settings, per-route mapper metrics, prediction arrays and exact observed-label indices, all four budget matrices, paired differences, the descriptive decomposition, a heatmap, and source-preservation diagnostics. New recorded fit totals are 2.22 seconds GP and 3.08 seconds neural ensembles under the usual single-map versus batched timing scopes. These are diagnostic fitting costs and do not change the original deployed-system timing record.
