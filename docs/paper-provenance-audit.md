# Tracing the DeepKriging tables to public artifacts

Audited November 5, 2026. The repository's complete public branch history (14 commits), 16 distinct notebook blobs reachable from that history, all four arXiv source archives, the published supplement, and public issues/PRs were inspected without rerunning training or contacting anyone. The previous [same-Keras audit](keras-reuse-audit.md) remains unchanged.

**Two numerical links are now established: the earlier manuscript's 2D Table 2 means match a saved 2020 notebook, and its GP fold results also match the published Table S4 GP mean/SD. The final Table S2 neural results are still not linked to an execution artifact.** Numerical agreement identifies a strong source relationship; it does not independently certify that every displayed notebook cell was the exact code executed to produce its saved output.

## Earlier table: saved fold output matches the manuscript

The [July 19, 2020 commit f321daa](https://github.com/aleksada/DeepKriging/commit/f321daa) introduced the 2D experiment. Its [saved notebook](https://github.com/aleksada/DeepKriging/blob/f321daa/Nonstationary_function-2d.ipynb) contains ten fold-level MSE/MAE outputs and a cell printing their means and population standard deviations. The corresponding 2D Table 2 in [arXiv v3, page 23](https://arxiv.org/pdf/2007.11972v3#page=23), also present in v1 and v2, reports MSE and MAE rather than the final supplement's RMSE/MAPE.

| Statistic, in the earlier table's scaled units | From saved 2020 folds | Earlier manuscript |
| --- | ---: | ---: |
| DeepKriging mean MSE × 10,000 | 0.433278 | 0.43 |
| Coordinate DNN mean MSE × 10,000 | 0.937697 | 0.94 |
| GP mean MSE × 10,000 | 2.899877 | 2.90 |
| DeepKriging mean MAE × 1,000 | 3.656069 | 3.66 |
| Coordinate DNN mean MAE × 1,000 | 5.141345 | 5.14 |
| GP mean MAE × 1,000 | 6.253740 | 6.25 |

All six mean cells match ordinary rounding. Most SD cells also agree; the earlier DeepKriging MSE SD is printed with a typographical ambiguity (`.0.68`) while the saved scaled SD is 0.686560. Do not claim every SD cell matches conventional rounding.

The visible notebook loop constructs models outside cross-validation and retains weights, the same behavior examined in our training-history audit. Its ten DeepKriging MSE values imply mean fold RMSE 0.00536117. Our modern reused-weight diagnostic's mean is 0.00541376, whereas the fresh-400 control is 0.01766695. This is consistent with the source-oriented diagnostic but is not proof of the exact historical random state or a validation of contaminated CV.

## Published Table S4: a GP numerical fingerprint

Taking the square root of each saved GP fold MSE, then aggregating those ten RMSEs, gives:

| Quantity | Saved 2020 notebook | Our independent GP fit | Published Table S4 |
| --- | ---: | ---: | ---: |
| Mean RMSE | 0.0151354579 | 0.0151355050 | 0.015135 |
| Population SD across folds | 0.0078042042 | 0.0078041978 | 0.007804 |

The published values are from [supplement Table S4, page 14](https://www3.stat.sinica.edu.tw/statistica/password.asp?art=13&num=1&sup=s&vol=34). Both means and SDs agree at the table's printed precision. The largest saved-versus-ours per-fold RMSE difference is 9.33e-7. This makes a major GP implementation discrepancy an implausible explanation for this particular Table S4 baseline. Our earlier summary CSV uses sample SD (`ddof=1`); this provenance comparison deliberately computes population SD (`ddof=0`) to match the notebook's `np.std` convention.

This is a link to the **GP columns only**, not a reproduction of Table S4's neural columns or its complete experimental pipeline. It also does not explain the different GP value 0.008552 in Table S2. The broad phrase “none of the paper's numerical values were recovered” would now be inaccurate; the unresolved target is specifically the final Table S2 and the neural table entries.

## Final Table S2: a version gap remains

The public chronology is:

| Date | Artifact | Relevant evidence |
| --- | --- | --- |
| 2020-07-19 | Git commit f321daa | Adds 2D notebook/Python code; saved outputs match earlier table means |
| 2020-07-23 / 2020-07-25 | arXiv v1 / v2 | Earlier MSE/MAE table |
| 2021-07-30 | arXiv v3 | Same earlier table means |
| 2022-01-15 | Git commit 0ff987f | Latest public code commit in the audited repository |
| 2022-05-24 | arXiv v4 | Final-style Table S2 includes DK RMSE 0.003466 and GP RMSE 0.008552 |
| 2024 | Journal publication | These Table S2 values appear in the published supplement |

Dates and revisions are documented in the [arXiv version history](https://arxiv.org/abs/2007.11972) and [last public code commit](https://github.com/aleksada/DeepKriging/commit/0ff987f). The final numbers first appear among the four inspected manuscript source versions in [v4 Table S2](https://arxiv.org/pdf/2007.11972v4#page=53). They are literal numeric constants in its LaTeX table. None of the inspected arXiv source archives contains a Python/R execution script, raw fold predictions, or a script connecting model execution to those table entries. These source archives contain manuscript/figure assets, not the missing final experiment workflow.

No saved plain-text numeric notebook output in the inspected public history matches the final Table S2 mean RMSE values 0.003466 / 0.006934 / 0.008552 within their printed rounding tolerance. This negative search is limited to these public artifacts; it does not exclude an unpublished run, unpushed code, another repository, or an unavailable historical environment.

**Inference:** the public code may predate the final revised experiment. That is consistent with the chronology and matching earlier table, but not established fact about the authors' workflow. Table S2 could also involve other unrecorded fitting, split, initialization, or model-selection changes. The current evidence cannot assign its numerical difference specifically to leakage or to a publication error.

Public discussion supports an environment/provenance gap without resolving it: [issue #5](https://github.com/aleksada/DeepKriging/issues/5) requests library/version requirements and has no reply in the inspected thread; [PR #4](https://github.com/aleksada/DeepKriging/pull/4) proposes correcting cell order and remains open. Neither identifies the final Table S2 run.

## Evidence needed to close the gap

The most direct next step is to request the **final Table S2 execution artifact**, rather than another generic repository link:

- The exact notebook/script or Git commit that generated the final table, and its cell execution order.
- Python, Keras, TensorFlow, and GPy versions, plus all random seeds and fold indices.
- Fold-level predictions/errors and the aggregation script, including MSE-to-RMSE handling and SD convention.
- Confirmation of whether weights and optimizer state reset between folds, and how depth/epochs were selected.

A final execution artifact with matching fold-level predictions would establish much more than obtaining a similar aggregate mean through tuning. An author explanation could clarify historical intent and missing files. No email, GitHub issue, or other message was sent as part of this audit.

## Local evidence

`outputs/paper-provenance-audit/findings.json` records the conclusions and limits. `legacy_table_comparison.csv` verifies the six mean-cell matches; `legacy_fold_output.txt` preserves their ten-fold source output, and `legacy_loop_source.py.txt` preserves the visible loop. The four `arxiv-v*-2d-tables.tex.txt` files preserve the table definitions inspected, and `manifest.json` hashes the evidence and downloaded source archives. The full clone and original arXiv archives are kept under ignored `external/` directories. Prior training and evaluation outputs are not modified.
