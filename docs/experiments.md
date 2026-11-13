# Experiment index

Start with the [Field Routes demo](../demo/index.html) and the [three-room benchmark](three-room-study.md). The studies below answer different questions; compare numbers within a study because fields, budgets, and evaluation designs vary.

| Question | Studies |
| --- | --- |
| How should measurements be allocated across rooms? | [Three unequal rooms](three-room-study.md), [two-room quotas](wall-room-quota-study.md), [greedy versus rollout with wall-aware ridge](wall-rollout-study.md) |
| Does the modeled wall improve prediction or route choice? | [Applications and simulator assumptions](barrier-applications.md), [geometry-aware GP](barrier-study.md), [wall topology control](wall-topology-study.md), [closed-loop and route replay](wall-closed-loop-study.md) |
| Do graph features help, and do they need a neural network? | [Graph DeepKriging](barrier-deepkriging-study.md), [same-feature ridge comparison](barrier-ridge-study.md), [earlier Bayesian basis control](basis-study.md) |
| How do mappers behave with scarce observations and under changing dependence? | [Covariance estimation](covariance-study.md), [sample size](sample-size-study.md), [DeepKriging-style mapping](deepkriging-study.md), [sharper transitions](transition-study.md), [fixed-route replay under two caps](resource-study.md), [uncertainty-driven routes](uncertainty-study.md), [mapper and route crossover](crossover-study.md) |
| When is it worth planning more than one move ahead? | [Two-step lookahead](lookahead-study.md), [small exact reference](exact-study.md), [greedy-completion rollout](rollout-study.md) |
| Can learned models improve mapping or exploration? | [Custom-task PPO](rl-study.md), [imitation learning](imitation-study.md), [frozen policy robustness](robustness-study.md), [conditional neural process](cnp-study.md), [attention-based mapper](attention-study.md) |
| What can be reproduced from the DeepKriging publication and its public code? | [Paper-function benchmark](paper-mapping-study.md), [Keras weight reuse control](keras-reuse-audit.md), [published-table provenance](paper-provenance-audit.md), [original-code review](original-code-review.md) |

Each report records its assumptions, reproduction commands, evaluation metrics, and limitations. The [original experiment plan](experiment-plan.md) describes how the project was staged.
