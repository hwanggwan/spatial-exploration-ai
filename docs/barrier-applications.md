# Application direction: dependence follows connectivity

## Why a sharper transition is not the whole application

The user proposed obstacles or terrain that change spatial dependence. A location-varying correlation scale, a discontinuous mean, an anisotropic field, a physical barrier, and a restricted robot path are distinct mechanisms. A benchmark should identify which mechanism it represents before selecting a model.

The current transition study changes mixing weights on a square with no obstacles. It does not encode where heat, air, water, radio signals, or a robot can travel. A physical barrier can make nearby coordinates poorly connected while distant coordinates remain connected along a corridor or channel. That is a change in topology or transport, beyond a sharper version of the existing mixing function.

## Evidence from applications

| Application | Evidence | Consequence for our model design |
| --- | --- | --- |
| Coastal water monitoring around islands/peninsulas | Bakka et al. describe inappropriate stationary smoothing across coastlines and apply a barrier-aware Gaussian model in the Finnish Archipelago Sea. [Author repository and paper](https://research-repository.st-andrews.ac.uk/handle/10023/19307) | Water sites separated by land should not be coupled solely by Euclidean proximity. A boat's feasible route and the water-field dependence both need explicit geometry. |
| Stream temperature or water quality | US Forest Service spatial stream-network models distinguish Euclidean, flow-connected, and flow-unconnected relationships and account for in-stream distance. [USFS SSN project](https://research.fs.usda.gov/rmrs/projects/spatial-stream-network) | Encode the river network and flow connections rather than interpolating indiscriminately across a 2D map. |
| Indoor air-quality mapping | PNNL's sensor guide notes room-to-room concentration differences and the limitations of one fixed sensor for characterizing a home. [PNNL guide](https://basc.pnnl.gov/resource-guides/indoor-air-quality-sensors) | A room/door/ventilation transport model is a plausible project hypothesis; the guide does not establish zero covariance across walls or validate our simulator. |
| Geology and groundwater influence on streams | USGS describes spatially discontinuous groundwater influences and controls from recharge, topography, geology and landcover. [USGS monitoring project](https://www.usgs.gov/centers/new-jersey-water-science-center/science/new-jersey-subbasin-paired-air-stream-water) | Terrain and hydrogeology can be covariates or transport controls; a different mean alone should not be mislabeled different residual dependence. |

These sources establish mechanisms and modeling precedents, not evidence that DeepKriging will outperform a GP. The barrier paper itself is a Gaussian-model solution. Walls and boundaries also do not imply complete independence: doors, ducts, shared weather, or transport around a barrier may retain dependence. The appropriate coupling depends on the measured quantity and timescale.

## Room-and-door experiment design

The [implemented geometry-aware study](barrier-study.md) tests two door placements and compares a graph-informed GP against a coordinate-only GP under both fixed and closed-loop acquisition. A room-and-door environmental mapping simulator is a concrete first application candidate because it connects mobile sensing, spatial prediction, and budgeted exploration in an interpretable layout. This is an experimental design proposal, not a claim that a real robot or building has been modeled or validated.

Start with two rooms separated by a wall with one doorway. Give the mapper the known floor plan; unknown-layout discovery is a separate task. A source or varying transport can create different field behavior by room. First hold mean and marginal variance constant to isolate dependence, then add source/mean effects as a separate condition. Begin with a frozen snapshot and disclose the static approximation; real ventilation can require time-dependent inference.

Use **two graphs** with separately specified edges:

- Movement graph: where the robot can go, with edge lengths and traversal costs. Wall crossings are infeasible; routes go through the door.
- Field-transport graph: how the target quantity couples nearby locations. Door/duct connections can be weak or strong. Heat conduction through a wall need not have the same connectivity as air transport or robot motion.

One tractable synthetic field generator is a positive-definite graph model `K_raw = (kappa^2*I + L)^(-alpha)`, with positive kappa/alpha and a weighted graph Laplacian L. Standardize its diagonal to one when isolating dependence. Reduce or remove transport edges across walls and keep a doorway edge. This is our proposed discrete generator, not a reproduction of the cited continuum barrier-SPDE method. Generate field values with evaluator-owned covariance, keeping the transport parameters hidden in non-oracle treatments. Do not assume that substituting arbitrary shortest-path distances into an RBF always yields a valid covariance matrix.

## Comparisons needed for a defensible AI result

1. Coordinate-only stationary GP and the current Euclidean-basis neural mapper: references that lack explicit barrier connectivity.
2. A fitted graph/barrier-aware GP: a stronger baseline, with physical parameters estimated only from available observations and separate development data.
3. A neural mapper with barrier/region/graph features: the AI extension. Compare it against a linear predictor on the same features to separate representation gains from nonlinear-network gains.
4. An oracle transport model: an explicit evaluator/reference with known generating parameters, not a deployable baseline.

Both non-oracle GP and neural models should receive the same known layout/covariates, labels, and development resources. A neural win against only an obstacle-blind GP would establish the value of supplying geometry, not the superiority of neural learning.

Hold observed sites fixed first, including points close together across a wall, points far apart in a connected room, and the doorway neighborhood. Vary barrier permeability and doorway placement on prespecified layouts. Reserve entirely new layouts as well as new field realizations for a frozen test; a coordinate network fitting one layout does not demonstrate transfer to another.

Only after mapper accuracy and uncertainty are credible, add movement constraints and compare planners under the same graph travel budget. Track held-out error, near-wall/door error, coverage and interval width, distance, sensing count and cost, route feasibility, model update time, and training cost. Use actual field observations or validated physical simulations before claiming application performance.
