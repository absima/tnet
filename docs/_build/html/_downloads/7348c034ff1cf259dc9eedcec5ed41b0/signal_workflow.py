"""Compare FC methods and transformations using generated signals."""
import numpy as np
from tnet_analysis.analysis_preparation import PrepareConnectivity, PrepareTemporalNetwork
from tnet_analysis.dfc_to_tnet import FindThresholdsForAverageDegrees
from tnet_analysis.latency_measures import (
    ComputeAverageDegree, ComputeSmartTemporalDistanceMeasures,
)

rng = np.random.default_rng(100)
signals = rng.standard_normal((100, 8))
degrees = [1.0, 2.0]
tags = [
    "original", "emp_static", "time", "uniform", "phase_rand0", "phase_rand1",
    "er_static", "sw_static", "sf_static", "eAct", "eActTemplate",
    "eRndTemplate", "erand_er", "erand_sws", "erand_sfs", "erand_swr",
    "erand_sfr",
]

for method_index, method in enumerate(["original1", "original2"]):
    for tag_index, tag in enumerate(tags):
        seed = 1000 + 100 * method_index + 10 * tag_index
        fcs = PrepareConnectivity(
            signals, tag=tag, fc_method=method, window=20, lag=5,
            random_seed=seed,
        )
        thresholds = FindThresholdsForAverageDegrees(fcs, degrees)
        for degree_index, degree in enumerate(degrees):
            network = PrepareTemporalNetwork(
                fcs, tag=tag, target_degree=degree,
                threshold=thresholds[degree_index], seed=seed + degree_index,
            )
            achieved, _ = ComputeAverageDegree(network)
            metrics = ComputeSmartTemporalDistanceMeasures(network, summary_stat="both")
            print(method, tag, degree, round(achieved, 3), metrics)
