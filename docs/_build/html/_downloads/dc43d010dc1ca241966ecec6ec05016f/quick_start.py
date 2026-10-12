"""Analyze a small generated temporal network."""
import numpy as np
from tnet_analysis.latency_measures import (
    ComputeSmartTemporalDistanceMeasures, ComputeDrunkTemporalDistanceMeasures,
    ComputeReturnLatency,
)
from tnet_analysis.dynamism_and_memory import ComputeDynamism, ComputeRetention
from tnet_analysis.segregation_and_cohesion import (
    ComputeSpatialClustering, ComputeTemporalClustering,
)

rng = np.random.default_rng(42)
upper = np.triu(rng.binomial(1, 0.3, size=(30, 8, 8)), k=1)
structure_network = (upper + upper.transpose(0, 2, 1)).astype(np.uint8)
walk_network = structure_network.copy()
walk_network[:, np.arange(8), np.arange(8)] = 1

results = {
    "smart": ComputeSmartTemporalDistanceMeasures(walk_network, summary_stat="both"),
    "drunk": ComputeDrunkTemporalDistanceMeasures(
        walk_network, n_reps=20, random_seed=43, summary_stat="both"),
    "dynamism": ComputeDynamism(walk_network),
    "spatial_clustering": ComputeSpatialClustering(structure_network, "both"),
    "temporal_clustering": ComputeTemporalClustering(
        structure_network, delta=1, summary_stat="both"),
    "return": ComputeReturnLatency(walk_network, summary_stat="both"),
    "retention": ComputeRetention(structure_network, "both", delta=5),
}
for name, values in results.items():
    print(name, values)
