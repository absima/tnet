"""Process subjects in independent workers; no files are written."""
import numpy as np
from joblib import Parallel, delayed, parallel_config
from tnet_analysis.analysis_preparation import PrepareConnectivity, PrepareTemporalNetwork
from tnet_analysis.latency_measures import ComputeSmartTemporalDistanceMeasures
from tnet_analysis.segregation_and_cohesion import ComputeSpatialClustering


def ComputeSubject(subject_index, signals):
    seed = 2000 + subject_index
    fcs = PrepareConnectivity(signals, window=20, lag=5)
    network = PrepareTemporalNetwork(fcs, "original", target_degree=2, seed=seed)
    no_diagonal = network.copy()
    diagonal = np.arange(network.shape[1])
    no_diagonal[:, diagonal, diagonal] = 0
    return {
        "subject_index": subject_index,
        "smart": ComputeSmartTemporalDistanceMeasures(network, summary_stat="both"),
        "spatial_clustering": ComputeSpatialClustering(no_diagonal, "both"),
    }


if __name__ == "__main__":
    subjects = np.random.default_rng(200).standard_normal((4, 100, 8))
    with parallel_config(backend="loky", n_jobs=2, inner_max_num_threads=1):
        results = Parallel()(
            delayed(ComputeSubject)(index, signals)
            for index, signals in enumerate(subjects)
        )
    for result in results:
        print(result)
