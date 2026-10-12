"""Run directly to smoke-test all measure APIs on subject index 5.

This checks execution, not scientific validity. NaN can be valid for isolated
nodes or insufficient observations. Results and failures remain available as
Python dictionaries. Function aliases are not counted as separate measures.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnet_analysis.dfc_to_tnet import BinarizeDFC
from tnet_analysis import latency_measures as distance
from tnet_analysis import dynamism_and_memory as memory
from tnet_analysis import segregation_and_cohesion as structure
from tnet_analysis import modularity_estimates as community


# Change these here; no command-line arguments or runner configuration needed.
input_file = "data/input/time_series.mat"
input_file = '/Users/sima/Documents/MATLAB/empirical/ts_100subjs.mat'
data_key = "tseries"
isubj = 5                         # Zero-based subject index, as in the other example.
degDist = np.logspace(np.log10(0.1), np.log10(83.04), 54)
avdeg = degDist[10]
window, lag = 20, 1
rand_seed, nreps = 85869, 100
max_return_latency = None        # Full horizon; optionally limit this for a quick check.


def Describe(value):
    """Print small outputs, but summarize matrices rather than dumping them."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {Describe(item)}" for key, item in value.items()) + "}"
    if isinstance(value, tuple):
        return "(" + ", ".join(Describe(item) for item in value) + ")"
    if isinstance(value, str) or value is None:
        return repr(value)
    array = np.asarray(value)
    if array.size <= 10:
        return str(value)
    return (f"array{array.shape}, finite={np.isfinite(array).sum()}, "
            f"NaN={np.isnan(array).sum()}, inf={np.isinf(array).sum()}")


def RunMeasureChecks(tnet, random_seed=85869, n_reps=100, max_latency=None):
    """
    Execute every canonical measure API, continuing after individual failures.

    Args:
        tnet: np.ndarray, shape (T, N, N). Nonempty symmetric binary temporal network.
        random_seed: int. Reproducible walk and community-detection seed.
        n_reps: int. Number of drunk-walk trials.
        max_latency: int or None. Return-path search limit; None uses the full horizon.

    Returns:
        results: dict. Successful function outputs, with arrays retained in full.
        failures: dict. Function names mapped to error messages.
    """
    tnet = np.array(tnet, copy=True)
    if tnet.ndim != 3 or tnet.shape[1] != tnet.shape[2] or not all(tnet.shape):
        raise ValueError("tnet must have shape (T, N, N), with T, N > 0.")
    diagonal = np.arange(tnet.shape[1])
    tnet[:, diagonal, diagonal] = 1
    nodiag = tnet.copy()
    nodiag[:, diagonal, diagonal] = 0
    # Use an informative snapshot rather than one that might have no edges.
    snapshot = nodiag[np.argmax(nodiag.sum(axis=(1, 2)))]
    results, failures = {}, {}

    def Check(name, compute):
        print(f"\n{name} ...", flush=True)
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = compute()
            results[name] = value
            print(f"  OK ({time.perf_counter()-started:.2f}s): {Describe(value)}", flush=True)
            for message in dict.fromkeys(str(item.message) for item in caught):
                print(f"  warning: {message}", flush=True)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"  FAILED: {failures[name]}", flush=True)

    # Walk matrices also exercise the lower-level distance-summary entry points.
    Check("ComputeAverageDegree", lambda: distance.ComputeAverageDegree(nodiag))
    Check("SmartWalker", lambda: distance.SmartWalker(tnet))
    Check("RandomWalker", lambda: distance.RandomWalker(tnet, n_reps, seed=random_seed))
    Check("MeanLatencyMatrixAnalysis", lambda: distance.MeanLatencyMatrixAnalysis(
        results["SmartWalker"], tnet.shape[0]+1, summary_stat="both"))
    Check("ComputeSmartTemporalDistanceMeasures", lambda: distance.ComputeSmartTemporalDistanceMeasures(tnet, summary_stat="both"))
    Check("ComputeDrunkTemporalDistanceMeasures", lambda: distance.ComputeDrunkTemporalDistanceMeasures(tnet, n_reps=n_reps, random_seed=random_seed, summary_stat="both"))
    Check("ComputeReturnLatency", lambda: distance.ComputeReturnLatency(tnet, max_latency=max_latency, summary_stat="both"))

    for function in [memory.ComputeTemporalMutualInformation, memory.ComputeDynamism,
                     memory.ComputeTemporalEdgeOverlap, memory.ComputeEdgePersistenceRate,
                     memory.ComputeLinkBurstiness]:
        Check(function.__name__, lambda function=function: function(nodiag))
    for function in [memory.ComputeNeighborhoodMemory, memory.ComputeRetention,
                     structure.ComputeSpatialClustering, structure.ComputeTemporalClustering,
                     structure.ComputePartnerStability, structure.ComputePartnerDiversity,
                     structure.ComputeNodePersistence]:
        Check(function.__name__, lambda function=function: function(nodiag, summary_stat="both"))
    Check("ComputeSnapshotTransitivity", lambda: structure.ComputeSnapshotTransitivity(snapshot))
    Check("ComputeTemporalTransitivity", lambda: structure.ComputeTemporalTransitivity(nodiag))

    Check("DetectCommunitiesLouvain", lambda: community.DetectCommunitiesLouvain(snapshot, seed=random_seed))
    labels = lambda: results["DetectCommunitiesLouvain"][0]
    Check("SnapshotModularity", lambda: community.SnapshotModularity(snapshot, labels()))
    Check("ComputeSnapshotModularity", lambda: community.ComputeSnapshotModularity(snapshot, detect_kwargs={"seed": random_seed}))
    Check("ComputeTemporalModularity", lambda: community.ComputeTemporalModularity(nodiag, detect_kwargs={"seed": random_seed}))
    frame_labels = lambda: results["ComputeTemporalModularity"][2]
    Check("ComputeSnapshotParticipationCoefficient", lambda: structure.ComputeSnapshotParticipationCoefficient(snapshot, labels()))
    Check("ComputeTemporalParticipationCoefficient", lambda: structure.ComputeTemporalParticipationCoefficient(nodiag, frame_labels()))
    print(f"\nFinished: {len(results)} OK, {len(failures)} FAILED.", flush=True)
    print("NaN/inf and warnings are reported, not automatically classified as failures.")
    return results, failures


# Simple subject example, intentionally without a main wrapper.
tsdata = loadmat(input_file, variable_names=[data_key])[data_key]
if not 0 <= isubj < len(tsdata):
    raise IndexError(f"Subject index {isubj} is outside the dataset.")
print(f"Subject index={isubj}, avdeg={avdeg:.6g}, window={window}, lag={lag}, hops=1")
print(f"Using toolkit: {Path(distance.__file__).resolve().parent}")
tnet = BinarizeDFC(tsdata[isubj], avdeg, window, lag, "original")
results, failures = RunMeasureChecks(tnet, rand_seed, nreps, max_return_latency)
