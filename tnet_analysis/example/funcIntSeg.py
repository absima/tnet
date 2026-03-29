import numpy as np
import scipy.io

from tnet_analysis.dfc_to_tnet import BinarizeDFC
from tnet_analysis.network_generators import GenerateNullModel
from tnet_analysis.latency_measures import (
    ComputeSmartTemporalDistanceMeasures,
    ComputeDrunkTemporalDistanceMeasures,
    ComputeCirculationLatency,
    ComputeAverageDegree,
)
from tnet_analysis.dynamism_and_memory import (
    ComputeDynamism,
    ComputeNeighborhoodMemory,
    ComputeReturnability,
)
from tnet_analysis.segregation_and_cohesion import (
    ComputeNodePersistence,
    ComputePartnerStability,
    ComputePartnerDiversity,
    ComputeStaticClustering,
    ComputeTemporalClustering,
)


def func0(x):
    return ComputeSmartTemporalDistanceMeasures(x)

def func1(x):
    return ComputeDrunkTemporalDistanceMeasures(x, n_reps=nreps)

def func2(x):
    return ComputeDynamism(x)

def func3(x):
    summary_persistence, n_significant = ComputeNodePersistence(
        x, 0.95, summary_stat=summary_stat
    )
    return [summary_persistence, n_significant]

def func4(x):
    return [ComputePartnerStability(x, summary_stat=summary_stat)]

def func5(x):
    return [ComputePartnerDiversity(x, summary_stat=summary_stat)]

def func6(x, window):
    return [ComputeNeighborhoodMemory(x, lag=1, summary_stat=summary_stat)]

def func7(x, window):
    return [ComputeNeighborhoodMemory(x, lag=max(1, window // 2), summary_stat=summary_stat)]

def func8(x, window):
    return [ComputeNeighborhoodMemory(x, lag=max(1, window), summary_stat=summary_stat)]

def func9(x):
    return [ComputeStaticClustering(x, summary_stat=summary_stat)]

def func10(x):
    return [ComputeTemporalClustering(x, summary_stat=summary_stat)]

def func11(x):
    summary_circulation, mean_count = ComputeCirculationLatency(
        x, summary_stat=summary_stat
    )
    return [summary_circulation, mean_count]

def func12(x):
    return [ComputeReturnability(x, summary_stat=summary_stat)]





def _pack_metrics(tnet, tnet_nodiag, window):
    """Pack all metric outputs into one flat vector."""
    out = np.full(nOut, np.nan, dtype=float)
    offset = 0

    for _, width, func, use_nodiag, use_window in METRIC_SPECS:
        x = tnet_nodiag if use_nodiag else tnet
        values = func(x, window) if use_window else func(x)
        values = np.asarray(values, dtype=float)
        if values.size != width:
            raise ValueError(
                f"{func.__name__} returned {values.size} values, expected {width}."
            )
        out[offset:offset + width] = values
        offset += width

    return out


def eval_subject_deg_tag(isubj, ideg, itag):
    tseries = tsdata[isubj]
    tdeg = degList[ideg]
    tag = tags[itag]

    otnet = BinarizeDFC(tseries, tdeg, window, lag)
    T, N, _ = otnet.shape
    numedges = np.sum(otnet) - T * N

    out_block = np.full((nProb, nOut), np.nan, dtype=float)
    if numedges == 0:
        return ideg, itag, out_block

    for iprob, prob in enumerate(probs):
        seedi = rand_seed + 100000 * isubj + 1000 * ideg + 100 * itag + iprob
        ntnet, _ = GenerateNullModel(otnet, tag=tag, p_rewire=prob, seed=seedi)

        ntnet_nodiag = np.array(ntnet, copy=True)
        diag = np.arange(N)
        ntnet_nodiag[:, diag, diag] = 0

        out_block[iprob] = _pack_metrics(ntnet, ntnet_nodiag, window)

    return ideg, itag, out_block



outdir = "data/output"
indir = "data/input"
indir = '/Users/sima/Documents/MATLAB/empirical'

tsdata = scipy.io.loadmat(f"{indir}/ts_100subjs.mat")["tseries"][:,:200,:]

degList = np.logspace(np.log10(0.1), np.log10(83.04), num=54)
tags = ["original"]
probs = [0.05]

rand_seed = 85869
window = 20
lag = 1
nreps = 100
summary_stat = "median"

# Metric schema:
# (name, width, func, use_nodiag, use_window)
# `use_nodiag=True` means run on the diagonal-zeroed temporal network.
# `use_window=True` means call the metric as func(x, window) instead of func(x).
METRIC_SPECS = [
    ("smart", 5, func0, False, False),
    ("drunk", 5, func1, False, False),
    ("dynamism", 7, func2, True, False),
    ("persistence", 2, func3, True, False),
    ("partner_stability", 1, func4, True, False),
    ("partner_diversity", 1, func5, True, False),
    ("memory_lag1", 1, func6, True, True),
    ("memory_half_window", 1, func7, True, True),
    ("memory_window", 1, func8, True, True),
    ("static_clustering", 1, func9, True, False),
    ("temporal_clustering", 1, func10, True, False),
    ("circulation", 2, func11, False, False),
    ("returnability", 1, func12, False, False),
]

funcs = [spec[2] for spec in METRIC_SPECS]
nMetric = len(METRIC_SPECS)
nOut = sum(spec[1] for spec in METRIC_SPECS)

metric_slices = {}
offset = 0
for name, width, _, _, _ in METRIC_SPECS:
    metric_slices[name] = slice(offset, offset + width)
    offset += width

nSubj = 100
nDeg = len(degList)
nTag = len(tags)
nFunc = len(funcs)
nProb = len(probs)
