import numpy as np

from tnet_analysis.dfc_to_tnet import BinarizeDFC
from tnet_analysis.network_generators import GenerateNullModel
from tnet_analysis.latency_measures import (
    ComputeSmartTemporalDistanceMeasures,
    ComputeDrunkTemporalDistanceMeasures,
    ComputeCirculationLatency,
)
from tnet_analysis.dynamism_and_memory import (
    ComputeDynamism,
    ComputeReturnability,
)
from tnet_analysis.segregation_and_cohesion import (
    ComputeStaticClustering,
    ComputeTemporalClustering,
)


def func0(x, hops_per_frame=1, latency_unit="frames"):
    return ComputeSmartTemporalDistanceMeasures(
        x, summary_stat="both", hops_per_frame=hops_per_frame,
        latency_unit=latency_unit,
    )


def func1(x, n_reps=None, random_seed=None, hops_per_frame=1,
          latency_unit="frames"):
    if n_reps is None:
        n_reps = nreps
    return ComputeDrunkTemporalDistanceMeasures(
        x,
        n_reps=n_reps,
        random_seed=random_seed,
        summary_stat="both",
        hops_per_frame=hops_per_frame,
        latency_unit=latency_unit,
    )


def func2(x):
    return ComputeDynamism(x)


def func3(x):
    return ComputeStaticClustering(x, summary_stat="both")


def func4(x):
    return ComputeTemporalClustering(x, summary_stat="both")


def func5(x, hops_per_frame=1, latency_unit="frames"):
    summary_circulation, mean_count = ComputeCirculationLatency(
        x, summary_stat="both", hops_per_frame=hops_per_frame,
        latency_unit=latency_unit,
    )
    return summary_circulation + [mean_count]


def func6(x):
    return ComputeReturnability(x, summary_stat="both")


def EvaluateMetric(ifunc, tnet, tnet_nodiag, random_seed=None, n_reps=None,
                   hops_per_frame=1, latency_unit="frames"):
    """Evaluate and validate one metric specification."""
    name, output_names, func, use_nodiag, use_window = METRIC_SPECS[ifunc]
    x = tnet_nodiag if use_nodiag else tnet

    if ifunc == 1:
        values = func(x, n_reps=n_reps, random_seed=random_seed,
                      hops_per_frame=hops_per_frame, latency_unit=latency_unit)
    elif ifunc in (0, 5):
        values = func(x, hops_per_frame=hops_per_frame,
                      latency_unit=latency_unit)
    elif use_window:
        values = func(x, window)
    else:
        values = func(x)

    values = np.asarray(values, dtype=float).reshape(-1)
    expected_width = len(output_names)
    if values.size != expected_width:
        raise ValueError(
            f"{func.__name__} returned {values.size} values, "
            f"expected {expected_width}."
        )
    return ifunc, name, values



def PadMetricValues(ifunc, values, fill_value=0.0):
    """Pad one meaningful metric vector to the fixed parallel-task width."""
    values = np.asarray(values, dtype=float).reshape(-1)
    expected_width = len(METRIC_SPECS[ifunc][1])
    if values.size != expected_width:
        func = METRIC_SPECS[ifunc][2]
        raise ValueError(
            f"{func.__name__} returned {values.size} values, "
            f"expected {expected_width}."
        )
    out = np.full(TASK_OUTPUT_WIDTH, fill_value, dtype=float)
    out[:expected_width] = values
    return out


def qIntSeg(tsdata, isubj, ideg, itag, ifunc, hops_per_frame=1,
            latency_unit="frames"):
    """Compute one validated metric for a subject, degree, and tag."""
    tseries = tsdata[isubj]
    tdeg = degList[ideg]
    tag = tags[itag]

    otnet = BinarizeDFC(tseries, tdeg, window, lag, tag)
    n_frames, n_nodes, _ = otnet.shape
    n_off_diagonal_edges = np.sum(otnet) - n_frames * n_nodes
    width = len(METRIC_SPECS[ifunc][1])
    if n_off_diagonal_edges == 0:
        name = METRIC_SPECS[ifunc][0]
        values = np.zeros(TASK_OUTPUT_WIDTH, dtype=float)
        values[:width] = np.nan
        return ideg, itag, ifunc, name, values

    random_seed = rand_seed + 100000 * isubj + 1000 * ideg + 100 * itag
    ntnet, _ = GenerateNullModel(otnet, tag, seed=random_seed)
    ntnet_nodiag = np.array(ntnet, copy=True)
    diagonal = np.arange(ntnet.shape[1])
    ntnet_nodiag[:, diagonal, diagonal] = 0

    _, name, values = EvaluateMetric(
        ifunc,
        ntnet,
        ntnet_nodiag,
        random_seed=random_seed,
        hops_per_frame=hops_per_frame,
        latency_unit=latency_unit,
    )
    return ideg, itag, ifunc, name, PadMetricValues(ifunc, values)


outdir = "data/output_multihop"
indir = "/Users/sima/Documents/MATLAB/empirical"

degList = np.logspace(np.log10(0.1), np.log10(83.04), num=54)[:10]

tags = [
    "original",
    # "emp_static"
]
hopList = [2, 5, 10]
nHops = len(hopList)
latency_unit = "frames"

window = 20
lag = 1
nreps = 100
rand_seed = 85869

def PairNames(prefix):
    return (f"{prefix}_mean", f"{prefix}_median")


# (name, output_names, function, use_nodiag, use_window)
METRIC_SPECS = [
    (
        "smart",
        PairNames("smart_impermeability")
        + PairNames("smart_latency")
        + PairNames("smart_resistance")
        + ("smart_inaccessibility", "smart_irrigation"),
        func0,
        False,
        False,
    ),
    (
        "drunk",
        PairNames("drunk_impermeability")
        + PairNames("drunk_latency")
        + PairNames("drunk_resistance")
        + ("drunk_inaccessibility", "drunk_irrigation"),
        func1,
        False,
        False,
    ),
    (
        "dynamism",
        (
            "transition_probability",
            "global_entropy",
            "cosine_similarity",
            "net_fluidity",
            "mutual_information",
            "mean_edge_count",
        ),
        func2,
        True,
        False,
    ),
    (
        "static_clustering",
        PairNames("static_clustering"),
        func3,
        True,
        False,
    ),
    (
        "temporal_clustering",
        PairNames("temporal_clustering"),
        func4,
        True,
        False,
    ),
    (
        "circulation",
        PairNames("circulation_latency")
        + ("circulation_mean_count",),
        func5,
        False,
        False,
    ),
    (
        "returnability",
        PairNames("returnability"),
        func6,
        False,
        False,
    ),
]
HOP_DEPENDENT_METRICS = {0, 1, 5}

funcs = [spec[2] for spec in METRIC_SPECS]
funcnames = [spec[0] for spec in METRIC_SPECS]
nFunc = len(METRIC_SPECS)
TASK_OUTPUT_WIDTH = 8
nOut = TASK_OUTPUT_WIDTH
metricNames = [
    list(output_names)
    + [
        f"{name}_padding_{slot:02d}"
        for slot in range(len(output_names), TASK_OUTPUT_WIDTH)
    ]
    for name, output_names, _, _, _ in METRIC_SPECS
]

nSubj = 100
nDeg = len(degList)
nTag = len(tags)
