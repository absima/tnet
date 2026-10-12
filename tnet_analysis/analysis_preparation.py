import numpy as np

from .dfc_to_tnet import BinarizeDFCFromConnectivity, ComputeFunctionalConnectivity
from .network_generators import GenerateNullModel
from .signal_surrogates import PhaseRandSurrogates


def ResolveAnalysisTag(tag):
    """
    Route a supported analysis tag to its signal and network processing stages.

    Args:
        tag: str. Empirical, signal-surrogate, or network-null selection.

    Returns:
        specification: tuple. (phase_mode, binarization_tag, network_tag, seed_id).
            phase_mode is None, 0, or 1. Stable seed IDs do not depend on tag order.
    """
    specifications = {
        "original": (None, "original", "original", 0),
        "emp_static": (None, "emp_static", "emp_static", 1),
        "time": (None, "original", "time", 2),
        "uniform": (None, "original", "uniform", 3),
        "phase_rand0": (0, "original", "original", 4),
        "phase_rand1": (1, "original", "original", 5),
        "er_static": (None, "original", "er_static", 6),
        "sw_static": (None, "original", "sw_static", 7),
        "sf_static": (None, "original", "sf_static", 8),
        "eAct": (None, "original", "eAct", 9),
        "eActTemplate": (None, "original", "eActTemplate", 10),
        "eRndTemplate": (None, "original", "eRndTemplate", 11),
        "erand_er": (None, "original", "erand_er", 12),
        "erand_sws": (None, "original", "erand_sws", 13),
        "erand_sfs": (None, "original", "erand_sfs", 14),
        "erand_swr": (None, "original", "erand_swr", 15),
        "erand_sfr": (None, "original", "erand_sfr", 16),
    }
    if tag in ("original1", "original2"):
        raise ValueError(f"{tag!r} is an FC method, not a null-model tag. Set fc_method = {tag!r}.")
    if not isinstance(tag, str) or tag not in specifications:
        raise ValueError(f"Unsupported analysis tag {tag!r}. Available tags: {', '.join(specifications)}")
    return specifications[tag]


def PrepareSignal(tseries, tag, random_seed=None):
    """
    Prepare a signal-based surrogate, or retain the original signals.

    Args:
        tseries: np.ndarray, shape (time, regions). Original subject signals.
        tag: str. Supported analysis tag.
        random_seed: int or None, optional (default=None). Signal-surrogate seed.

    Returns:
        signals: np.ndarray. Original or phase-randomized signals.
        network_tag: str. Subsequent network-null selection.
    """
    phase_mode, _, network_tag, _ = ResolveAnalysisTag(tag)
    if phase_mode is not None:
        return PhaseRandSurrogates(tseries, phase_mode, random_seed), network_tag
    return tseries, network_tag


def PrepareConnectivity(tseries, tag="original", window=20, lag=1,
                        fc_method="original1", random_seed=None):
    """
    Prepare the selected signals and compute Pearson or pointwise connectivity.

    Args:
        tseries: np.ndarray, shape (time, regions). Original subject signals.
        tag: str, optional (default='original'). Supported analysis selection.
        window: int, optional (default=20). Ignored for original2.
        lag: int, optional (default=1). Ignored for original2.
        fc_method: {'original1', 'original2'}, optional (default='original1'). FC method.
        random_seed: int or None, optional (default=None). Signal-surrogate seed.

    Returns:
        fcs: np.ndarray, shape (frames, regions, regions). Selected connectivity.
    """
    signals, _ = PrepareSignal(tseries, tag, random_seed)
    return ComputeFunctionalConnectivity(signals, window, lag, method=fc_method)


def PrepareTemporalNetwork(fcs, tag, target_degree, threshold=None, seed=None,
                           dtype=np.uint8):
    """
    Binarize prepared connectivity and apply only the selected network-null step.

    Args:
        fcs: np.ndarray, shape (frames, regions, regions). Already prepared FC;
            for phase tags this must have been computed from the surrogate signals.
        tag: str. Supported analysis selection.
        target_degree: float. Target average degree.
        threshold: float or None, optional (default=None). Cached temporal threshold.
        seed: int or None, optional (default=None). Network-null seed.
        dtype: numpy dtype, optional (default=np.uint8). Output adjacency dtype.

    Returns:
        tnet: np.ndarray. Binarized temporal network with waiting diagonals.
    """
    _, binarization_tag, network_tag, _ = ResolveAnalysisTag(tag)
    tnet = BinarizeDFCFromConnectivity(fcs, target_degree, binarization_tag,
                                     threshold=threshold, dtype=dtype)
    T, N, _ = tnet.shape
    if np.sum(tnet) - T * N == 0:
        return tnet
    if seed is not None:
        # Network generators use NumPy's global RNG; Loky workers are isolated.
        np.random.seed(seed)
    tnet, _ = GenerateNullModel(tnet, network_tag, seed=seed)
    return np.asarray(tnet, dtype=dtype)
