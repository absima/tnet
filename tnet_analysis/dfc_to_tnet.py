import numpy as np

from .latency_measures import SetDiagonals, ComputeAverageDegree

# =========================================================
# Dynamic functional connectivity (DFC) estimation
# =========================================================


def ComputeDynamicFunctionalConnectivity(time_series, window, lag):
    """
    Compute dynamic functional connectivity (DFC) matrices from time series.

    Args:
        time_series: np.ndarray, shape (T, N). Input time series with T time points and N regions/nodes.
        window: int. Length of the sliding time window.
        lag: int. Step size (lag) between successive windows.

    Returns:
        fcs: np.ndarray, shape (n_snapshots, N, N). Sequence of functional connectivity matrices computed as Pearson correlation coefficients within each window.
    """
    t_length, n_nodes = time_series.shape
    n_snapshots = int((t_length - window) / lag)

    fcs = np.zeros((n_snapshots, n_nodes, n_nodes))
    for i in range(n_snapshots):
        shift = i * lag
        tsi = time_series[shift : shift + window]
        cc = np.corrcoef(tsi.T)
        fcs[i] = cc

    return fcs


def ComputeInstantaneousFunctionalConnectivity(time_series, ddof=0):
    """
    Compute pointwise connectivity as products of region-wise z-scored signals.

    Args:
        time_series: np.ndarray, shape (T, N). Finite, real-valued signals with
            T time points and N regions/nodes. Each region is standardized over time.
        ddof: {0, 1}, optional (default=0). Standard-deviation convention:
            population (0), or sample (1, as in MATLAB's default zscore).

    Returns:
        fcs: np.ndarray, shape (T, N, N). One frame per input time point,
            with fcs[t, i, j] = z[t, i] * z[t, j]. No window or lag is used.

    Notes:
        - With ddof=0, the time-average equals whole-record Pearson correlation.
          With ddof=1, multiply the time-average by T / (T - 1) to obtain it.
        - Products are signed and not bounded by [-1, 1]. Diagonals are z[t, i]**2,
          not forced to one; binarization handles self-edges separately.
        - Constant regions are rejected because their z-scores are undefined.
    """
    signals = np.asarray(time_series)
    if signals.ndim != 2 or signals.shape[0] < 2 or signals.shape[1] < 1:
        raise ValueError("time_series must have shape (T, N) with T >= 2 and N >= 1.")
    if np.iscomplexobj(signals) or not np.isfinite(signals).all():
        raise ValueError("time_series must contain finite, real-valued signals.")
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof not in (0, 1):
        raise ValueError("ddof must be 0 (population) or 1 (sample).")

    signals = np.asarray(signals, dtype=float)
    scales = signals.std(axis=0, ddof=ddof)
    if np.any(scales == 0):
        raise ValueError("Cannot z-score constant region time series.")
    zscores = signals - signals.mean(axis=0)
    zscores /= scales
    # Allocate only the final connectivity tensor, without an outer-product temporary.
    return np.einsum("ti,tj->tij", zscores, zscores)


def ComputeFunctionalConnectivity(time_series, window=20, lag=1, method="original1", ddof=0):
    """
    Select sliding-window Pearson or instantaneous z-score-product connectivity.

    Args:
        time_series: np.ndarray, shape (T, N). Input time series.
        window: int, optional (default=20). Pearson sliding-window length;
            ignored for 'original2'.
        lag: int, optional (default=1). Pearson window step; ignored for 'original2'.
        method: {'original1', 'original2'}, optional (default='original1').
            'original1' uses Pearson correlations; 'original2' uses z-score products.
            The method name 'original' is also accepted as an alias for 'original1'.
        ddof: {0, 1}, optional (default=0). Z-score standardization convention;
            used only for 'original2'.

    Returns:
        fcs: np.ndarray. Shape (n_snapshots, N, N) for Pearson, or (T, N, N)
            for instantaneous products.
    """
    if method in ("original1", "original"):
        return ComputeDynamicFunctionalConnectivity(time_series, window, lag)
    if method == "original2":
        return ComputeInstantaneousFunctionalConnectivity(time_series, ddof=ddof)
    raise ValueError("method must be 'original1' (Pearson) or 'original2' (z-score products).")


# =========================================================
# Threshold search for target average degree
# =========================================================


def FindThresholdsForAverageDegrees(adj_tensor, target_degrees, resolution=100):
    """
    Find thresholds for several average degrees using one sorted connectivity tensor.

    Args:
        adj_tensor: np.ndarray, shape (T, N, N). Temporal connectivity matrices.
        target_degrees: array-like. Desired average degrees.
        resolution: int, optional (default=100). Number of threshold candidates.

    Returns:
        thresholds: np.ndarray. Selected thresholds in the target-degree order.
    """
    T, N, _ = adj_tensor.shape
    if not np.isfinite(adj_tensor).all():
        raise ValueError("Connectivity contains non-finite values; check for constant or missing region time series.")
    adj_no_diag = SetDiagonals(adj_tensor, 0)
    candidates = np.linspace(adj_no_diag.min(), adj_no_diag.max(), resolution)
    values = np.sort(adj_no_diag, axis=None)
    edge_counts = values.size - np.searchsorted(values, candidates, side="right")
    # Zeroed diagonals pass negative thresholds but do not count as edges.
    edge_counts -= (candidates < 0) * (T * N)
    achieved_degrees = edge_counts / float(T * N)
    targets = np.asarray(target_degrees, dtype=float).reshape(-1)
    if not np.isfinite(targets).all():
        raise ValueError("target_degrees must contain finite values.")
    best = np.abs(achieved_degrees[:, None] - targets[None, :]).argmin(axis=0)
    return candidates[best]


def FindThresholdForAverageDegree(adj_tensor, target_degree, resolution=100):
    """
    Find the correlation threshold that produces a desired average degree.

    Args:
        adj_tensor: np.ndarray, shape (T, N, N). Temporal connectivity matrices (e.g., correlations or z-score products). Symmetric per snapshot.
        target_degree: float. Desired average degree to achieve after thresholding.
        resolution: int, optional (default=100). Number of threshold candidates between min and max to evaluate.

    Returns:
        best_threshold: float. Threshold value that yields an average degree closest to target_degree.
    """
    return float(FindThresholdsForAverageDegrees(
        adj_tensor, [target_degree], resolution
    )[0])


# =========================================================
# Pipeline: Binarize DFC given subject & target degree
# =========================================================


def BinarizeDFC(time_series, target_degree, window, lag, tag='original'):
    """
    Compute and binarize dynamic functional connectivity.

    Args:
        time_series: np.ndarray, shape (T, N). Input time series with T time points and N regions/nodes.
        target_degree: float. Desired average degree per snapshot.
        window: int. Length of sliding window.
        lag: int. Step size between windows.
        tag: str, optional. Binarization mode: `original` for per-snapshot
            thresholding, or `emp_static` for one static FC repeated across time.

    Returns:
        tnet: np.ndarray, shape (n_snapshots, N, N). Binarized temporal adjacency matrices.
    """
    fcs = ComputeDynamicFunctionalConnectivity(time_series, window, lag)
    return BinarizeDFCFromConnectivity(fcs, target_degree, tag)


def BinarizeDFCFromConnectivity(fcs, target_degree, tag='original', threshold=None, dtype=int):
    """
    Binarize existing connectivity without recomputing sliding-window correlations.

    Args:
        fcs: np.ndarray, shape (T, N, N). Temporal connectivity matrices (correlations or z-score products).
        target_degree: float. Desired average degree.
        tag: str, optional (default='original'). Use 'emp_static' for static FC repeated across frames.
        threshold: float or None, optional (default=None). Precomputed temporal threshold; ignored for 'emp_static'.
        dtype: numpy dtype, optional (default=int). Output adjacency dtype; uint8 avoids large integer tensors.

    Returns:
        tnet: np.ndarray, shape (T, N, N). Binary temporal adjacency with diagonal ones.
    """
    if tag == 'emp_static':
        static_fc = np.mean(fcs, axis=0)

        N = static_fc.shape[0]
        iu = np.triu_indices(N, k=1)
        vals = np.abs(static_fc[iu])

        n_edges_target = int(round(target_degree * N / 2.0))
        n_edges_target = max(0, min(n_edges_target, len(vals)))

        static_bin = np.zeros_like(static_fc, dtype=dtype)

        if n_edges_target > 0:
            sorted_vals = np.sort(vals)[::-1]
            threshold = sorted_vals[n_edges_target - 1]

            mask = (np.abs(static_fc) >= threshold).astype(dtype)

            mask = np.triu(mask, 1)
            static_bin = mask + mask.T

        np.fill_diagonal(static_bin, 1)

        tnet = np.repeat(static_bin[np.newaxis, :, :], fcs.shape[0], axis=0)
    else:
        if threshold is None:
            threshold = FindThresholdForAverageDegree(fcs, target_degree)
        tnet = (fcs > threshold).astype(dtype)
        diagonal = np.arange(tnet.shape[1])
        tnet[:, diagonal, diagonal] = 1

    return tnet
