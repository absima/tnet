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


# =========================================================
# Threshold search for target average degree
# =========================================================


def FindThresholdForAverageDegree(adj_tensor, target_degree, resolution=100):
    """
    Find the correlation threshold that produces a desired average degree.

    Args:
        adj_tensor: np.ndarray, shape (T, N, N). Temporal correlation matrices (e.g., DFC). Symmetric per snapshot.
        target_degree: float. Desired average degree to achieve after thresholding.
        resolution: int, optional (default=100). Number of threshold candidates between min and max to evaluate.

    Returns:
        best_threshold: float. Threshold value that yields an average degree closest to target_degree.
    """
    T, N, _ = adj_tensor.shape
    adj_no_diag = adj_tensor.copy()
    for t in range(T):
        np.fill_diagonal(adj_no_diag[t], 0)

    min_val = adj_no_diag.min()
    max_val = adj_no_diag.max()

    thresholds = np.linspace(min_val, max_val, resolution)
    best_thresh = None
    best_diff = float("inf")

    for thresh in thresholds:
        avg_deg, _ = ComputeAverageDegree(adj_tensor, thresh)
        diff = abs(avg_deg - target_degree)

        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh

    return best_thresh


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
    fcs0 = SetDiagonals(fcs, 0)

    if tag == 'emp_static':
        static_fc = np.mean(fcs0, axis=0)

        N = static_fc.shape[0]
        iu = np.triu_indices(N, k=1)
        vals = np.abs(static_fc[iu])

        n_edges_target = int(round(target_degree * N / 2.0))
        n_edges_target = max(0, min(n_edges_target, len(vals)))

        static_bin = np.zeros_like(static_fc, dtype=int)

        if n_edges_target > 0:
            sorted_vals = np.sort(vals)[::-1]
            threshold = sorted_vals[n_edges_target - 1]

            mask = (np.abs(static_fc) >= threshold).astype(int)

            mask = np.triu(mask, 1)
            static_bin = mask + mask.T

        np.fill_diagonal(static_bin, 1)

        tnet = np.repeat(static_bin[np.newaxis, :, :], fcs.shape[0], axis=0)
    else:
        threshold = FindThresholdForAverageDegree(fcs0, target_degree)
        tnet0 = (fcs0 > threshold).astype(int)
        tnet = SetDiagonals(tnet0, 1)

    return tnet
