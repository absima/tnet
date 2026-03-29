import numpy as np


def _summarize_scores(values, summary_stat="median"):
    """Reduce a 1D score array to a median or mean."""
    if summary_stat == "median":
        return float(np.nanmedian(values))
    if summary_stat == "mean":
        return float(np.nanmean(values))
    raise ValueError("summary_stat must be 'median' or 'mean'.")


def ComputeCirculationLatency(
    adjacency_matrices,
    max_latency=None,
    return_full=False,
    summary_stat="median",
):
    """
    Compute node-wise circulation latency in a temporal network (vectorized).

    Args:
        adjacency_matrices: array, shape (T, N, N). Sequence of adjacency matrices (binary, 0/1), with waiting (diagonal ones).
        max_latency: int, optional. Maximum number of steps to search. If None, set to T.
        return_full: bool, optional. If True, return nodewise details and summary scalars.
        summary_stat: `median` or `mean` for reducing nodewise latency.

    Returns:
        If `return_full` is False, `(summary_latency, mean_count_hits)`. If `return_full` is True, a dict with nodewise arrays plus global mean, std, median, and requested summary latency.
    """

    T, N, _ = adjacency_matrices.shape
    if max_latency is None:
        max_latency = T

    node_activity = (
        adjacency_matrices.sum(axis=(0, 1))
        + adjacency_matrices.sum(axis=(0, 2))
    )
    active_nodes = np.where(node_activity > 0)[0]
    n_active = len(active_nodes)
    if n_active == 0:
        raise ValueError("No active nodes in the temporal network.")

    adj = (adjacency_matrices[:, active_nodes][:, :, active_nodes] > 0).astype(np.int64)
    for t in range(T):
        np.fill_diagonal(adj[t], 1)

    min_latency = np.full(n_active, np.inf, dtype=float)
    sum_latency = np.zeros(n_active, dtype=float)
    count_hits = np.zeros(n_active, dtype=np.int64)

    for start_time in range(T - 1):
        A0 = adj[start_time]

        deg_mask = (A0.sum(axis=1) > 1)
        if not np.any(deg_mask):
            continue

        path = A0.copy()
        prev_diag = np.diag(path).copy()
        recorded = np.zeros(n_active, dtype=bool)

        max_L = min(max_latency, T - start_time)
        for latency in range(1, max_L):
            path = path @ adj[start_time + latency]
            new_diag = np.diag(path)

            hit = (new_diag > prev_diag) & (~recorded) & deg_mask
            if np.any(hit):
                lat_val = latency + 1
                sum_latency[hit] += lat_val
                count_hits[hit] += 1
                np.minimum(min_latency[hit], lat_val, out=min_latency[hit])
                recorded[hit] = True

            if recorded[deg_mask].all():
                break

            prev_diag = new_diag

    nodewise_mean_latency = np.full(n_active, np.nan, dtype=float)
    nonzero = count_hits > 0
    nodewise_mean_latency[nonzero] = sum_latency[nonzero] / count_hits[nonzero]

    global_mean_latency = np.nanmean(nodewise_mean_latency)
    global_std_latency = np.nanstd(nodewise_mean_latency)
    global_median_latency = np.nanmedian(nodewise_mean_latency)
    global_summary_latency = _summarize_scores(
        nodewise_mean_latency, summary_stat=summary_stat
    )

    if return_full:
        min_latency_out = min_latency.copy()
        min_latency_out[~nonzero] = np.nan
        return {
            'nodewise_mean_latency': nodewise_mean_latency,
            'min_latency': min_latency_out,
            'hit_count': count_hits,
            'global_mean_latency': float(global_mean_latency),
            'global_std_latency': float(global_std_latency),
            'global_median_latency': float(global_median_latency),
            'global_summary_latency': float(global_summary_latency),
            'summary_stat': summary_stat,
            'active_nodes': active_nodes
        }
    else:
        return float(global_summary_latency), float(np.mean(count_hits))
