import numpy as np
import warnings


def _summarize_scores(values, summary_stat="median"):
    """Reduce a 1D score array to a median or mean."""
    if summary_stat == "median":
        return float(np.nanmedian(values))
    if summary_stat == "mean":
        return float(np.nanmean(values))
    raise ValueError("summary_stat must be 'median' or 'mean'.")


def SetDiagonals(tensor, value=1):
    """
    Set the diagonals of all matrices in a temporal or static adjacency tensor.

    Args:
        tensor: np.ndarray, shape (T, N, N) or (N, N). Input adjacency tensor or matrix. If 2D, it is treated as a single snapshot.
        value: int or float, optional (default=1). Value to assign along the diagonal.

    Returns:
        tensor_out: np.ndarray. The modified tensor with diagonals set to `value`.
    """
    arr = tensor.copy()
    if arr.ndim == 2:  # single snapshot case
        np.fill_diagonal(arr, value)
    elif arr.ndim == 3:  # temporal case
        T, n_nodes, _ = arr.shape
        idx = np.arange(n_nodes)
        arr[:, idx, idx] = value
    else:
        raise ValueError("Input must be 2D or 3D array.")
    return arr


def ComputeAverageDegree(adj_tensor, threshold=None):
    """
    Compute the average and standard deviation of degrees over a temporal network.

    Args:
        adj_tensor: np.ndarray, shape (T, N, N). Temporal adjacency (binary or weighted). Assumed symmetric per snapshot.
        threshold: float or None, optional (default=None). If provided, binarize as (A > threshold). If None, use the input as-is (cast to int).

    Returns:
        mean_deg: float. Mean degree averaged over all nodes and time snapshots.
        std_deg: float. Standard deviation of degrees over all nodes and time snapshots.

    Notes:
        - Diagonals are set to 0 before degree computation.
        - Degree per snapshot is computed as row-sums of the binarized adjacency.
    """
    A = adj_tensor.copy()
    if threshold is None:
        binarized = (A != 0).astype(int)
    else:
        binarized = (A > threshold).astype(int)

    T, n_nodes, _ = binarized.shape
    idx = np.arange(n_nodes)
    binarized[:, idx, idx] = 0

    degrees = binarized.sum(axis=2)
    mean_deg = degrees.mean()
    std_deg = degrees.std()
    return mean_deg, std_deg


def SmartWalker(tnet):
    """
    Compute earliest-arrival temporal distances using a 'smart walker'.
    The algorithm aggregates temporal reachability cumulatively across time.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), assumed symmetric per snapshot. Diagonals are forced to 1 internally (self-reachability).

    Returns:
        dmtx: np.ndarray, shape (N, N). Earliest arrival time (in snapshot steps) from i to j. np.nan on diagonal; np.inf if j is never reached from i.
    """
    tnet_copy = tnet.copy()
    np.einsum('tii->ti', tnet_copy)[:] = 1

    n = tnet_copy.shape[1]
    dmtx = np.full((n, n), np.inf)
    np.fill_diagonal(dmtx, 0)

    reach = np.eye(n, dtype=int)
    for t, net in enumerate(tnet_copy):
        reach = (reach @ net > 0).astype(int)
        cprod = reach.copy()
        np.fill_diagonal(cprod, 0)
        time_layer = np.where(cprod > 0, t + 1, np.inf)
        dmtx = np.minimum(dmtx, time_layer)

    np.fill_diagonal(dmtx, np.nan)
    return dmtx


def RandomWalker(tnet, n_trials, return_fpt_matrix=False, seed=None):
    """
    Estimate temporal distances via Monte Carlo random walks.
    For each trial and source node, simulate a walk over snapshots and record
    first-passage times.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), assumed symmetric per snapshot. Diagonals are forced to 1 internally (self-reachability).
        n_trials: int. Number of Monte Carlo trials.
        return_fpt_matrix: bool, optional (default=False). If True, also return the per-trial FPT tensors.
        seed: int or None, optional (default=None). Seed for reproducibility.

    Returns:
        dmtx: np.ndarray, shape (N, N). Mean first passage time across trials, ignoring np.inf entries. np.nan on diagonal; np.inf if a node is never reached across all trials.
        mean_q6: list[float]. Mean of the summary statistics across trials.
        all_fpt: np.ndarray, optional, shape (n_trials, N, N). Returned only if `return_fpt_matrix=True`.
    """
    rng = np.random.default_rng(seed)

    tnet_copy = tnet.copy()
    np.einsum('tii->ti', tnet_copy)[:] = 1

    T, n_nodes, _ = tnet_copy.shape
    all_dmtx = np.zeros((n_trials, n_nodes, n_nodes))
    q6_array = np.zeros((n_trials, 6))

    for trial in range(n_trials):
        dmtx_trial = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(dmtx_trial, 0)

        for source in range(n_nodes):
            where = source
            first_hit = np.full(n_nodes, np.inf)
            first_hit[source] = np.nan
            visited = {source}

            for t in range(T):
                adj = tnet_copy[t]
                neighbors = np.where(adj[where] > 0)[0]

                if len(neighbors) == 0:
                    break

                next_node = rng.choice(neighbors)

                if next_node not in visited:
                    first_hit[next_node] = t + 1
                    visited.add(next_node)

                where = next_node
                if len(visited) == n_nodes:
                    break

            dmtx_trial[source] = first_hit

        all_dmtx[trial] = dmtx_trial
        q6_array[trial] = MeanLatencyMatrixAnalysis(dmtx_trial, T + 1)

    mean_q6 = q6_array.mean(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dmtx = np.mean(all_dmtx, axis=0, where=~np.isinf(all_dmtx))
        all_inf_mask = np.all(np.isinf(all_dmtx), axis=0)
        dmtx[all_inf_mask] = np.inf

    np.fill_diagonal(dmtx, np.nan)

    if return_fpt_matrix:
        return dmtx, mean_q6.tolist(), all_dmtx
    else:
        return dmtx, mean_q6.tolist()


def MeanLatencyMatrixAnalysis(dmx, penalty):
    """
    Summarize a distance/first-passage matrix with five quantities.
    Given a matrix `dmx` where diagonal is NaN, finite off-diagonals are distances, and +inf denotes unreachable pairs, compute:
    1) impermeability: Mean of distance normalized by source irrigation, averaged over sources. 2) latency: Mean of all finite distances. 3) resistance: Mean distance after replacing `+inf` with `penalty`. 4) inaccessibility: Count of `+inf` entries. 5) irrigation: Count of finite entries.

    Args:
        dmx: np.ndarray, shape (N, N). Distance or first-passage matrix with NaN on the diagonal and possibly +inf.
        penalty: float. Replacement value used to compute penalized mean distance.

    Returns:
        stats: list[float]. [impermeability, latency, resistance, inaccessibility, irrigation]
    """
    dmx = dmx.copy()
    np.fill_diagonal(dmx, np.nan)

    dmtx = np.where(np.isinf(dmx), np.nan, dmx)
    latency = np.nanmean(dmtx)

    finite_mask = ~np.isnan(dmtx)
    irrigation_by_source = finite_mask.sum(axis=1)

    # Normalize each row by source irrigation, ignoring empty rows.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        impermeability_matrix = dmtx / irrigation_by_source[:, None]
    impermeability = np.nanmean(impermeability_matrix)

    inf_mask = np.isinf(dmx)
    inaccessibility = inf_mask.sum()
    irrigation = np.isfinite(dmx).sum() - np.isnan(np.diag(dmx)).sum()

    resistance_matrix = dmx.copy()
    resistance_matrix[inf_mask] = penalty
    resistance = np.nanmean(resistance_matrix)

    return [
        float(impermeability),
        float(latency),
        float(resistance),
        float(inaccessibility),
        float(irrigation),
    ]


def ComputeSmartTemporalDistanceMeasures(tnet, random_seed=None):
    """
    Compute the smart-walker distance summary for a temporal network.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), symmetric per snapshot.
        random_seed: int or None, optional (default=None). Seed for reproducibility (used by RandomWalker).

    Returns:
        metrics: list. [smart_impermeability, smart_latency, smart_resistance, smart_inaccessibility, smart_irrigation]
    """
    T = tnet.shape[0]
    penalty = T + 1

    smart_latency_matrix = SmartWalker(tnet)
    smart_metrics = MeanLatencyMatrixAnalysis(smart_latency_matrix, penalty)
    return smart_metrics


def ComputeDrunkTemporalDistanceMeasures(tnet, n_reps=100, random_seed=None):
    """
    Compute the drunk-walker distance summary for a temporal network.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), symmetric per snapshot.
        n_reps: int, optional (default=100). Number of random-walk trials for the Monte Carlo estimator.
        random_seed: int or None, optional (default=None). Seed for reproducibility (used by RandomWalker).

    Returns:
        metrics: list. [drunk_impermeability, drunk_latency, drunk_resistance, drunk_inaccessibility, drunk_irrigation]
    """
    drunk_latency_matrix, drunk_metrics = RandomWalker(tnet, n_reps, seed=random_seed)
    return drunk_metrics


def ComputeCirculationLatencyAndRate(
    adjacency_matrices,
    max_latency=None,
    return_full=False,
    summary_stat="median",
):
    """
    Compute circulation latency and rate in one pass for a temporal network.
    For each start time τ and node i: • A start is *eligible* if deg_i(A_τ) > 1 (i.e., at least one neighbor besides self). • We allow waiting (self-loops) in every snapshot by forcing diag=1. • We detect the first time a walk that starts at i returns to i by tracking diagonal increases in cumulative products: P_τ,0 = A_τ; P_τ,ℓ = P_τ,ℓ-1 @ A_{τ+ℓ}. • When a new diagonal hit occurs for i at step ℓ, we record a latency of ℓ+1 (keeps your original convention), and count a hit for rate.
    The function trims to *active* nodes (any off-diagonal activity across time) for efficiency, then pads results conceptually via index mapping (active_nodes).

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Temporal adjacency (binary or weighted). Treated as binary for products. Symmetry is assumed but not required.
        max_latency: int or None, optional (default=None). Maximum steps searched for a return after τ. If None, uses T. Also capped by remaining horizon T-τ.
        return_full: bool, optional (default=False). If True, return detailed per-node arrays in addition to global summaries.
        summary_stat: `median` or `mean` for reducing nodewise latency.

    Returns:
        If `return_full` is False, a summary dict including median nodewise latency and rate statistics. If `return_full` is True, the same summaries plus nodewise detail arrays.

    Notes:
        • Diagonals are forced to 1 across all snapshots (waiting allowed). • Eligibility uses the start snapshot only (deg > 1). • Complexity ~ O(T · M^3) in dense worst-case (M = active nodes), typically less due to early exits once all eligible nodes have hit for a given τ.
    """
    if adjacency_matrices.ndim != 3:
        raise ValueError("adjacency_matrices must be (T, N, N).")

    T, N, _ = adjacency_matrices.shape
    if max_latency is None:
        max_latency = T
    else:
        max_latency = int(max_latency)
        if max_latency <= 0:
            raise ValueError("max_latency must be a positive integer.")

    # Identify active nodes: any off-diagonal activity over time
    aggregated = adjacency_matrices.sum(axis=0).copy()
    np.fill_diagonal(aggregated, 0)
    node_activity = aggregated.sum(axis=0) + aggregated.sum(axis=1)
    active_nodes = np.where(node_activity > 0)[0]
    M = len(active_nodes)
    if M == 0:
        raise ValueError("No active nodes in the temporal network.")

    # Trim to active; binarize; force diag=1 (waiting)
    adj = (adjacency_matrices[:, active_nodes][:, :, active_nodes] > 0).astype(np.int64)
    np.einsum('tii->ti', adj)[:] = 1

    # Per-node accumulators
    # Latency stats
    min_latency   = np.full(M, np.inf, dtype=float)  # shortest observed latency per node
    sum_latency   = np.zeros(M, dtype=float)         # sum of first-return latencies
    count_hits    = np.zeros(M, dtype=np.int64)      # number of (τ,i) starts with a return

    # Rate stats
    eligible      = np.zeros(M, dtype=np.int64)      # number of eligible starts per node

    # Sweep start times
    for start_time in range(T - 1):
        A0 = adj[start_time]

        # Eligible nodes at τ: deg > 1 (beyond self-loop)
        deg_mask = (A0.sum(axis=1) > 1)
        if not np.any(deg_mask):
            continue
        eligible += deg_mask.astype(np.int64)

        # Cumulative product kernel
        path = A0.copy()
        prev_diag = np.diag(path).copy()
        recorded = np.zeros(M, dtype=bool)  # ensure one hit max per (node, τ)

        max_L = min(max_latency, T - start_time)
        for latency in range(1, max_L):
            path = path @ adj[start_time + latency]
            new_diag = np.diag(path)

            hit = (new_diag > prev_diag) & (~recorded) & deg_mask
            if np.any(hit):
                lat_val = latency + 1  # keep your convention
                sum_latency[hit] += lat_val
                count_hits[hit]  += 1
                np.minimum(min_latency[hit], lat_val, out=min_latency[hit])
                recorded[hit] = True

                # Optional early exit: when every eligible node for this τ has hit
                if recorded[deg_mask].all():
                    break

            prev_diag = new_diag

    # Nodewise aggregates
    nodewise_mean_latency = np.full(M, np.nan, dtype=float)
    hit_mask = count_hits > 0
    nodewise_mean_latency[hit_mask] = sum_latency[hit_mask] / count_hits[hit_mask]

    nodewise_min_latency = min_latency.copy()
    nodewise_min_latency[~hit_mask] = np.nan

    nodewise_rate = np.full(M, np.nan, dtype=float)
    elig_mask = eligible > 0
    nodewise_rate[elig_mask] = count_hits[elig_mask] / eligible[elig_mask]

    # Global summaries
    latency_mean  = float(np.nanmean(nodewise_mean_latency))
    latency_std   = float(np.nanstd(nodewise_mean_latency))
    latency_median = float(np.nanmedian(nodewise_mean_latency))
    latency_summary = _summarize_scores(nodewise_mean_latency, summary_stat=summary_stat)

    rate_nodewise_mean = float(np.nanmean(nodewise_rate))
    total_hits = int(count_hits.sum())
    total_eligible = int(eligible.sum())
    rate_overall = float(total_hits / total_eligible) if total_eligible > 0 else np.nan

    mean_hits_per_node = float(count_hits.mean())

    if not return_full:
        return {
            'latency_median': latency_median,
            'latency_summary': float(latency_summary),
            'summary_stat': summary_stat,
            'latency_mean': latency_mean,
            'latency_std': latency_std,
            'rate_nodewise_mean': rate_nodewise_mean,
            'rate_overall': rate_overall,
            'mean_hits_per_node': mean_hits_per_node
        }

    return {
        'latency_median': latency_median,
        'latency_summary': float(latency_summary),
        'summary_stat': summary_stat,
        'latency_mean': latency_mean,
        'latency_std': latency_std,
        'rate_nodewise_mean': rate_nodewise_mean,
        'rate_overall': rate_overall,
        'mean_hits_per_node': mean_hits_per_node,
        'active_nodes': active_nodes,
        'nodewise_mean_latency': nodewise_mean_latency,
        'nodewise_min_latency': nodewise_min_latency,
        'nodewise_hits': count_hits,
        'nodewise_eligible': eligible,
        'nodewise_rate': nodewise_rate
    }


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

    # Active nodes (same criterion as your original)
    node_activity = adjacency_matrices.sum(axis=(0, 1)) + adjacency_matrices.sum(axis=(0, 2))
    active_nodes = np.where(node_activity > 0)[0]
    n_active = len(active_nodes)
    if n_active == 0:
        raise ValueError("No active nodes in the temporal network.")

    # Trim, binarize, force diag=1 (waiting)
    adj = (adjacency_matrices[:, active_nodes][:, :, active_nodes] > 0).astype(np.int64)
    for t in range(T):
        np.fill_diagonal(adj[t], 1)

    # N-length accumulators (memory O(N) instead of O(N*T))
    min_latency   = np.full(n_active, np.inf, dtype=float)
    sum_latency   = np.zeros(n_active, dtype=float)
    count_hits    = np.zeros(n_active, dtype=np.int64)

    # Iterate start times; vectorized across nodes
    for start_time in range(T - 1):
        A0 = adj[start_time]

        # Nodes with only waiting at start_time cannot circulate
        deg_mask = (A0.sum(axis=1) > 1)
        if not np.any(deg_mask):
            continue

        path = A0.copy()
        prev_diag = np.diag(path).copy()
        recorded = np.zeros(n_active, dtype=bool)  # ensure only the first latency per (node, start_time)

        max_L = min(max_latency, T - start_time)
        for latency in range(1, max_L):
            path = path @ adj[start_time + latency]
            new_diag = np.diag(path)

            hit = (new_diag > prev_diag) & (~recorded) & deg_mask
            if np.any(hit):
                lat_val = latency + 1  
                # Update accumulators
                sum_latency[hit] += lat_val
                count_hits[hit]  += 1
                # Shortest seen latency per node
                np.minimum(min_latency[hit], lat_val, out=min_latency[hit])
                # Prevent double counting at this start_time for those nodes
                recorded[hit] = True

            # Early exit if all eligible nodes got a first return
            if recorded[deg_mask].all():
                break

            prev_diag = new_diag

    # Nodewise means (NaN where no hits)
    nodewise_mean_latency = np.full(n_active, np.nan, dtype=float)
    nonzero = count_hits > 0
    nodewise_mean_latency[nonzero] = sum_latency[nonzero] / count_hits[nonzero]

    # Global summaries from the per-node *mean* latencies (matches previous behavior)
    global_mean_latency = np.nanmean(nodewise_mean_latency)
    global_std_latency  = np.nanstd(nodewise_mean_latency)
    global_median_latency = np.nanmedian(nodewise_mean_latency)
    global_summary_latency = _summarize_scores(
        nodewise_mean_latency, summary_stat=summary_stat
    )

    if return_full:
        # Replace inf (no hits) with NaN for readability
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






def LatencyMatrixAnalysis(dmtx, penalty): #summarize_temporal_distance_matrix(dmtx):
    """
    Summarize a temporal distance matrix into average values

    Args:
        dmtx: array, shape (N, N). Matrix of minimal distances between nodes (inf for unreachable pairs).

    Returns:
        - irrigation,
        - resistance,
        - distance,
        - penalized distance (for inf pairs) with penalty,
        - number of infinite distance pairs,
        - number of finite distance pairs
    """
    
    np.fill_diagonal(dmx, np.nan) # diag to nan
    dmtx = np.where(np.isinf(dmx), np.nan, dmx) # inf to nan; 
    dist = np.nanmean(dmtx)
    dbool = ~ np.isnan(dmtx)
    reach = np.sum(dbool, 1)
    irr = np.mean(reach)
    resmtx = dmtx/reach[:, None]
    res = np.nanmean(resmtx)

    mask = np.isinf(dmx)
    ninf = np.sum(mask)
    nfin = np.sum(np.isfinite(dmx))

    pmtx = dmx.copy()
    pmtx[mask] = penalty

    pdist = np.nanmean(pmtx)

    return [irr, res, dist, pdist, ninf, nfin]
    
