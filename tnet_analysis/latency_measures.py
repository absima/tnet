import numpy as np
import warnings


def SetDiagonals(tensor, value=1):
    """
    Set the diagonals of all matrices in a temporal or static adjacency tensor.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N, N) or (N, N)
        Input adjacency tensor or matrix. If 2D, it is treated as a single snapshot.
    value : int or float, optional (default=1)
        Value to assign along the diagonal.

    Returns
    -------
    tensor_out : np.ndarray
        The modified tensor with diagonals set to `value`.
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

# =========================================================
# Degree statistics
# =========================================================

def ComputeAverageDegree(adj_tensor, threshold=None):
    """
    Compute the average and standard deviation of degrees over a temporal network.

    Parameters
    ----------
    adj_tensor : np.ndarray, shape (T, N, N)
        Temporal adjacency (binary or weighted). Assumed symmetric per snapshot.
    threshold : float or None, optional (default=None)
        If provided, binarize as (A > threshold). If None, use the input as-is
        (cast to int).

    Returns
    -------
    mean_deg : float
        Mean degree averaged over all nodes and time snapshots.
    std_deg : float
        Standard deviation of degrees over all nodes and time snapshots.

    Notes
    -----
    - Diagonals are set to 0 before degree computation.
    - Degree per snapshot is computed as row-sums of the binarized adjacency.
    """
    A = adj_tensor.copy()
    if threshold is None:
        binarized = (A != 0).astype(int)
    else:
        binarized = (A > threshold).astype(int)

    # zero out diagonals per snapshot
    T, n_nodes, _ = binarized.shape
    idx = np.arange(n_nodes)
    binarized[:, idx, idx] = 0

    degrees = binarized.sum(axis=2)   # shape (T, N)
    mean_deg = degrees.mean()
    std_deg = degrees.std()
    return mean_deg, std_deg







# =========================================================
# Smart walker / earliest-arrival temporal distance
# =========================================================

def SmartWalker(tnet):
    """
    Compute earliest-arrival temporal distances using a 'smart walker'.

    The algorithm aggregates temporal reachability cumulatively across time.
    For each time t, it multiplies the current reachability matrix by snapshot t,
    marking nodes reachable by time t+1 if a path exists using snapshots up to t.

    Parameters
    ----------
    tnet : np.ndarray, shape (T, N, N)
        Temporal adjacency (binary), assumed symmetric per snapshot.
        Diagonals are forced to 1 internally (self-reachability).

    Returns
    -------
    dmtx : np.ndarray, shape (N, N)
        Earliest arrival time (in snapshot steps) from i to j.
        np.nan on diagonal; np.inf if j is never reached from i.
    """
    tnet_copy = tnet.copy()
    # force all diagonals to 1 (self-reachability at every time)
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


# =========================================================
# Random walker (Monte Carlo first-passage times)
# =========================================================

def RandomWalker(tnet, n_trials, return_fpt_matrix=False, seed=None):
    """
    Estimate temporal distances via Monte Carlo random walks (first passage times).

    For each trial and each source node, simulate a random walk over T snapshots:
      - At time t, move uniformly at random to a neighbor in snapshot t.
      - Record first time each node is visited (first passage), in steps [1..T].
      - If a node is never visited, its FPT is set to np.inf for that source.

    Parameters
    ----------
    tnet : np.ndarray, shape (T, N, N)
        Temporal adjacency (binary), assumed symmetric per snapshot.
        Diagonals are forced to 1 internally (self-reachability).
    n_trials : int
        Number of Monte Carlo trials.
    return_fpt_matrix : bool, optional (default=False)
        If True, also return the per-trial FPT tensors.
    seed : int or None, optional (default=None)
        Seed for reproducibility.

    Returns
    -------
    dmtx : np.ndarray, shape (N, N)
        Mean first passage time across trials, ignoring np.inf entries.
        np.nan on diagonal; np.inf if a node is never reached across all trials.
    mean_q6 : list[float]
        Mean of the six summary statistics (see MeanDistance) across trials.
    all_fpt : np.ndarray, optional, shape (n_trials, N, N)
        Returned only if `return_fpt_matrix=True`.
    """
    rng = np.random.default_rng(seed)

    tnet_copy = tnet.copy()
    np.einsum('tii->ti', tnet_copy)[:] = 1  # force diagonal=1 at all times

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
                    break  # stuck (should be rare with diag=1)

                next_node = rng.choice(neighbors)

                if next_node not in visited:
                    first_hit[next_node] = t + 1
                    visited.add(next_node)

                where = next_node
                if len(visited) == n_nodes:
                    break

            dmtx_trial[source] = first_hit

        all_dmtx[trial] = dmtx_trial
        q6_array[trial] = MeanLatencyMatrixAnalysis(dmtx_trial, T + 1)  # penalty = T+1

    mean_q6 = q6_array.mean(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # mean over trials, ignoring inf entries
        dmtx = np.mean(all_dmtx, axis=0, where=~np.isinf(all_dmtx))
        # if a pair is inf in all trials, keep it as inf
        all_inf_mask = np.all(np.isinf(all_dmtx), axis=0)
        dmtx[all_inf_mask] = np.inf

    # diag as nan for distance-style matrices
    np.fill_diagonal(dmtx, np.nan)

    if return_fpt_matrix:
        return dmtx, mean_q6.tolist(), all_dmtx
    else:
        return dmtx, mean_q6.tolist()


# =========================================================
# Distance summaries (mean distance, reachability, penalties)
# =========================================================

def MeanLatencyMatrixAnalysis(dmx, penalty):
    """
    Summarize a distance/first-passage matrix with six quantities.

    Given a matrix `dmx` where diagonal is NaN, finite off-diagonals are distances,
    and +inf denotes unreachable pairs, compute:

    1) irr   : Mean out-reachability per source (# of finite targets).
    2) res   : Mean of (distance normalized by source reach), averaged over sources.
    3) dist  : Mean of all finite distances (pooled).
    4) pdist : Mean distance after replacing +inf with `penalty`.
    5) ninf  : Count of +inf entries.
    6) nfin  : Count of finite entries.

    Parameters
    ----------
    dmx : np.ndarray, shape (N, N)
        Distance or first-passage matrix with NaN on the diagonal and possibly +inf.
    penalty : float
        Replacement value used to compute penalized mean distance.

    Returns
    -------
    stats : list[float]
        [irr, res, dist, pdist, ninf, nfin]
    """
    dmx = dmx.copy()
    np.fill_diagonal(dmx, np.nan)

    dmtx = np.where(np.isinf(dmx), np.nan, dmx)
    dist = np.nanmean(dmtx)

    finite_mask = ~np.isnan(dmtx)
    reach = finite_mask.sum(axis=1)  # per-source reachable count
    irr = reach.mean()

    # normalize each row by its reach (avoid div-by-zero via nan)
    # with np.errstate(invalid='ignore', divide='ignore'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        resmtx = dmtx / reach[:, None]
    res = np.nanmean(resmtx)

    inf_mask = np.isinf(dmx)
    ninf = inf_mask.sum()
    nfin = np.isfinite(dmx).sum() - np.isnan(np.diag(dmx)).sum()  # exclude diag NaNs

    pmtx = dmx.copy()
    pmtx[inf_mask] = penalty
    pdist = np.nanmean(pmtx)

    return [float(irr), float(res), float(dist), float(pdist), float(ninf), float(nfin)]


# =========================================================
# End-to-end bundle for temporal distance measures
# =========================================================

def ComputeSmartTemporalDistanceMeasures(tnet, random_seed=None):
    """
    Compute a bundle of distance-related measures for a temporal network.

    This function reports:
      - SmartWalker summaries (earliest-arrival distances)

    Parameters
    ----------
    tnet : np.ndarray, shape (T, N, N)
        Temporal adjacency (binary), symmetric per snapshot.
    random_seed : int or None, optional (default=None)
        Seed for reproducibility (used by RandomWalker).

    Returns
    -------
    metrics : list
        - irrigation
        - impermeability
        - latency
        - penalized latency
        - number of infinite pairs 
        - number of finitie pairs ('irrigation')
    """
    T = tnet.shape[0]
    penalty = T + 1

    # avg_deg, std_deg = ComputeAverageDegree(tnet.copy())

    sdmtx = SmartWalker(tnet)
    smart_q6 = MeanLatencyMatrixAnalysis(sdmtx, penalty)

    # rdmtx, drunk_q6_mn = RandomWalker(tnet, n_reps, seed=random_seed)
    # drunk_q6 = MeanLatencyMatrixAnalysis(rdmtx, penalty)

    # Return flattened list of all summaries
    return smart_q6 

def ComputeDrunkTemporalDistanceMeasures(tnet, n_reps=100, random_seed=None):
    """
    Compute a bundle of distance-related measures for a temporal network.

    This function reports:
      - RandomWalker summaries (Monte Carlo FPT), plus a combined mean-Q6 across trials

    Parameters
    ----------
    tnet : np.ndarray, shape (T, N, N)
        Temporal adjacency (binary), symmetric per snapshot.
    n_reps : int, optional (default=100)
        Number of random-walk trials for the Monte Carlo estimator.
    random_seed : int or None, optional (default=None)
        Seed for reproducibility (used by RandomWalker).

    Returns
    -------
    metrics : list
        metrics : list
            - irrigation
            - impermeability
            - latency
            - penalized latency
            - number of infinite pairs 
            - number of finitie pairs ('irrigation')
    """
    # T = tnet.shape[0]
    # penalty = T + 1

    # avg_deg, std_deg = ComputeAverageDegree(tnet.copy())

    # sdmtx = SmartWalker(tnet)
    # smart_q6 = MeanLatencyMatrixAnalysis(sdmtx, penalty)

    rdmtx, drunk_q6_mn = RandomWalker(tnet, n_reps, seed=random_seed)
    # drunk_q6 = MeanLatencyMatrixAnalysis(rdmtx, penalty)

    # Return flattened list of all summaries
    return drunk_q6_mn


def ComputeCirculationLatencyAndRate(adjacency_matrices, max_latency=None, return_full=False):
    """
    Compute circulation latency and rate in one pass for a temporal network.

    For each start time τ and node i:
      • A start is *eligible* if deg_i(A_τ) > 1 (i.e., at least one neighbor besides self).
      • We allow waiting (self-loops) in every snapshot by forcing diag=1.
      • We detect the first time a walk that starts at i returns to i by tracking
        diagonal increases in cumulative products: P_τ,0 = A_τ; P_τ,ℓ = P_τ,ℓ-1 @ A_{τ+ℓ}.
      • When a new diagonal hit occurs for i at step ℓ, we record a latency of ℓ+1
        (keeps your original convention), and count a hit for rate.

    The function trims to *active* nodes (any off-diagonal activity across time) for
    efficiency, then pads results conceptually via index mapping (active_nodes).

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Temporal adjacency (binary or weighted). Treated as binary for products.
        Symmetry is assumed but not required.
    max_latency : int or None, optional (default=None)
        Maximum steps searched for a return after τ. If None, uses T. Also capped
        by remaining horizon T-τ.
    return_full : bool, optional (default=False)
        If True, return detailed per-node arrays in addition to global summaries.

    Returns
    -------
    If return_full is False:
        global : dict
            {
              'latency_percentiles'       : np.ndarray (5,)  # [5,25,50,75,95] of per-node mean latencies
              'latency_mean'              : float            # mean of per-node mean latencies
              'latency_std'               : float            # std  of per-node mean latencies
              'rate_nodewise_mean'        : float            # NaN-mean of per-node rates
              'rate_overall'              : float            # total hits / total eligible
              'mean_hits_per_node'        : float            # average hit count per node
            }

    If return_full is True:
        result : dict
            All the above plus:
            {
              'active_nodes'              : np.ndarray (M,),
              'nodewise_mean_latency'     : np.ndarray (M,),  # NaN where no hits
              'nodewise_min_latency'      : np.ndarray (M,),  # NaN where no hits
              'nodewise_hits'             : np.ndarray (M,),
              'nodewise_eligible'         : np.ndarray (M,),
              'nodewise_rate'             : np.ndarray (M,),  # hits/eligible, NaN if eligible=0
            }

    Notes
    -----
    • Diagonals are forced to 1 across all snapshots (waiting allowed).
    • Eligibility uses the start snapshot only (deg > 1).
    • Complexity ~ O(T · M^3) in dense worst-case (M = active nodes), typically
      less due to early exits once all eligible nodes have hit for a given τ.
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
    latency_pct   = np.nanpercentile(nodewise_mean_latency, [5, 25, 50, 75, 95])

    rate_nodewise_mean = float(np.nanmean(nodewise_rate))
    total_hits = int(count_hits.sum())
    total_eligible = int(eligible.sum())
    rate_overall = float(total_hits / total_eligible) if total_eligible > 0 else np.nan

    mean_hits_per_node = float(count_hits.mean())

    if not return_full:
        return {
            'latency_percentiles': latency_pct,
            'latency_mean': latency_mean,
            'latency_std': latency_std,
            'rate_nodewise_mean': rate_nodewise_mean,
            'rate_overall': rate_overall,
            'mean_hits_per_node': mean_hits_per_node
        }

    return {
        'latency_percentiles': latency_pct,
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


def ComputeCirculationLatency(adjacency_matrices, max_latency=None, return_full=False):
    """
        Compute node-wise circulation latency in a temporal network (vectorized).

        Parameters
        ----------
        adjacency_matrices : array, shape (T, N, N)
            Sequence of adjacency matrices (binary, 0/1), with waiting (diagonal ones).
        max_latency : int, optional
            Maximum number of steps to search. If None, set to T.
        return_full : bool, optional (default: False)
            If True, return the full dictionary (matrix + summaries).
            If False, return only global_percentiles_latency to match your current behavior.

        Returns
        -------
        If return_full=False (default):
            global_percentiles_latency : ndarray of shape (5,)
                (5,25,50,75,95) percentiles of nodewise mean latency.
            mean count hits (number of circulations recorded)
        If return_full=True:
            dict with keys:
                'circulation_matrix'        : (N_active, T) latency matrix (NaN if no circulation at that start time)
                'nodewise_mean_latency'     : (N_active,) mean latency per node
                'global_mean_latency'       : float
                'global_std_latency'        : float
                'global_percentiles_latency': array of (5,25,50,75,95)
                'active_nodes'              : indices of the kept nodes
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
                lat_val = latency + 1  # keep your convention
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
    global_percentiles_latency = np.nanpercentile(nodewise_mean_latency, [5, 25, 50, 75, 95])

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
            'global_percentiles_latency': global_percentiles_latency,
            'active_nodes': active_nodes
        }
    else:
        return global_percentiles_latency.tolist(), np.mean(count_hits)






def LatencyMatrixAnalysis(dmtx, penalty): #summarize_temporal_distance_matrix(dmtx):
    """
    Summarize a temporal distance matrix into average values 

    Parameters
    ----------
    dmtx : array, shape (N, N)
        Matrix of minimal distances between nodes (inf for unreachable pairs).

    Returns
    -------
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
    


