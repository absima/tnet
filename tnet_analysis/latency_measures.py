import numpy as np
import warnings
from numba import njit


def _summarize_scores(values, summary_stat=None):
    """Summarize node scores with percentiles or a requested statistic."""
    if summary_stat is None or summary_stat == "percentiles":
        return np.nanpercentile(values, [5, 25, 50, 75, 95]).tolist()
    if summary_stat == "both":
        return [float(np.nanmean(values)), float(np.nanmedian(values))]
    if summary_stat == "median":
        return float(np.nanmedian(values))
    if summary_stat == "mean":
        return float(np.nanmean(values))
    raise ValueError(
        "summary_stat must be None, 'percentiles', 'mean', 'median', or 'both'."
    )


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


def _ValidateMultiHopInputs(tnet, hops_per_frame, latency_unit="frames"):
    """Return a binary temporal network with waiting, validating hop settings."""
    valid_latency_units = {"frames", "frame_fraction", "substeps"}
    arr = np.asarray(tnet)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2] or not all(arr.shape):
        raise ValueError("tnet must have shape (T, N, N) with T, N > 0.")
    if isinstance(hops_per_frame, bool) or not isinstance(hops_per_frame, (int, np.integer)):
        raise TypeError("hops_per_frame must be a positive integer.")
    if hops_per_frame < 1:
        raise ValueError("hops_per_frame must be at least 1.")
    if latency_unit not in valid_latency_units:
        raise ValueError(f"latency_unit must be one of {sorted(valid_latency_units)}.")
    network = (arr > 0).astype(np.int16)
    diagonal = np.arange(arr.shape[1])
    network[:, diagonal, diagonal] = 1
    return network


def _ConvertSubstepDistances(distances, hops_per_frame, latency_unit):
    """Convert substep arrivals to frame, fractional-frame, or substep units."""
    converted = np.array(distances, dtype=float, copy=True)
    finite = np.isfinite(converted)
    if latency_unit == "frames":
        converted[finite] = np.ceil(converted[finite] / hops_per_frame)
    elif latency_unit == "frame_fraction":
        converted[finite] /= hops_per_frame
    return converted


def _DefaultMultiHopPenalty(n_frames, hops_per_frame, latency_unit):
    """Keep unreachable pairs one frame beyond the observed horizon."""
    if latency_unit == "substeps":
        return float(hops_per_frame * (n_frames + 1))
    return float(n_frames + 1)


def _SmartWalkerSubsteps(network, hops_per_frame):
    """Record the first propagation substep at which each target is reached."""
    n_nodes = network.shape[1]
    reach = np.eye(n_nodes, dtype=bool)
    arrival = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(arrival, 0)
    substep = 0
    for frame in network:
        for _ in range(hops_per_frame):
            substep += 1
            reach = (reach.astype(np.int16) @ frame) > 0
            arrival[reach & np.isinf(arrival)] = substep
    np.fill_diagonal(arrival, np.nan)
    return arrival


def SmartWalker(tnet, hops_per_frame=1, latency_unit="frames"):
    """
    Compute earliest-arrival temporal distances using a 'smart walker'.
    The algorithm aggregates temporal reachability cumulatively across time.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), assumed symmetric per snapshot. Diagonals are forced to 1 internally (self-reachability).
        hops_per_frame: int, optional (default=1). Maximum propagation steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.

    Returns:
        dmtx: np.ndarray, shape (N, N). Earliest arrival from i to j in the requested unit; np.nan on diagonal and np.inf for unreachable pairs.
    """
    network = _ValidateMultiHopInputs(tnet, hops_per_frame, latency_unit)
    if hops_per_frame > 1 or latency_unit != "frames":
        arrival = _SmartWalkerSubsteps(network, hops_per_frame)
        return _ConvertSubstepDistances(arrival, hops_per_frame, latency_unit)

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


# Compiled kernels used only by RandomWalker.
@njit(cache=True)
def _RandomWalkerNeighborTable(network):
    """Store each frame/node's neighbors once, including the waiting loop."""
    n_frames, n_nodes, _ = network.shape
    neighbors = np.empty((n_frames, n_nodes, n_nodes), dtype=np.int32)
    degrees = np.zeros((n_frames, n_nodes), dtype=np.int32)
    for t in range(n_frames):
        for node in range(n_nodes):
            count = 0
            for target in range(n_nodes):
                if network[t, node, target]:
                    neighbors[t, node, count] = target
                    count += 1
            degrees[t, node] = count
    return neighbors, degrees


@njit(cache=True)
def _RandomWalkerFirstHits(neighbors, degrees, n_trials, hops_per_frame, seed):
    """Run the same within-frame walk rule as the Python implementation."""
    np.random.seed(seed)
    n_frames, n_nodes = degrees.shape
    hits = np.full((n_trials, n_nodes, n_nodes), -1, dtype=np.int32)
    for trial in range(n_trials):
        for source in range(n_nodes):
            current = source
            visited = np.zeros(n_nodes, dtype=np.uint8)
            visited[source] = 1
            n_visited = 1
            substep = 0
            for t in range(n_frames):
                for _ in range(hops_per_frame):
                    substep += 1
                    degree = degrees[t, current]
                    current = neighbors[t, current, np.random.randint(degree)]
                    if visited[current] == 0:
                        hits[trial, source, current] = substep
                        visited[current] = 1
                        n_visited += 1
                    if n_visited == n_nodes:
                        break
                if n_visited == n_nodes:
                    break
    return hits


def RandomWalker(tnet, n_trials, return_fpt_matrix=False, seed=None,
                 summary_stat="mean", hops_per_frame=1, latency_unit="frames",
                 penalty=None):
    """
    Estimate temporal distances via compiled Monte Carlo random walks.
    For each trial and source node, simulate a walk over snapshots and record
    first-passage times.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), assumed symmetric per snapshot. Diagonals are forced to 1 internally (self-reachability).
        n_trials: int. Number of Monte Carlo trials.
        return_fpt_matrix: bool, optional (default=False). If True, also return the per-trial FPT tensors.
        seed: int or None, optional (default=None). Seed for reproducibility.
        summary_stat: {"mean", "both"}, optional (default="mean"). Return five legacy values or three adjacent mean/median pairs followed by two counts.
        hops_per_frame: int, optional (default=1). Maximum random-walk steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.
        penalty: float or None, optional (default=None). Unreachable-pair replacement; defaults to one frame beyond the horizon in the requested unit.

    Returns:
        dmtx: np.ndarray, shape (N, N). Mean first passage time across trials, ignoring np.inf entries. np.nan on diagonal; np.inf if a node is never reached across all trials.
        mean_metrics: list[float]. Trial-averaged summary statistics (five by default, eight for "both").
        all_fpt: np.ndarray, optional, shape (n_trials, N, N). Returned only if `return_fpt_matrix=True`.

    Notes:
        Requires Numba. A fixed seed reproduces this compiled implementation;
    """
    network = _ValidateMultiHopInputs(tnet, hops_per_frame, latency_unit)
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if summary_stat not in ("mean", "both"):
        raise ValueError("summary_stat must be 'mean' or 'both'.")
    if penalty is None:
        penalty = _DefaultMultiHopPenalty(network.shape[0], hops_per_frame, latency_unit)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    else:
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer or None.")
        seed = int(seed)
        if seed >= 2**32:
            seed = int(np.random.SeedSequence(seed).generate_state(1)[0])

    neighbors, degrees = _RandomWalkerNeighborTable(network)
    hits = _RandomWalkerFirstHits(neighbors, degrees, n_trials, hops_per_frame, seed)
    n_nodes = network.shape[1]
    finite_sum = np.zeros((n_nodes, n_nodes), dtype=float)
    finite_count = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    metric_array = np.empty((n_trials, 8 if summary_stat == "both" else 5))
    all_fpt = [] if return_fpt_matrix else None

    for trial in range(n_trials):
        distances = hits[trial].astype(float)
        distances[distances < 0] = np.inf
        np.fill_diagonal(distances, np.nan)
        distances = _ConvertSubstepDistances(distances, hops_per_frame, latency_unit)
        finite = np.isfinite(distances)
        finite_sum += np.where(finite, distances, 0.0)
        finite_count += finite
        metric_array[trial] = MeanLatencyMatrixAnalysis(
            distances, penalty, summary_stat=summary_stat
        )
        if return_fpt_matrix:
            all_fpt.append(distances)

    dmtx = np.full((n_nodes, n_nodes), np.inf)
    np.divide(finite_sum, finite_count, out=dmtx, where=finite_count > 0)
    np.fill_diagonal(dmtx, np.nan)
    mean_metrics = metric_array.mean(axis=0).tolist()
    if return_fpt_matrix:
        return dmtx, mean_metrics, np.stack(all_fpt)
    return dmtx, mean_metrics


def MeanLatencyMatrixAnalysis(dmx, penalty, summary_stat="mean"):
    """
    Summarize a distance/first-passage matrix with five quantities.
    Given a matrix `dmx` where diagonal is NaN, finite off-diagonals are distances, and +inf denotes unreachable pairs, compute:
    1) impermeability: Mean of distance normalized by source irrigation, averaged over sources. 2) latency: Mean of all finite distances. 3) resistance: Mean distance after replacing `+inf` with `penalty`. 4) inaccessibility: Count of `+inf` entries. 5) irrigation: Count of finite entries.

    Args:
        dmx: np.ndarray, shape (N, N). Distance or first-passage matrix with NaN on the diagonal and possibly +inf.
        penalty: float. Replacement value used to compute penalized mean distance.
        summary_stat: {"mean", "both"}, optional (default="mean"). For "both", interleave mean and median for the first three measures, then return each count once.

    Returns:
        stats: list[float]. Five legacy values by default, or eight values for "both".
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
    # The diagonal is NaN and therefore already excluded by np.isfinite.
    irrigation = np.isfinite(dmx).sum()

    resistance_matrix = dmx.copy()
    resistance_matrix[inf_mask] = penalty
    resistance = np.nanmean(resistance_matrix)

    if summary_stat == "both":
        return [
            float(impermeability), float(np.nanmedian(impermeability_matrix)),
            float(latency), float(np.nanmedian(dmtx)),
            float(resistance), float(np.nanmedian(resistance_matrix)),
            float(inaccessibility), float(irrigation),
        ]
    if summary_stat != "mean":
        raise ValueError("summary_stat must be 'mean' or 'both'.")
    return [float(impermeability), float(latency), float(resistance),
            float(inaccessibility), float(irrigation)]


def ComputeSmartTemporalDistanceMeasures(tnet, random_seed=None, summary_stat="mean",
                                         hops_per_frame=1, latency_unit="frames",
                                         penalty=None):
    """
    Compute the smart-walker distance summary for a temporal network.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), symmetric per snapshot.
        random_seed: int or None, optional (default=None). Seed for reproducibility (used by RandomWalker).
        summary_stat: {"mean", "both"}, optional (default="mean"). Select the five-value legacy or eight-value mixed schema.
        hops_per_frame: int, optional (default=1). Maximum propagation steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.
        penalty: float or None, optional (default=None). Unreachable-pair replacement in the requested unit.

    Returns:
        metrics: list. Five legacy values by default, or three mean/median pairs followed by two counts.
    """
    T = tnet.shape[0]
    if penalty is None:
        penalty = _DefaultMultiHopPenalty(T, hops_per_frame, latency_unit)

    smart_latency_matrix = SmartWalker(
        tnet, hops_per_frame=hops_per_frame, latency_unit=latency_unit
    )
    smart_metrics = MeanLatencyMatrixAnalysis(
        smart_latency_matrix, penalty, summary_stat=summary_stat
    )
    return smart_metrics


def ComputeDrunkTemporalDistanceMeasures(tnet, n_reps=100, random_seed=None,
                                         summary_stat="mean", hops_per_frame=1,
                                         latency_unit="frames", penalty=None):
    """
    Compute the drunk-walker distance summary for a temporal network.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), symmetric per snapshot.
        n_reps: int, optional (default=100). Number of random-walk trials for the Monte Carlo estimator.
        random_seed: int or None, optional (default=None). Seed for reproducibility (used by RandomWalker).
        summary_stat: {"mean", "both"}, optional (default="mean"). Select the five-value legacy or eight-value mixed schema.
        hops_per_frame: int, optional (default=1). Maximum random-walk steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.
        penalty: float or None, optional (default=None). Unreachable-pair replacement in the requested unit.

    Returns:
        metrics: list. Five legacy values by default, or eight values averaged across random-walk trials.
    """
    drunk_latency_matrix, drunk_metrics = RandomWalker(
        tnet, n_reps, seed=random_seed, summary_stat=summary_stat,
        hops_per_frame=hops_per_frame, latency_unit=latency_unit,
        penalty=penalty,
    )
    return drunk_metrics


def ComputeCirculationLatencyAndRate(
    adjacency_matrices,
    max_latency=None,
    return_full=False,
    summary_stat="median",
    hops_per_frame=1,
    latency_unit="frames",
):
    """
    Compute circulation latency and rate in one pass for a temporal network.
    For each start time τ and node i: • A start is *eligible* if deg_i(A_τ) > 1 (i.e., at least one neighbor besides self). • We allow waiting (self-loops) in every snapshot by forcing diag=1. • We detect the first time a walk that starts at i returns to i by tracking diagonal increases in cumulative products: P_τ,0 = A_τ; P_τ,ℓ = P_τ,ℓ-1 @ A_{τ+ℓ}. • When a new diagonal hit occurs for i at step ℓ, we record a latency of ℓ+1 (keeps your original convention), and count a hit for rate.
    The function trims to *active* nodes (any off-diagonal activity across time) for efficiency, then pads results conceptually via index mapping (active_nodes).

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Temporal adjacency (binary or weighted). Treated as binary for products. Symmetry is assumed but not required.
        max_latency: int or None, optional (default=None). Maximum steps searched for a return after τ. If None, uses T. Also capped by remaining horizon T-τ.
        return_full: bool, optional (default=False). If True, return detailed per-node arrays in addition to global summaries.
        summary_stat: {"median", "mean"}, optional (default="median"). Scalar reduction for nodewise latency.
        hops_per_frame: int, optional (default=1). Maximum propagation steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of return latency.

    Returns:
        If `return_full` is False, a summary dict including median nodewise latency and rate statistics. If `return_full` is True, the same summaries plus nodewise detail arrays.

    Notes:
        • Diagonals are forced to 1 across all snapshots (waiting allowed). • Eligibility uses the start snapshot only (deg > 1). • Complexity ~ O(T · M^3) in dense worst-case (M = active nodes), typically less due to early exits once all eligible nodes have hit for a given τ.
    """
    _ValidateMultiHopInputs(adjacency_matrices, hops_per_frame, latency_unit)
    if hops_per_frame > 1:
        details = ComputeCirculationLatency(
            adjacency_matrices, max_latency=max_latency, return_full=True,
            summary_stat=summary_stat, hops_per_frame=hops_per_frame,
            latency_unit=latency_unit,
        )
        network = _ValidateMultiHopInputs(adjacency_matrices, hops_per_frame, latency_unit)
        active = details['active_nodes']
        eligible = (network[:, active][:, :, active].sum(axis=2) > 1).sum(axis=0)
        hits = details['hit_count']
        nodewise_rate = np.full(active.size, np.nan)
        np.divide(hits, eligible, out=nodewise_rate, where=eligible > 0)
        total_eligible = int(eligible.sum())
        result = {
            'latency_median': details['global_median_latency'],
            'latency_summary': float(details['global_summary_latency']),
            'summary_stat': summary_stat,
            'latency_mean': details['global_mean_latency'],
            'latency_std': details['global_std_latency'],
            'rate_nodewise_mean': float(np.nanmean(nodewise_rate)),
            'rate_overall': float(hits.sum() / total_eligible) if total_eligible else np.nan,
            'mean_hits_per_node': float(hits.mean()),
        }
        if return_full:
            result.update({
                'active_nodes': active,
                'nodewise_mean_latency': details['nodewise_mean_latency'],
                'nodewise_min_latency': details['min_latency'],
                'nodewise_hits': hits,
                'nodewise_eligible': eligible,
                'nodewise_rate': nodewise_rate,
                'hops_per_frame': int(hops_per_frame),
                'latency_unit': latency_unit,
            })
        return result

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


def _ComputeCirculationLatencyMultiHop(adjacency_matrices, max_latency,
                                      return_full, summary_stat,
                                      hops_per_frame, latency_unit):
    """Find first non-waiting returns, allowing several steps in each frame."""
    adj = _ValidateMultiHopInputs(adjacency_matrices, hops_per_frame, latency_unit)
    n_frames, n_nodes, _ = adj.shape
    if max_latency is None:
        max_latency = n_frames
    if isinstance(max_latency, bool) or not isinstance(max_latency, (int, np.integer)) or max_latency < 1:
        raise ValueError("max_latency must be a positive integer number of frames.")

    node_activity = adj.sum(axis=(0, 1)) + adj.sum(axis=(0, 2))
    active_nodes = np.flatnonzero(node_activity > 0)
    if active_nodes.size == 0:
        raise ValueError("No active nodes in the temporal network.")
    adj = adj[:, active_nodes][:, :, active_nodes]
    n_active = active_nodes.size
    min_latency = np.full(n_active, np.inf)
    sum_latency = np.zeros(n_active, dtype=float)
    count_hits = np.zeros(n_active, dtype=np.int64)

    for start_time in range(n_frames):
        eligible = adj[start_time].sum(axis=1) > 1
        if not np.any(eligible):
            continue
        path = np.eye(n_active, dtype=np.int16)
        recorded = np.zeros(n_active, dtype=bool)
        n_substeps = min(max_latency, n_frames - start_time) * hops_per_frame
        for step_index in range(n_substeps):
            frame = adj[start_time + step_index // hops_per_frame]
            path = np.minimum(path @ frame, 2).astype(np.int16)
            if step_index == 0:
                continue
            hit = (np.diag(path) > 1) & ~recorded & eligible
            if np.any(hit):
                substep = step_index + 1
                if latency_unit == "frames":
                    latency = float(np.ceil(substep / hops_per_frame))
                elif latency_unit == "frame_fraction":
                    latency = substep / hops_per_frame
                else:
                    latency = float(substep)
                sum_latency[hit] += latency
                count_hits[hit] += 1
                min_latency[hit] = np.minimum(min_latency[hit], latency)
                recorded[hit] = True
            if recorded[eligible].all():
                break

    nodewise_mean_latency = np.full(n_active, np.nan)
    nonzero = count_hits > 0
    nodewise_mean_latency[nonzero] = sum_latency[nonzero] / count_hits[nonzero]
    global_summary_latency = _summarize_scores(
        nodewise_mean_latency, summary_stat=summary_stat
    )
    mean_count = float(np.mean(count_hits))
    if not return_full:
        return global_summary_latency, mean_count

    min_latency[~nonzero] = np.nan
    return {
        'nodewise_mean_latency': nodewise_mean_latency,
        'min_latency': min_latency,
        'hit_count': count_hits,
        'global_mean_latency': float(np.nanmean(nodewise_mean_latency)),
        'global_std_latency': float(np.nanstd(nodewise_mean_latency)),
        'global_median_latency': float(np.nanmedian(nodewise_mean_latency)),
        'global_summary_latency': global_summary_latency,
        'summary_stat': summary_stat,
        'active_nodes': active_nodes,
        'hops_per_frame': int(hops_per_frame),
        'latency_unit': latency_unit,
    }


def ComputeCirculationLatency(
    adjacency_matrices,
    max_latency=None,
    return_full=False,
    summary_stat=None,
    hops_per_frame=1,
    latency_unit="frames",
):
    """
    Compute node-wise circulation latency in a temporal network (vectorized).

    Args:
        adjacency_matrices: array, shape (T, N, N). Sequence of adjacency matrices (binary, 0/1), with waiting (diagonal ones).
        max_latency: int, optional. Maximum number of steps to search. If None, set to T.
        return_full: bool, optional. If True, return nodewise details and summary scalars.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].
        hops_per_frame: int, optional (default=1). Maximum propagation steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of return latency.

    Returns:
        If `return_full` is False, `(summary_latency, mean_count_hits)`. If `return_full` is True, a dict with nodewise arrays plus global mean, std, median, and requested summary latency.
    """

    _ValidateMultiHopInputs(adjacency_matrices, hops_per_frame, latency_unit)
    if hops_per_frame > 1:
        return _ComputeCirculationLatencyMultiHop(
            adjacency_matrices, max_latency, return_full, summary_stat,
            hops_per_frame, latency_unit,
        )

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
            'global_summary_latency': global_summary_latency,
            'summary_stat': summary_stat,
            'active_nodes': active_nodes
        }
    else:
        return global_summary_latency, float(np.mean(count_hits))






def LatencyMatrixAnalysis(dmtx, penalty):
    """
    Summarize a temporal distance matrix with the five-value latency schema.

    Args:
        dmtx: np.ndarray, shape (N, N). Matrix of minimal distances between nodes, with infinity for unreachable pairs.
        penalty: float. Replacement value used for unreachable pairs when computing resistance.

    Returns:
        stats: list[float]. [impermeability, latency, resistance, inaccessibility, irrigation]
    """
    return MeanLatencyMatrixAnalysis(dmtx, penalty)
    
