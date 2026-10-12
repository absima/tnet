import numpy as np
import warnings
from numba import njit
from ._summary import SummarizeScores as _summarize_scores


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
        threshold: float or None, optional (default=None). If provided, an edge is A > threshold. If None, every nonzero entry is an edge.

    Returns:
        mean_deg: float. Mean degree averaged over all nodes and time snapshots.
        std_deg: float. Standard deviation of degrees over all nodes and time snapshots.

    Notes:
        - Diagonals are ignored without modifying the caller's array.
        - Degree counts edges, not weighted strength. Directed inputs count outgoing edges.
        - Process one frame at a time; no full copied/binarized temporal tensor.
    """
    A = np.asarray(adj_tensor)
    if A.ndim != 3 or A.shape[1] != A.shape[2] or A.shape[1] == 0:
        raise ValueError("adj_tensor must have shape (T, N, N), with N > 0.")
    if threshold is not None and (np.ndim(threshold) != 0 or not np.isfinite(threshold)):
        raise ValueError("threshold must be a finite scalar or None.")
    degrees = np.empty(A.shape[:2], dtype=np.int64)
    for t, frame in enumerate(A):
        if not np.isfinite(frame).all():
            raise ValueError("Adjacency must be finite.")
        present = frame != 0 if threshold is None else frame > threshold
        degrees[t] = np.count_nonzero(present, axis=1) - np.diag(present)
    if not degrees.size:
        return np.nan, np.nan
    return float(degrees.mean()), float(degrees.std())


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


# Shared packing for SmartWalker and one-hop return.
@njit(cache=True)
def _PackTemporalNeighbors(network):
    """Store binary neighbor sets as 64-bit words for temporal path searches."""
    n_frames, n_nodes, _ = network.shape
    n_words = (n_nodes + 63) // 64
    packed = np.zeros((n_frames, n_nodes, n_words), dtype=np.uint64)
    for t in range(n_frames):
        for source in range(n_nodes):
            for target in range(n_nodes):
                if network[t, source, target]:
                    packed[t, source, target // 64] |= np.uint64(1) << np.uint64(target % 64)
    return packed


@njit(cache=True)
def _SmartWalkerFirstArrivals(packed, n_nodes, hops_per_frame):
    """Propagate one frozen reachable set per substep, stopping completed rows."""
    n_frames, _, n_words = packed.shape
    reach = np.zeros((n_nodes, n_words), dtype=np.uint64)
    counts = np.ones(n_nodes, dtype=np.int64)
    arrivals = np.full((n_nodes, n_nodes), np.inf)
    for source in range(n_nodes):
        reach[source, source // 64] = np.uint64(1) << np.uint64(source % 64)
        arrivals[source, source] = np.nan
    unfinished = 0
    for source in range(n_nodes):
        active = False
        source_word = source // 64
        source_bit = np.uint64(1) << np.uint64(source % 64)
        for t in range(n_frames):
            for word in range(n_words):
                neighbors = packed[t, source, word]
                if word == source_word:
                    neighbors &= ~source_bit
                if neighbors:
                    active = True
                    break
            if active:
                break
        if active:
            unfinished += 1
        else:
            # A source that can only wait never reaches another node.
            counts[source] = n_nodes
    if unfinished == 0:
        return arrivals

    substep = 0
    for t in range(n_frames):
        for _ in range(hops_per_frame):
            substep += 1
            for source in range(n_nodes):
                if counts[source] == n_nodes:
                    continue
                # Read only the previous set: newly reached nodes cannot
                # propagate again until the next permitted substep.
                next_row = reach[source].copy()
                for node in range(n_nodes):
                    bit = np.uint64(1) << np.uint64(node % 64)
                    if reach[source, node // 64] & bit:
                        for word in range(n_words):
                            next_row[word] |= packed[t, node, word]
                for target in range(n_nodes):
                    word = target // 64
                    bit = np.uint64(1) << np.uint64(target % 64)
                    if (next_row[word] & bit) and not (reach[source, word] & bit):
                        arrivals[source, target] = substep
                        counts[source] += 1
                reach[source] = next_row
                if counts[source] == n_nodes:
                    unfinished -= 1
            if unfinished == 0:
                return arrivals
    return arrivals


def SmartWalker(tnet, hops_per_frame=1, latency_unit="frames"):
    """
    Compute earliest-arrival temporal distances using a 'smart walker'.
    Propagate bit-packed reachability, recording first arrivals and stopping
    finished sources. Waiting makes previously reached targets stay reachable.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency (binary), assumed symmetric per snapshot. Diagonals are forced to 1 internally (self-reachability).
        hops_per_frame: int, optional (default=1). Maximum propagation steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.

    Returns:
        dmtx: np.ndarray, shape (N, N). Earliest arrival from i to j in the requested unit; np.nan on diagonal and np.inf for unreachable pairs.
    """
    network = _ValidateMultiHopInputs(tnet, hops_per_frame, latency_unit)
    packed = _PackTemporalNeighbors(network)
    arrivals = _SmartWalkerFirstArrivals(packed, network.shape[1], hops_per_frame)
    return _ConvertSubstepDistances(arrivals, hops_per_frame, latency_unit)


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
        summary_stat: {"mean", "both"}, optional (default="mean"). Return five values or three adjacent mean/median pairs followed by two counts.
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
        stats: list[float]. Five values by default, or eight values for "both".
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
        summary_stat: {"mean", "both"}, optional (default="mean"). Select the five-value or eight-value mixed schema.
        hops_per_frame: int, optional (default=1). Maximum propagation steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.
        penalty: float or None, optional (default=None). Unreachable-pair replacement in the requested unit.

    Returns:
        metrics: list. Five values by default, or three mean/median pairs followed by two counts.
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
        summary_stat: {"mean", "both"}, optional (default="mean"). Select the five-value or eight-value mixed schema.
        hops_per_frame: int, optional (default=1). Maximum random-walk steps within each frame.
        latency_unit: {"frames", "frame_fraction", "substeps"}, optional (default="frames"). Unit of finite distances.
        penalty: float or None, optional (default=None). Unreachable-pair replacement in the requested unit.

    Returns:
        metrics: list. Five values by default, or eight values averaged across random-walk trials.
    """
    drunk_latency_matrix, drunk_metrics = RandomWalker(
        tnet, n_reps, seed=random_seed, summary_stat=summary_stat,
        hops_per_frame=hops_per_frame, latency_unit=latency_unit,
        penalty=penalty,
    )
    return drunk_metrics


def _ComputeReturnLatencyMultiHop(adjacency_matrices, max_latency,
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


@njit(cache=True)
def _ReturnFirstReturns(packed, n_nodes, max_latency):
    """Find first non-waiting returns without counting all possible walks."""
    n_frames, _, n_words = packed.shape
    minimum = np.full(n_nodes, np.inf)
    totals = np.zeros(n_nodes, dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)
    for start in range(n_frames - 1):
        reach = packed[start].copy()
        pending = np.zeros(n_nodes, dtype=np.bool_)
        remaining = 0
        for source in range(n_nodes):
            bit = np.uint64(1) << np.uint64(source % 64)
            reach[source, source // 64] &= ~bit
            for word in range(n_words):
                if reach[source, word]:
                    pending[source] = True
                    remaining += 1
                    break
        if remaining == 0:
            continue
        for latency in range(1, min(max_latency, n_frames - start)):
            frame = start + latency
            for source in range(n_nodes):
                if not pending[source]:
                    continue
                source_word = source // 64
                source_bit = np.uint64(1) << np.uint64(source % 64)
                returned = False
                # Check for a return before expanding the reachable set.
                for node in range(n_nodes):
                    bit = np.uint64(1) << np.uint64(node % 64)
                    if (reach[source, node // 64] & bit) and (packed[frame, node, source_word] & source_bit):
                        returned = True
                        break
                if returned:
                    distance = float(latency + 1)
                    totals[source] += distance
                    counts[source] += 1
                    minimum[source] = min(minimum[source], distance)
                    pending[source] = False
                    remaining -= 1
                    continue
                # Also allow paths that wait at the source before departing.
                next_row = packed[frame, source].copy()
                next_row[source_word] &= ~source_bit
                for node in range(n_nodes):
                    bit = np.uint64(1) << np.uint64(node % 64)
                    if reach[source, node // 64] & bit:
                        for word in range(n_words):
                            next_row[word] |= packed[frame, node, word]
                reach[source] = next_row
            if remaining == 0:
                break
    return minimum, totals, counts


def ComputeReturnLatency(
    adjacency_matrices,
    max_latency=None,
    return_full=False,
    summary_stat=None,
    hops_per_frame=1,
    latency_unit="frames",
):
    """
    Compute node-wise return latency; one-hop searches use packed reachability.

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

    network = _ValidateMultiHopInputs(adjacency_matrices, hops_per_frame, latency_unit)
    if hops_per_frame > 1:
        return _ComputeReturnLatencyMultiHop(
            adjacency_matrices, max_latency, return_full, summary_stat,
            hops_per_frame, latency_unit,
        )

    T, N, _ = adjacency_matrices.shape
    if max_latency is None:
        max_latency = T
    if isinstance(max_latency, bool) or not isinstance(max_latency, (int, np.integer)) or max_latency < 1:
        raise ValueError("max_latency must be a positive integer number of frames.")

    # Nodes with at least one successful return.
    node_activity = adjacency_matrices.sum(axis=(0, 1)) + adjacency_matrices.sum(axis=(0, 2))
    active_nodes = np.where(node_activity > 0)[0]
    n_active = len(active_nodes)
    if n_active == 0:
        raise ValueError("No active nodes in the temporal network.")

    adj = network if n_active == N else network[:, active_nodes][:, :, active_nodes]
    packed = _PackTemporalNeighbors(adj)
    min_latency, sum_latency, count_hits = _ReturnFirstReturns(
        packed, n_active, max_latency
    )

    # Nodewise means (NaN where no hits)
    nodewise_mean_latency = np.full(n_active, np.nan, dtype=float)
    nonzero = count_hits > 0
    nodewise_mean_latency[nonzero] = sum_latency[nonzero] / count_hits[nonzero]

    # Global summaries from per-node mean latencies.
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


# Alternative function names supported as aliases.
ComputeCirculationLatency = ComputeReturnLatency
_ComputeCirculationLatencyMultiHop = _ComputeReturnLatencyMultiHop
_CirculationFirstReturns = _ReturnFirstReturns
    
