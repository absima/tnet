import numpy as np
import warnings
from scipy.stats import norm
from sklearn.metrics import normalized_mutual_info_score
from numba import njit
from ._summary import SummarizeScores as _summarize_scores
from ._temporal_statistics import (
    ValidateTemporalAdjacency, PositiveInteger, EdgeOverlapAndPersistence,
    NeighborhoodMemoryScores, LinkBurstinessScores,
)


@njit(cache=True)
def _RetentionScores(adjacency):
    """Track previously seen neighbors without building Python sets."""
    T, N, _ = adjacency.shape
    scores = np.full(N, np.nan)
    for i in range(N):
        past = np.zeros(N, dtype=np.bool_)
        has_past = False
        total = 0.0
        observations = 0
        for t in range(T):
            degree = 0
            returning = 0
            for j in range(N):
                if j != i and adjacency[t, i, j] > 0:
                    degree += 1
                    if past[j]:
                        returning += 1
                    past[j] = True
            if degree:
                if has_past:
                    total += returning / degree
                    observations += 1
                has_past = True
        if observations:
            scores[i] = total / observations
    return scores


@njit(cache=True)
def _WindowedRetentionScores(adjacency, delta):
    """Track each neighbor's last contact in O(N) workspace per node."""
    T, N, _ = adjacency.shape
    scores = np.full(N, np.nan)
    for i in range(N):
        last_seen = np.full(N, -1, dtype=np.int64)
        has_past = False
        total = 0.0
        observations = 0
        for t in range(T):
            degree = 0
            returning = 0
            for j in range(N):
                if j != i and adjacency[t, i, j] > 0:
                    degree += 1
                    if last_seen[j] >= 0 and t - last_seen[j] <= delta:
                        returning += 1
                    last_seen[j] = t
            if degree:
                if has_past:
                    total += returning / degree
                    observations += 1
                has_past = True
        if observations:
            scores[i] = total / observations
    return scores


def ComputeTemporalMutualInformation(adjacency_matrices):
    """
    Compute average normalized mutual information between consecutive adjacency snapshots, optimized for symmetric (undirected) networks.

    Args:
        adjacency_matrices: array, shape (T, N, N).
        Sequence of binary adjacency matrices (0/1).

    Returns:
        avg_mutual_information: float.
        Mean normalized mutual information between consecutive snapshots.
    """
    T, N, _ = adjacency_matrices.shape

    # Precompute indices of upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)

    # Flattened views
    flattened_snapshots = [A[triu_indices] for A in adjacency_matrices]

    nmis = []

    for t in range(T - 1):
        nmi = normalized_mutual_info_score(
            flattened_snapshots[t],
            flattened_snapshots[t + 1],
        )
        nmis.append(nmi)

    if nmis:
        avg_mutual_information = np.mean(nmis)
    else:
        avg_mutual_information = np.nan

    return avg_mutual_information


def ComputeDynamism(tnet):
    """
    Summarize global temporal “dynamism” using entropy, transitions, and similarity.
    Given a binary temporal network `tnet` (T, N, N), this computes: 1) transition_probability, 2) global_entropy, 3) cosine_similarity, 4) net_fluidity, 5) mutual_information, and 6) mean_edge_count.

    Notes:
        - Uses lower-triangular edges (i<j) for all edgewise computations.
        - Entropy term gEnt excludes edges that are always 0 or always 1 to avoid log(0).
        - Entropy normalization uses: maxH = -log(0.5) * [N(N-1)/2].
        - Cosine similarity is computed on flattened lower-triangular edge vectors; rows
        with zero norm (empty graphs) are handled so that sim(t,t+1)=1 when both are empty, and rows involving an empty snapshot are set to 0 elsewhere to avoid spurious effects.

    Args:
        tnet: np.ndarray, shape (T, N, N). Binary temporal adjacency (symmetric snapshots expected).

    Returns:
        transition_probability: float. Estimated edge flip rate across consecutive snapshots.
        global_entropy: float. Normalized global entropy of edge activation probabilities.
        cosine_similarity: float. Mean cosine similarity between successive edge-state vectors.
        net_fluidity: float. Combined dynamism index `global_entropy * (1 - cosine_similarity)`.
        mutual_information: float. Average normalized mutual information across consecutive snapshots.
        mean_edge_count: float. Average undirected edge count per snapshot.

    Notes:
        Requires `ComputeTemporalMutualInformation(tnet)` to be available in scope.
    """
    T, N, _ = tnet.shape

    # Average undirected edges per snapshot (exclude diagonal)
    mean_edge_count = (np.sum(tnet) - N * T) / (2 * T)

    # Extract lower-triangular edges as time-by-edge matrix
    src, tgt = np.tril_indices(N, k=-1)
    tedges = tnet[:, src, tgt].astype(int)  # shape: (T, n_edges)

    # -----------------------------
    # Transition probability p_trans
    # -----------------------------
    if T <= 1:
        transition_probability = 0.0
    else:
        dtnet = tedges[:-1, :] - tedges[1:, :]
        denom = (T - 1) * N * (N - 1)
        changes = np.count_nonzero(dtnet)
        transition_probability = 2.0 * changes / denom if denom > 0 else 0.0

    # -----------------------------
    # Global dynamism via entropy gEnt
    # -----------------------------
    # Edge activation probabilities across time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p_edges = tedges.mean(axis=0)
    mask = (p_edges > 0) & (p_edges < 1)
    p_sel = p_edges[mask]

    if p_sel.size == 0:
        global_entropy = 0.0
    else:
        gH = -(p_sel * np.log(p_sel) + (1 - p_sel) * np.log(1 - p_sel))
        maxH = -np.log(0.5) * (N * (N - 1) / 2.0)
        global_entropy = float(np.sum(gH) / maxH) if maxH > 0 else 0.0

    # -----------------------------
    # Successive similarity (cosine) mnSim
    # -----------------------------
    norms = np.linalg.norm(tedges, axis=1, keepdims=True)
    normalized = np.divide(tedges, norms, where=norms != 0)
    csim = normalized @ normalized.T

    zero_norm = (norms == 0).flatten()
    csim[np.outer(zero_norm, zero_norm)] = 1.0
    csim[np.ix_(zero_norm, ~zero_norm)] = 0.0
    csim[np.ix_(~zero_norm, zero_norm)] = 0.0

    if T <= 1:
        cosine_similarity = np.nan
    else:
        sim_seq = np.diag(csim, k=1)
        cosine_similarity = float(np.mean(sim_seq)) if sim_seq.size > 0 else np.nan

    net_fluidity = float(
        global_entropy
        * (1.0 - (cosine_similarity if np.isfinite(cosine_similarity) else 0.0))
    )
    mutual_information = ComputeTemporalMutualInformation(tnet)
    return [
        float(transition_probability),
        float(global_entropy),
        float(cosine_similarity),
        float(net_fluidity),
        float(mutual_information),
        float(mean_edge_count),
    ]


def ComputeTemporalEdgeOverlap(adjacency_matrices):
    """
    Average Jaccard overlap of edge sets between consecutive snapshots.
    For symmetric binary matrices with ones on the diagonal. Only i<j edges are used.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Binary temporal adjacency.

    Returns:
        avg_jaccard: float. Mean Jaccard similarity across consecutive pairs (t, t+1). NaN if T < 2 or all unions are empty.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices, undirected=True)
    return float(EdgeOverlapAndPersistence(adjacency_matrices)[0])


def ComputeEdgePersistenceRate(adjacency_matrices):
    """
    Fraction of active edges at time t that remain active at time t+1, averaged over t.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Binary temporal adjacency.

    Returns:
        avg_persist: float. Mean persistence fraction over source frames t < T-1 with at least one edge. NaN if there are no eligible frames. Frames, not individual edges, receive equal weight.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices, undirected=True)
    return float(EdgeOverlapAndPersistence(adjacency_matrices)[1])


def ComputeNeighborhoodMemory(adjacency_matrices, lag=1, summary_stat=None):
    """
    Compute neighborhood memory for each node via Jaccard similarity of neighbor sets.
    For each node i and snapshot t, compare its neighbors at t with its neighbors at t+lag. High memory indicates that a node tends to retain the same partners across time separated by the lag.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.
        lag: int, optional (default=1). Temporal lag between snapshots to compare (1 = consecutive snapshots).
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.

    Notes:
        - Node i’s score is the mean Jaccard similarity across all valid (t, t+lag) pairs.
        - If a node has no neighbors at both t and t+lag, that pair is skipped.
        - Nodes with no valid pairs get memory = 0.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices)
    lag = PositiveInteger(lag, "lag")
    memory_scores = NeighborhoodMemoryScores(adjacency_matrices, lag)
    return _summarize_scores(memory_scores, summary_stat=summary_stat)


def ComputeRetention(adjacency_matrices, summary_stat=None, *, delta=None):
    """
    Compute node-wise retention in a temporal network.
    The instantaneous score is the fraction of current neighbors encountered
    previously. With delta=None, all earlier frames are eligible.
    With positive integer delta, only frames max(0,t-delta)
    through t-1 are eligible. A node's score averages eligible nonempty frames.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].
        delta: positive int or None, optional (default=None). Lookback lifetime in adjacency frames; None retains unlimited history. delta=1 compares against the immediately preceding frame.

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.

    Notes:
        - The first nonempty neighbor observation is excluded.
        - Empty current neighborhoods are skipped. After a node's first
        nonempty observation, a nonempty frame with no eligible recent partners
        contributes zero, even when all previous contacts have expired.
        - Nodes with fewer than two nonempty observations have NaN scores.
        Self-connections are excluded. Current-frame contacts cannot count as
        their own history; each partner's lifetime resets when it is observed.
        - delta counts adjacency frames, not FC window length, hop substeps or
        elapsed physical time. There is no wrapping at recording boundaries.
        - delta=None uses unlimited history; a positive delta restricts the lookback lifetime.
        Both cached compiled modes take O(T*N^2) time and O(N) workspace beyond
        the input/output; finite windows do not require copying past frames.
    """
    if delta is None:
        retention_scores = _RetentionScores(adjacency_matrices)
    else:
        delta = PositiveInteger(delta, "delta")
        # Larger lifetimes are equivalent to full history; bound the integer
        # passed to the compiled kernel even for arbitrarily large Python ints.
        delta = min(delta, adjacency_matrices.shape[0])
        retention_scores = _WindowedRetentionScores(adjacency_matrices, delta)
    return _summarize_scores(retention_scores, summary_stat=summary_stat)

def ComputeLinkBurstiness(adjacency_matrices, *, event_mode="activations"):
    """
    Compute burstiness index B = (σ - μ) / (σ + μ) per link over time.

    Args:
        adjacency_matrices: array-like, shape (T, N, N). Sequence of binary (0/1) adjacency matrices for an undirected network. Diagonal is ignored if present.
        event_mode: {"activations", "onsets"}, optional (default="activations"). Count every active frame, or only the start of each uninterrupted active run.

    Returns:
        link_burstiness: np.ndarray, shape (N, N). Burstiness per link; NaN where a link has fewer than 3 events in the selected mode. Diagonal is NaN.

    Notes:
        - Uses population standard deviation of the observed inter-event gaps.
        Regular intervals give -1; positive values indicate variable gaps.
        A sustained link gives -1 in activation mode, but only one event (NaN)
        in onset mode. These definitions answer different questions.
        - The uncorrected burstiness statistic is sample-size dependent; values
        based on only a few gaps should not be treated as reliable inference.
        Recording-boundary gaps are excluded; no periodic wrapping is assumed.
        - Streaming compiled accumulation uses O(N^2) workspace, no copied
        T-by-N-by-N binary tensor or per-link event lists. Input is never modified.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices, undirected=True)
    if event_mode not in {"activations", "onsets"}:
        raise ValueError("event_mode must be 'activations' or 'onsets'.")
    return LinkBurstinessScores(adjacency_matrices, event_mode == "onsets")


# Alternative function name supported as an alias.
LinkBurstiness = ComputeLinkBurstiness

# Alternative function names supported as aliases.
ComputeReturnability = ComputeRetention
_ReturnabilityScores = _RetentionScores
_WindowedReturnabilityScores = _WindowedRetentionScores
