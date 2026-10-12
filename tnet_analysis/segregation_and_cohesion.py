import numpy as np
import warnings
from scipy.stats import norm
from scipy import sparse
from numba import njit
from ._summary import SummarizeScores as _summarize_scores
from ._temporal_statistics import (
    ValidateTemporalAdjacency, PositiveInteger, SnapshotAdjacency,
    SnapshotTransitivity, ParticipationScores, PartnerCountsAndStability,
    PartnerDiversityScores, ConditionalPersistencePValues,
)


@njit(cache=True)
def _ClusteringPopcount(value):
    """Count set bits in one packed neighbor word."""
    count = 0
    while value:
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=True)
def _ClusteringNeighbors(adjacency):
    """Pack positive entries without modifying the supplied diagonals."""
    T, N, _ = adjacency.shape
    packed = np.zeros((T, N, (N + 63) // 64), dtype=np.uint64)
    degrees = np.zeros((T, N), dtype=np.int64)
    for t in range(T):
        for i in range(N):
            for j in range(N):
                if adjacency[t, i, j] > 0:
                    packed[t, i, j // 64] |= np.uint64(1) << np.uint64(j % 64)
                    degrees[t, i] += 1
    return packed, degrees


@njit(cache=True)
def _SpatialClusteringScores(packed, degrees):
    """Count each neighbor pair once using packed intersections."""
    T, N, W = packed.shape
    scores = np.zeros(N)
    for t in range(T):
        for i in range(N):
            degree = degrees[t, i]
            if degree < 2:
                continue
            links = 0
            for u in range(N):
                word = u // 64
                if not (packed[t, i, word] & (np.uint64(1) << np.uint64(u % 64))):
                    continue
                # Count each unordered pair once (u < v).
                if u % 64 < 63:
                    mask = ~((np.uint64(1) << np.uint64(u % 64 + 1)) - np.uint64(1))
                    links += _ClusteringPopcount(packed[t, i, word] & packed[t, u, word] & mask)
                for w in range(word + 1, W):
                    links += _ClusteringPopcount(packed[t, i, w] & packed[t, u, w])
            scores[i] += links / (degree * (degree - 1) / 2.0)
    return scores / T


@njit(cache=True)
def _TemporalClosingWindows(packed, delta):
    """Union closing edges over delta frames, starting at each second-edge frame."""
    if delta == 1:
        return packed
    T, N, W = packed.shape
    closing = packed.copy()
    for t2 in range(T - 1):
        for t3 in range(t2 + 1, min(t2 + delta, T)):
            for j in range(N):
                for w in range(W):
                    closing[t2, j, w] |= packed[t3, j, w]
    return closing


@njit(cache=True)
def _TemporalClusteringScores(packed, degrees, delta):
    """Count triplets with delta-frame incident-edge and delayed-closure windows."""
    T, N, W = packed.shape
    triangles = np.zeros(N, dtype=np.int64)
    triplets = np.zeros(N, dtype=np.int64)
    closing = _TemporalClosingWindows(packed, delta)
    for t1 in range(T - 1):
        for t2 in range(t1 + 1, min(t1 + delta + 1, T)):
            for i in range(N):
                common = 0
                for w in range(W):
                    common += _ClusteringPopcount(packed[t1, i, w] & packed[t2, i, w])
                triplets[i] += degrees[t1, i] * degrees[t2, i] - common
                for j in range(N):
                    word = j // 64
                    bit = np.uint64(1) << np.uint64(j % 64)
                    if not (packed[t1, i, word] & bit):
                        continue
                    for w in range(W):
                        closed = packed[t2, i, w] & closing[t2, j, w]
                        if w == word:
                            closed &= ~bit
                        triangles[i] += _ClusteringPopcount(closed)
    scores = np.zeros(N)
    for i in range(N):
        if triplets[i]:
            scores[i] = triangles[i] / triplets[i]
    return scores


def ComputeSpatialClustering(adjacency_matrices, summary_stat=None):
    """
    Compute average spatial clustering coefficient per node across time.
    For each snapshot, the clustering coefficient of node i is defined as the fraction of realized links among i's neighbors out of the maximum possible. Final score per node is the time-average across snapshots.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Sequence of binary adjacency matrices (0/1). Symmetric; provide zero diagonals to exclude self-neighbors.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.

    Notes:
        - Nodes with degree < 2 get clustering = 0 for that snapshot.
        - This is the time-averaged version of the classic Watts–Strogatz clustering coefficient.
        - Cached compiled kernels count links using packed neighbor intersections.
    """
    packed, degrees = _ClusteringNeighbors(adjacency_matrices)
    clustering_scores = _SpatialClusteringScores(packed, degrees)
    return _summarize_scores(clustering_scores, summary_stat=summary_stat)


def ComputeTemporalClustering(adjacency_matrices, delta=1, summary_stat=None):
    """
    Compute temporal clustering coefficient for each node based on time-respecting triangles.
    A temporal triangle for node i is formed if: - i–j is active at time t1, - i–k is active at time t2 with t1 < t2 ≤ t1+delta, - and j–k is connected at some time t3 with t2 ≤ t3 ≤ t2+delta-1. Each temporal triplet is counted once even if its closing edge appears in multiple eligible frames.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Sequence of binary adjacency matrices (0/1). Symmetric; provide zero diagonals to exclude self-neighbors.
        delta: int, optional (default=1). Shared tolerance in frames. The second central connection may occur up to delta frames after the first; the closing edge is accepted within delta frames starting at the second connection's frame.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.

    Notes:
        - Nodes with no valid triplets get clustering = 0.
        - delta=1 uses consecutive snapshots, with closure in the second snapshot.
        - Closing edges are accepted only once both central connections have been observed.
        - Closing windows stop at the end of the series without wrapping. Eligible triplets with shorter remaining windows stay in the denominator.
        - Packed closing-edge unions are computed once per frame; intersections avoid enumerating individual triplets.
    """
    if isinstance(delta, (bool, np.bool_)) or not isinstance(delta, (int, np.integer)):
        raise TypeError("delta must be a positive integer number of frames.")
    if delta < 1:
        raise ValueError("delta must be at least 1.")
    adjacency = np.asarray(adjacency_matrices)
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2] or not all(adjacency.shape):
        raise ValueError("adjacency_matrices must have shape (T, N, N) with T, N > 0.")
    # Extra frames beyond the observed series cannot add incident or closing edges.
    delta = min(int(delta), adjacency.shape[0])
    packed, degrees = _ClusteringNeighbors(adjacency)
    temporal_clustering = _TemporalClusteringScores(packed, degrees, delta)
    return _summarize_scores(temporal_clustering, summary_stat=summary_stat)


def ComputeSnapshotTransitivity(adj_matrix):
    """
    Global transitivity (clustering) for an undirected simple graph snapshot.
    Positive weights define edges; diagonal is ignored. Triangles are counted
    by sparse neighbor intersections rather than dense matrix cubing.

    Args:
        adj_matrix: np.ndarray or scipy sparse matrix, shape (N, N). Nonnegative symmetric adjacency; positive entries define edges, diagonal ignored.

    Returns:
        transitivity: float. 3 * (#triangles) / (#connected triplets). Returns 0 if no triplets exist.
    """
    A = SnapshotAdjacency(adj_matrix)
    return float(SnapshotTransitivity(A.indptr, A.indices))


def ComputeTemporalTransitivity(adjacency_matrices):
    """
    Average snapshot transitivity (global clustering) over time.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Binary temporal adjacency.

    Returns:
        avg_transitivity: float. Mean transitivity across snapshots (ignores NaNs).
    """
    Ts = []
    for A in adjacency_matrices:
        Ts.append(ComputeSnapshotTransitivity(A))
    return float(np.mean(Ts)) if len(Ts) else np.nan


def ComputeSnapshotParticipationCoefficient(adj_matrix, labels, eps=1e-12):
    """
    Participation coefficient (network segregation/integration per node).
    For node i: P_i = 1 - sum_c (k_i^c / k_i)^2, where k_i^c is strength/degree of i to community c. Returns NaN for isolated nodes (k_i = 0).

    Args:
        adj_matrix: np.ndarray, shape (N, N). Binary or weighted symmetric adjacency (diag ignored).
        labels: array-like, shape (N,). Community labels per node.
        eps: float, optional (default=1e-12). Numerical guard to avoid division-by-zero issues.

    Returns:
        P: np.ndarray, shape (N,). Participation coefficient per node (NaN for isolated nodes).
    """
    A = SnapshotAdjacency(adj_matrix)
    labels = np.asarray(labels)
    n = A.shape[0]
    if labels.shape != (n,):
        raise ValueError("labels must have shape (N,).")
    if labels.dtype.kind in 'buifc' and not np.isfinite(labels).all():
        raise ValueError("Community labels must be finite.")
    if labels.dtype == object and any(label is None for label in labels):
        raise ValueError("Community labels cannot be missing.")
    if np.ndim(eps) != 0 or not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative.")
    communities, encoded = np.unique(labels, return_inverse=True)
    return ParticipationScores(A.indptr, A.indices, A.data,
                               encoded.astype(np.int64), len(communities), eps)


def ComputeTemporalParticipationCoefficient(adjacency_matrices, labels_per_snapshot):
    """
    Average participation coefficient over time.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N).
        labels_per_snapshot: np.ndarray, shape (T, N) or (N,). A fixed partition may be shared across frames.

    Returns:
        avg_participation: float. Mean of nodewise participation coefficients averaged across time (ignores NaNs for isolated nodes).
        P_all: np.ndarray, shape (T, N). Per-snapshot nodewise participation coefficients.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices)
    T, N, _ = adjacency_matrices.shape
    labels_per_snapshot = np.asarray(labels_per_snapshot)
    if labels_per_snapshot.shape not in {(N,), (T, N)}:
        raise ValueError("labels_per_snapshot must have shape (N,) or (T, N).")
    P_all = np.full((T, N), np.nan, dtype=float)
    for t in range(T):
        P_all[t] = ComputeSnapshotParticipationCoefficient(
            adjacency_matrices[t],
            labels_per_snapshot if labels_per_snapshot.ndim == 1 else labels_per_snapshot[t],
        )
    finite = P_all[np.isfinite(P_all)]
    return float(finite.mean()) if finite.size else np.nan, P_all


def ComputePartnerStability(adjacency_matrices, summary_stat=None, *,
                            normalization="duration"):
    """
    Compute partner stability for each node, based on recurrence of connections over time.
    Stability is higher when neighbor interactions recur more frequently, i.e., when the mean temporal gap between repeated connections is smaller.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].
        normalization: {"duration", "gap"}, optional (default="duration"). 1-mean_gap/T score, or reciprocal mean gap in frames.

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.

    Notes:
        - Node i’s score is computed as 1 - mean_gap/T, where mean_gap is the average
        interval (in snapshots) between consecutive interactions with the same neighbor.
        - Nodes with no recurring neighbors get stability = 0.
        - This is a horizon-normalized recurrence-gap index, not adjacent-frame
        edge survival. Scores from different recording lengths are not directly
        comparable. All recurrence intervals, not partners, have equal weight.
        - normalization='gap' gives 1/mean_gap, avoiding normalization by
        recording duration. It is a recurrence rate per frame, not a probability
        of next-frame edge survival; frame duration still matters.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices)
    if normalization not in {"duration", "gap"}:
        raise ValueError("normalization must be 'duration' or 'gap'.")
    _, stability_scores = PartnerCountsAndStability(adjacency_matrices, normalization == "gap")
    return _summarize_scores(stability_scores, summary_stat=summary_stat)


def ComputePartnerDiversity(adjacency_matrices, window_size=20, summary_stat=None,
                            *, normalization="observed"):
    """
    Compute partner diversity for each node using normalized entropy across temporal windows.
    The default measures evenness of partners' window coverage, not the number
    of partners or temporal ordering. Equal coverage by two or many partners
    both scores 1. Use normalization='available' to also reflect partner breadth.

    Args:
        adjacency_matrices: np.ndarray, shape (T, N, N). Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.
        window_size: int, optional (default=20). Temporal window size for grouping interactions.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].
        normalization: {"observed", "available"}, optional (default="observed"). Normalize entropy by log(observed partners) or log(N-1).

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.

    Notes:
        - For each neighbor j of node i, we collect the set of windows where i–j
        connections occurred. Temporal span per neighbor is normalized by total windows.
        - Diversity is then computed as the entropy of these spans, normalized by
        log(# of distinct partners).
        - Nodes with zero or one partner get diversity = 0.
        - A final partial window is included; windows longer than the recording
        form one window. Window coverage ignores repeated events within a window.
    """
    adjacency_matrices = ValidateTemporalAdjacency(adjacency_matrices)
    window_size = PositiveInteger(window_size, "window_size")
    if normalization not in {"observed", "available"}:
        raise ValueError("normalization must be 'observed' or 'available'.")
    diversity_scores = PartnerDiversityScores(
        adjacency_matrices, window_size, normalization == "available")
    return _summarize_scores(diversity_scores, summary_stat=summary_stat)


def ComputeNodePersistence(tnet, confidence=0.95, summary_stat=None, *,
                           null_probability=None, return_details=False):
    """
    Compute occupancy among ever-connected partners and an iid-null exceedance count.
    Persistence is the mean active-frame fraction among a node's ever-connected
    partners, excluding self-connections. This is occupancy, not edge survival.

    Args:
        tnet: np.ndarray, shape (T, N, N). Temporal adjacency matrices (binary or weighted). Symmetric assumed.
        confidence: float, optional (default=0.95). One-sided confidence level; compare conditional upper-tail probabilities with 1-confidence.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None). Return five percentiles by default, one scalar, or [mean, median].
        null_probability: float or None, optional (default=None). Independent Bernoulli edge probability. None estimates it from off-diagonal density.
        return_details: bool, optional (default=False). Append a dictionary with node scores, conditional tail probabilities and the null probability.

    Returns:
        summary: list[float] or float. Five percentiles by default, or the requested scalar summary.
        n_persistent_nodes: int. Number of nodes with conditional upper-tail probability < 1-confidence. No multiple-testing correction is applied.
        details: dict, optional. Present only when return_details=True.

    Notes:
        - Weighted inputs are binarized (>0 → 1).
        - Node i’s score is the average persistence probability across all
        neighbors that ever connected to i.
        - Conditional on K ever-active partners, their activation counts follow
        K independent Binomial(T,p) distributions conditioned on positive counts.
        Their sum's upper tail is computed by FFT convolution (not a normal approximation).
        - The null assumes homogeneous independent links and independent frames.
        Overlapping FC windows generally violate this assumption. With estimated
        p, these are plug-in model tail probabilities, NOT calibrated significance
        claims. Use independently specified p or appropriate surrogates for inference.
        - Isolated nodes have NaN scores and are never counted. Diagonal ignored.
    """

    tnet = ValidateTemporalAdjacency(tnet)
    if np.ndim(confidence) != 0 or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    T, N, _ = tnet.shape
    counts, _ = PartnerCountsAndStability(tnet)
    partners = np.count_nonzero(counts, axis=1)
    persistence_scores = np.full(N, np.nan)
    valid = partners > 0
    if T:
        persistence_scores[valid] = counts.sum(axis=1)[valid] / (T * partners[valid])
    if null_probability is None:
        null_probability = float(counts.sum() / (T * N * (N - 1))) if T and N > 1 else 0.0
    if np.ndim(null_probability) != 0 or not np.isfinite(null_probability) or not 0 <= null_probability <= 1:
        raise ValueError("null_probability must be between 0 and 1.")
    pvalues = ConditionalPersistencePValues(counts, T, float(null_probability))
    n_persistent_nodes = int(np.count_nonzero(pvalues < 1 - confidence))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = (_summarize_scores(persistence_scores, summary_stat), n_persistent_nodes)
    if return_details:
        return (*result, {"node_scores": persistence_scores,
                          "tail_probabilities": pvalues,
                          "null_probability": float(null_probability)})
    return result


# Alternative function names supported as aliases.
ComputeStaticClustering = ComputeSpatialClustering
_StaticClusteringScores = _SpatialClusteringScores
    
