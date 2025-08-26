import numpy as np
import warnings
from scipy.stats import norm


def ComputeStaticClustering(adjacency_matrices):
    """
    Compute average static clustering coefficient per node across time.

    For each snapshot, the clustering coefficient of node i is defined as the
    fraction of realized links among i's neighbors out of the maximum possible.
    Final score per node is the time-average across snapshots.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.

    Returns
    -------
    score_percentiles : np.ndarray, shape (5,)
        [5, 25, 50, 75, 95] percentiles of node static clustering scores (0–1).

    Notes
    -----
    - Nodes with degree < 2 get clustering = 0 for that snapshot.
    - This is the time-averaged version of the classic Watts–Strogatz clustering coefficient.
    """
    T, N, _ = adjacency_matrices.shape
    clustering_trajectories = np.zeros((T, N))

    for t in range(T):
        adj = adjacency_matrices[t]
        for i in range(N):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k >= 2:
                links = 0
                for idx1, u in enumerate(neighbors):
                    for v in neighbors[idx1 + 1:]:
                        if adj[u, v] > 0:
                            links += 1
                max_links = k * (k - 1) / 2
                clustering_trajectories[t, i] = links / max_links
            else:
                clustering_trajectories[t, i] = 0.0

    clustering_scores = np.mean(clustering_trajectories, axis=0)
    score_percentiles = np.nanpercentile(clustering_scores, [5, 25, 50, 75, 95])
    return score_percentiles.tolist()


def ComputeTemporalClustering(adjacency_matrices, delta=1):
    """
    Compute temporal clustering coefficient for each node based on time-respecting triangles.

    A temporal triangle for node i is formed if:
      - i–j is active at time t1,
      - i–k is active at time t2 with t1 < t2 ≤ t1+delta,
      - and j–k is connected at some time t3 with t1 < t3 ≤ t2.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.
    delta : int, optional (default=1)
        Maximum temporal gap (in snapshots) allowed between edges to form a temporal triangle.

    Returns
    -------
    score_percentiles : np.ndarray, shape (5,)
        [5, 25, 50, 75, 95] percentiles of node temporal clustering scores (0–1).

    Notes
    -----
    - Nodes with no valid triplets get clustering = 0.
    - delta=1 reduces to checking consecutive snapshots only.
    - Complexity is O(T * N * k^2 * delta) per node, where k is degree.
    """
    T, N, _ = adjacency_matrices.shape
    triangle_counts = np.zeros(N)
    triplet_counts = np.zeros(N)

    for t1 in range(T):
        adj1 = adjacency_matrices[t1]
        for i in range(N):
            neighbors1 = np.where(adj1[i] > 0)[0]
            for j in neighbors1:
                for t2 in range(t1 + 1, min(t1 + delta + 1, T)):
                    adj2 = adjacency_matrices[t2]
                    neighbors2 = np.where(adj2[i] > 0)[0]
                    for k in neighbors2:
                        if k == j:
                            continue
                        triplet_counts[i] += 1
                        # Check j–k connection in window [t1+1, t2]
                        for t3 in range(t1 + 1, t2 + 1):
                            adj3 = adjacency_matrices[t3]
                            if adj3[j, k] > 0:
                                triangle_counts[i] += 1
                                break

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        temporal_clustering = np.where(triplet_counts > 0,
                                       triangle_counts / triplet_counts,
                                       0.0)

    score_percentiles = np.nanpercentile(temporal_clustering, [5, 25, 50, 75, 95])
    return score_percentiles.tolist()
    
    
def ComputeSnapshotTransitivity(adj_matrix):
    """
    Global transitivity (clustering) for an undirected simple graph snapshot.

    Assumes binary symmetric adjacency with zero diagonal.
    Triangles = trace(A^3)/6; connected triplets = sum_i C(deg_i, 2).

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Binary symmetric adjacency with zero diagonal.

    Returns
    -------
    transitivity : float
        3 * (#triangles) / (#connected triplets). Returns 0 if no triplets exist.
    """
    A = adj_matrix.astype(int)
    np.fill_diagonal(A, 0)
    deg = A.sum(axis=1)
    triplets = np.sum(deg * (deg - 1)) / 2
    if triplets == 0:
        return 0.0
    # triangles = trace(A^3)/6
    A2 = A @ A
    A3 = A2 @ A
    triangles = np.trace(A3) / 6.0
    return float(3.0 * triangles / triplets)


def ComputeTemporalTransitivity(adjacency_matrices):
    """
    Average snapshot transitivity (global clustering) over time.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Binary temporal adjacency.

    Returns
    -------
    avg_transitivity : float
        Mean transitivity across snapshots (ignores NaNs).
    """
    Ts = []
    for A in adjacency_matrices:
        Ts.append(ComputeSnapshotTransitivity(A))
    return float(np.mean(Ts)) if len(Ts) else np.nan


def ComputeSnapshotModularity(adj_matrix, labels):
    """
    Newman-Girvan modularity Q for an undirected simple graph snapshot.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Binary (or weighted) symmetric adjacency. Diagonal is ignored.
    labels : array-like, shape (N,)
        Community labels per node.

    Returns
    -------
    Q : float
        Modularity value in [-0.5, 1) (typical ranges), depending on structure.

    Notes
    -----
    Q = (1/2m) sum_{ij} [A_ij - (k_i k_j)/(2m)] * delta(c_i, c_j)
    Works for weighted A as well (k_i := sum_j A_ij, m := sum_ij A_ij / 2).
    """
    A = adj_matrix.astype(float).copy()
    np.fill_diagonal(A, 0.0)
    k = A.sum(axis=1)
    m = k.sum() / 2.0
    if m == 0:
        return 0.0

    labels = np.asarray(labels)
    same_com = (labels[:, None] == labels[None, :]).astype(float)

    expected = np.outer(k, k) / (2.0 * m)
    B = A - expected
    Q = (B * same_com).sum() / (2.0 * m)
    return float(Q)


def ComputeTemporalModularity(adjacency_matrices, labels_per_snapshot):
    """
    Average modularity across time given community labels at each snapshot.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Binary or weighted symmetric adjacency per snapshot.
    labels_per_snapshot : np.ndarray, shape (T, N)
        Community labels per node at each snapshot (can vary over time).

    Returns
    -------
    avg_Q : float
        Mean modularity across snapshots.
    Qs : np.ndarray, shape (T,)
        Per-snapshot modularity values.
    """
    T = adjacency_matrices.shape[0]
    Qs = np.zeros(T, dtype=float)
    for t in range(T):
        Qs[t] = ComputeSnapshotModularity(adjacency_matrices[t], labels_per_snapshot[t])
    return float(np.mean(Qs)), Qs


def ComputeSnapshotParticipationCoefficient(adj_matrix, labels, eps=1e-12):
    """
    Participation coefficient (network segregation/integration per node).

    For node i: P_i = 1 - sum_c (k_i^c / k_i)^2, where k_i^c is strength/degree
    of i to community c. Returns NaN for isolated nodes (k_i = 0).

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Binary or weighted symmetric adjacency (diag ignored).
    labels : array-like, shape (N,)
        Community labels per node.
    eps : float, optional (default=1e-12)
        Numerical guard to avoid division-by-zero issues.

    Returns
    -------
    P : np.ndarray, shape (N,)
        Participation coefficient per node (NaN for isolated nodes).
    """
    A = adj_matrix.astype(float).copy()
    np.fill_diagonal(A, 0.0)
    labels = np.asarray(labels)
    n = A.shape[0]

    k = A.sum(axis=1)
    P = np.full(n, np.nan, dtype=float)

    # group mask per community
    comms = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    for i in range(n):
        ki = k[i]
        if ki <= eps:
            continue
        frac_sq = 0.0
        for c, idx in comms.items():
            ki_c = A[i, idx].sum()
            frac = ki_c / ki
            frac_sq += frac * frac
        P[i] = 1.0 - frac_sq
    return P


def ComputeTemporalParticipationCoefficient(adjacency_matrices, labels_per_snapshot):
    """
    Average participation coefficient over time.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
    labels_per_snapshot : np.ndarray, shape (T, N)

    Returns
    -------
    avg_participation : float
        Mean of nodewise participation coefficients averaged across time
        (ignores NaNs for isolated nodes).
    P_all : np.ndarray, shape (T, N)
        Per-snapshot nodewise participation coefficients.
    """
    T, N, _ = adjacency_matrices.shape
    P_all = np.full((T, N), np.nan, dtype=float)
    for t in range(T):
        P_all[t] = ComputeSnapshotParticipationCoefficient(
            adjacency_matrices[t], labels_per_snapshot[t]
        )
    return float(np.nanmean(P_all)), P_all





def ComputePartnerStability(adjacency_matrices):
    """
    Compute partner stability for each node, based on recurrence of connections over time.

    Stability is higher when neighbor interactions recur more frequently, i.e.,
    when the mean temporal gap between repeated connections is smaller.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).
        Symmetric, diagonal ignored.

    Returns
    -------
    score_percentiles : np.ndarray, shape (5,)
        [5, 25, 50, 75, 95] percentiles of node stability scores (0–1).

    Notes
    -----
    - Node i’s score is computed as 1 - mean_gap/T, where mean_gap is the average
      interval (in snapshots) between consecutive interactions with the same neighbor.
    - Nodes with no recurring neighbors get stability = 0.
    """
    T, N, _ = adjacency_matrices.shape
    stability_scores = np.zeros(N)

    for i in range(N):
        gaps_all = []
        for j in range(N):
            if i == j:
                continue
            times_connected = np.where(adjacency_matrices[:, i, j] > 0)[0]
            if len(times_connected) > 1:
                gaps_all.extend(np.diff(times_connected))
        if gaps_all:
            mean_gap = np.mean(gaps_all)
            stability_scores[i] = 1 - (mean_gap / T)
        else:
            stability_scores[i] = 0.0

    score_percentiles = np.nanpercentile(stability_scores, [5, 25, 50, 75, 95])
    return score_percentiles.tolist()


def ComputePartnerDiversity(adjacency_matrices, window_size=None):
    """
    Compute partner diversity for each node using normalized entropy across temporal windows.

    Diversity is higher when a node’s connections are distributed across many
    partners and many time windows, rather than concentrated in a few.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).
        Symmetric, diagonal ignored.
    window_size : int, optional (default=10)
        Temporal window size for grouping interactions.

    Returns
    -------
    score_percentiles : np.ndarray, shape (5,)
        [5, 25, 50, 75, 95] percentiles of node diversity scores (0–1).

    Notes
    -----
    - For each neighbor j of node i, we collect the set of windows where i–j
      connections occurred. Temporal span per neighbor is normalized by total windows.
    - Diversity is then computed as the entropy of these spans, normalized by
      log(# of distinct partners).
    - Nodes with no partners get diversity = 0.
    """
    if window_size is None or window_size <= 0:
        window_size = 1
    
    T, N, _ = adjacency_matrices.shape
    diversity_scores = np.zeros(N, dtype=float)
    n_windows = T // window_size if window_size > 0 else 1

    for i in range(N):
        neighbor_time_activity = {}
        for t in range(T):
            neighbors = np.where(adjacency_matrices[t, i] > 0)[0]
            neighbors = neighbors[neighbors != i]
            if neighbors.size == 0:
                continue
                
            w = t // window_size     
            for j in neighbors:
                if j not in neighbor_time_activity:
                    neighbor_time_activity[j] = set()
                neighbor_time_activity[j].add(w)

        if neighbor_time_activity:
            temporal_spans = np.array([
                len(windows) / n_windows for windows in neighbor_time_activity.values()],
                dtype=float)
            total = np.sum(temporal_spans)
            if total <= 0:
                diversity_scores[i] = 0.0
                continue
            probs = temporal_spans / total
            probs = probs[probs > 0]
            if probs.size == 0:
                diversity_scores[i] = 0.0
                continue
            entropy = -np.sum(probs * np.log(probs))
            norm = np.log(len(probs))
            diversity_scores[i] = entropy / norm if norm > 0 else np.nan
        else:
            diversity_scores[i] = 0.0

    score_percentiles = np.nanpercentile(diversity_scores, [5, 25, 50, 75, 95])
    return score_percentiles.tolist()




    
        
def ComputeNodePersistence(tnet, confidence=0.95):
     """
     Compute node-level persistence scores and identify significantly persistent nodes.

     For each node i, persistence is defined as the mean probability that i is
     connected to its ever-connected neighbors across time. A node is deemed
     'significantly persistent' if its persistence score exceeds a theoretical
     chance threshold derived from link density and sampling variance.

     Parameters
     ----------
     tnet : np.ndarray, shape (T, N, N)
         Temporal adjacency matrices (binary or weighted). Symmetric assumed.
     confidence : float, optional (default=0.95)
         Confidence level (e.g., 0.95, 0.99) used to set the theoretical
         threshold for persistence.

     Returns
     -------
     score_percentiles : np.ndarray, shape (5,)
         [5, 25, 50, 75, 95] percentiles of node persistence scores.
     n_persistent_nodes : int
         Number of nodes exceeding the theoretical persistence threshold.

     Notes
     -----
     - Weighted inputs are binarized (>0 → 1).
     - Node i’s score is the average persistence probability across all
       neighbors that ever connected to i.
     - Theoretical threshold is computed from the mean temporal edge density,
       with binomial standard error scaled by z_(confidence).
     """
     def ComputeTheoreticalThreshold(adj_matrices, confidence):
         """Helper: compute significance threshold for persistence scores."""
         T, N, _ = adj_matrices.shape
         k = norm.ppf(1 - (1 - confidence) / 2.0)

         link_counts = np.sum(adj_matrices > 0, axis=(1, 2))  # total edges per snapshot
         max_links = N * (N - 1) / 2.0  # max undirected links

         p_t = link_counts / max_links
         p_avg = np.mean(p_t)

         std_persistence = np.sqrt(p_avg * (1 - p_avg) / T)
         threshold = p_avg + k * std_persistence
         return threshold

     T, N, _ = tnet.shape

     # Binarize if weighted
     binary_matrices = (tnet > 0).astype(int)

     # Persistence counts across time
     persistence_counts = np.sum(binary_matrices, axis=0)  # (N, N)
     persistence_probs = persistence_counts / T           # (N, N)

     # Node-level persistence scores
     persistence_scores = np.zeros(N)
     for i in range(N):
         ever_connected = persistence_counts[i] > 0
         if np.sum(ever_connected) == 0:
             persistence_scores[i] = np.nan
         else:
             persistence_scores[i] = np.mean(persistence_probs[i][ever_connected])

     # Theoretical threshold
     threshold = ComputeTheoreticalThreshold(tnet, confidence)

     # Significantly persistent nodes
     persistent_nodes = np.where(persistence_scores > threshold)[0]

     score_percentiles = np.nanpercentile(persistence_scores, [5, 25, 50, 75, 95])
     n_persistent_nodes = len(persistent_nodes)
     return score_percentiles.tolist(), n_persistent_nodes  
    

def SummarizeSegregationStructure(adjacency_matrices, labels_per_snapshot=None, return_details=False):
    """
    Summarize segregation/cohesion structure of a temporal network (Group 5).

    This collects structure-focused metrics (NOT dynamism):
      - Global clustering (transitivity) averaged over time
      - Nodewise static clustering (time-averaged) → percentiles
      - Nodewise temporal clustering (time-respecting triangles) → percentiles
      - Partner stability → percentiles
      - Partner diversity → percentiles
      - Node persistence vs. chance → percentiles + count of persistent nodes
      - (Optional) Modularity and Participation (if labels provided)

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Binary (or weighted where applicable) symmetric temporal adjacency.
    labels_per_snapshot : np.ndarray or None, shape (T, N), optional
        Community labels per node at each snapshot. If provided, modularity and
        participation metrics are included.
    return_details : bool, optional (default=False)
        If True, also returns per-metric raw arrays where available.

    Returns
    -------
    summary : dict
        Keys (always present):
          - 'avg_transitivity'            : float
          - 'static_clustering_pct'       : np.ndarray (5,)   # [5,25,50,75,95]
          - 'temporal_clustering_pct'     : np.ndarray (5,)
          - 'partner_stability_pct'       : np.ndarray (5,)
          - 'partner_diversity_pct'       : np.ndarray (5,)
          - 'node_persistence_pct'        : np.ndarray (5,)
          - 'n_persistent_nodes'          : int

        If labels_per_snapshot is provided:
          - 'avg_modularity'              : float
          - 'modularity_ts'               : np.ndarray (T,)
          - 'avg_participation'           : float
          - 'participation_ts'            : np.ndarray (T, N)

        If return_details is True, may also include:
          - 'temporal_clustering_scores'  : np.ndarray (N,)      # nodewise (when computed internally)
          - 'static_clustering_scores'    : np.ndarray (N,)      # nodewise
          - 'node_persistence_scores'     : np.ndarray (N,)      # nodewise

    Notes
    -----
    This summary intentionally excludes *dynamism/memory* metrics such as:
    edge overlap, edge persistence rate, neighborhood memory, returnability,
    link burstiness, or mutual information — those are covered in a separate
    Group 4 summary.
    """
    summary = {}

    # Global clustering / transitivity (temporal average)
    avg_transitivity = ComputeTemporalTransitivity(adjacency_matrices)
    summary['avg_transitivity'] = float(avg_transitivity)

    # Nodewise clustering (static & temporal) → percentiles
    static_clust_pct = ComputeStaticClustering(adjacency_matrices)
    temporal_clust_pct = ComputeTemporalClustering(adjacency_matrices)
    summary['static_clustering_pct'] = static_clust_pct
    summary['temporal_clustering_pct'] = temporal_clust_pct

    # Partner stability & diversity → percentiles
    partner_stab_pct = ComputePartnerStability(adjacency_matrices)
    partner_div_pct = ComputePartnerDiversity(adjacency_matrices)
    summary['partner_stability_pct'] = partner_stab_pct
    summary['partner_diversity_pct'] = partner_div_pct

    # Node persistence vs chance → percentiles + count
    node_persist_pct, n_persistent_nodes = ComputeNodePersistence(adjacency_matrices)
    summary['node_persistence_pct'] = node_persist_pct
    summary['n_persistent_nodes'] = int(n_persistent_nodes)

    # Optional: label-based structure metrics
    if labels_per_snapshot is not None:
        avg_Q, Qs = ComputeTemporalModularity(adjacency_matrices, labels_per_snapshot)
        avg_P, P_all = ComputeTemporalParticipationCoefficient(adjacency_matrices, labels_per_snapshot)
        summary.update({
            'avg_modularity': float(avg_Q),
            'modularity_ts': Qs,
            'avg_participation': float(avg_P),
            'participation_ts': P_all,
        })

    if return_details:
        # If you later add nodewise-return options to the primitives,
        # you can include them here (placeholders below).
        # For now, we expose only what’s naturally available without changing your APIs.
        pass

    return summary
    
    
    
    
    