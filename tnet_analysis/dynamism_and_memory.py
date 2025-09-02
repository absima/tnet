import numpy as np
import warnings
from scipy.stats import norm
from sklearn.metrics import normalized_mutual_info_score



def ComputeTemporalMutualInformation(adjacency_matrices):
    """
    Compute average normalized mutual information between consecutive adjacency snapshots,
    optimized for symmetric (undirected) networks.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
    Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    avg_mutual_information : float
    Mean normalized mutual information between consecutive snapshots.
    """
    T, N, _ = adjacency_matrices.shape

    # Precompute indices of upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)

    # Flattened views
    flattened_snapshots = [A[triu_indices] for A in adjacency_matrices]

    nmis = []

    for t in range(T-1):
        nmi = normalized_mutual_info_score(flattened_snapshots[t], flattened_snapshots[t+1])
        nmis.append(nmi)

    if nmis:
        avg_mutual_information = np.mean(nmis)
    else:
        avg_mutual_information = np.nan

    return avg_mutual_information  
    
def ComputeDynamism(tnet):
    """
    Summarize global temporal “dynamism” using entropy, transitions, and similarity.

    Given a binary temporal network `tnet` (T, N, N), this computes:
      1) p_trans : Transition probability of edges across consecutive snapshots.
      2) gEnt    : Global edge-activation entropy, normalized by max possible entropy
                   assuming all N(N-1)/2 undirected edges are eligible.
      3) mnSim   : Mean cosine similarity between successive flattened edge states.
      4) dynH    : Combined dynamism index = gEnt * (1 - mnSim).
      5) mi      : Average normalized mutual information between consecutive snapshots
                   (via ComputeTemporalMutualInformation).
      6) nE_snap : Average number of undirected edges per snapshot.

    Conventions / Notes
    -------------------
    - Uses lower-triangular edges (i<j) for all edgewise computations.
    - Entropy term gEnt excludes edges that are always 0 or always 1 to avoid log(0).
    - Entropy normalization uses: maxH = -log(0.5) * [N(N-1)/2] (matches original code).
    - Cosine similarity is computed on flattened lower-triangular edge vectors; rows
      with zero norm (empty graphs) are handled so that sim(t,t+1)=1 when both are empty,
      and rows involving an empty snapshot are set to 0 elsewhere to avoid spurious effects.

    Parameters
    ----------
    tnet : np.ndarray, shape (T, N, N)
        Binary temporal adjacency (symmetric snapshots expected).

    Returns
    -------
    p_trans : float
        Estimated transition probability across consecutive snapshots.
    gEnt : float
        Normalized global entropy of edge activation probabilities.
    mnSim : float
        Mean cosine similarity between successive edge-state vectors.
    dynH : float
        Combined dynamism index = gEnt * (1 - mnSim).
    mi : float
        Average normalized mutual information across consecutive snapshots.
    nE_snap : float
        Average undirected edge count per snapshot.

    Dependencies
    ------------
    Requires `ComputeTemporalMutualInformation(tnet)` to be available in scope.
    """
    T, N, _ = tnet.shape

    # Average undirected edges per snapshot (exclude diagonal)
    nE_snap = (np.sum(tnet) - N * T) / (2 * T)

    # Extract lower-triangular edges as time-by-edge matrix
    src, tgt = np.tril_indices(N, k=-1)
    tedges = tnet[:, src, tgt].astype(int)  # shape: (T, n_edges)

    # -----------------------------
    # Transition probability p_trans
    # -----------------------------
    if T <= 1:
        p_trans = 0.0
    else:
        dtnet = tedges[:-1, :] - tedges[1:, :]             # changes across time
        denom = (T - 1) * N * (N - 1)                      # equals 2 * (T-1) * n_edges
        changes = np.count_nonzero(dtnet)
        p_trans = 2.0 * changes / denom if denom > 0 else 0.0

    # -----------------------------
    # Global dynamism via entropy gEnt
    # -----------------------------
    # Edge activation probabilities across time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p_edges = tedges.mean(axis=0)                      # in [0,1]
    # Keep only edges that are neither always 0 nor always 1
    mask = (p_edges > 0) & (p_edges < 1)
    p_sel = p_edges[mask]

    if p_sel.size == 0:
        gEnt = 0.0
    else:
        # Binary entropy per selected edge
        gH = -(p_sel * np.log(p_sel) + (1 - p_sel) * np.log(1 - p_sel))
        # Normalization (matches original): max at p=0.5 for ALL undirected pairs
        maxH = -np.log(0.5) * (N * (N - 1) / 2.0)
        gEnt = float(np.sum(gH) / maxH) if maxH > 0 else 0.0

    # -----------------------------
    # Successive similarity (cosine) mnSim
    # -----------------------------
    # Norms per time row
    norms = np.linalg.norm(tedges, axis=1, keepdims=True)
    # Safe normalization
    normalized = np.divide(tedges, norms, where=norms != 0)
    csim = normalized @ normalized.T

    # Handle rows with zero norm (empty snapshot)
    zero_norm = (norms == 0).flatten()
    # If both snapshots empty => cosine = 1 on their mutual pair
    csim[np.outer(zero_norm, zero_norm)] = 1.0
    # Any pair involving an empty snapshot is set to 0 elsewhere
    csim[np.ix_(zero_norm, ~zero_norm)] = 0.0
    csim[np.ix_(~zero_norm, zero_norm)] = 0.0

    # First superdiagonal: similarity between successive snapshots
    if T <= 1:
        mnSim = np.nan
    else:
        sim_seq = np.diag(csim, k=1)
        mnSim = float(np.mean(sim_seq)) if sim_seq.size > 0 else np.nan

    # Combined index
    dynH = float(gEnt * (1.0 - (mnSim if np.isfinite(mnSim) else 0.0)))

    # -----------------------------
    # Temporal mutual information
    # -----------------------------
    mi = ComputeTemporalMutualInformation(tnet)

    return [float(p_trans), float(gEnt), float(mnSim), float(dynH), float(mi), float(nE_snap)]
 
 

def ComputeTemporalEdgeOverlap(adjacency_matrices):
    """
    Average Jaccard overlap of edge sets between consecutive snapshots.

    For symmetric binary matrices with ones on the diagonal. Only i<j edges are used.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Binary temporal adjacency.

    Returns
    -------
    avg_jaccard : float
        Mean Jaccard similarity across consecutive pairs (t, t+1).
        NaN if T < 2 or all unions are empty.
    """
    T, N, _ = adjacency_matrices.shape
    if T < 2:
        return np.nan

    iu, ju = np.triu_indices(N, k=1)
    jaccards = []

    for t in range(T - 1):
        a = adjacency_matrices[t][iu, ju] > 0
        b = adjacency_matrices[t + 1][iu, ju] > 0
        inter = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)
        if union == 0:
            continue
        jaccards.append(inter / union)

    return np.mean(jaccards) if len(jaccards) else np.nan



def ComputeEdgePersistenceRate(adjacency_matrices):
    """
    Fraction of active edges at time t that remain active at time t+1, averaged over t.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Binary temporal adjacency.

    Returns
    -------
    avg_persist : float
        Mean persistence fraction over t where at least one edge existed.
        NaN if T < 2 or no edges ever exist at any t.
    """
    T, N, _ = adjacency_matrices.shape
    if T < 2:
        return np.nan

    iu, ju = np.triu_indices(N, k=1)
    vals = []

    for t in range(T - 1):
        a = adjacency_matrices[t][iu, ju] > 0
        b = adjacency_matrices[t + 1][iu, ju] > 0
        active_t = np.count_nonzero(a)
        if active_t == 0:
            continue
        persist = np.count_nonzero(a & b) / active_t
        vals.append(persist)

    return np.mean(vals) if len(vals) else np.nan
    
    




def ComputeNeighborhoodMemory(adjacency_matrices, lag=1):
    """
    Compute neighborhood memory for each node via Jaccard similarity of neighbor sets.

    For each node i and snapshot t, compare its neighbors at t with its neighbors
    at t+lag. High memory indicates that a node tends to retain the same partners
    across time separated by the lag.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).
        Symmetric, diagonal ignored.
    lag : int, optional (default=1)
        Temporal lag between snapshots to compare (1 = consecutive snapshots).

    Returns
    -------
    score_percentiles : np.ndarray, shape (5,)
        [5, 25, 50, 75, 95] percentiles of node neighborhood memory scores (0–1).

    Notes
    -----
    - Node i’s score is the mean Jaccard similarity across all valid (t, t+lag) pairs.
    - If a node has no neighbors at both t and t+lag, that pair is skipped.
    - Nodes with no valid pairs get memory = 0.
    """
    T, N, _ = adjacency_matrices.shape
    memory_scores = np.zeros(N)

    for i in range(N):
        overlaps = []
        for t in range(T - lag):
            neighbors_t = set(np.where(adjacency_matrices[t, i] > 0)[0])
            neighbors_tlag = set(np.where(adjacency_matrices[t + lag, i] > 0)[0])

            if neighbors_t or neighbors_tlag:  # at least one neighbor present
                intersection = neighbors_t.intersection(neighbors_tlag)
                union = neighbors_t.union(neighbors_tlag)
                jaccard_similarity = len(intersection) / len(union)
                overlaps.append(jaccard_similarity)

        memory_scores[i] = np.mean(overlaps) if overlaps else 0.0

    score_percentiles = np.nanpercentile(memory_scores, [5, 25, 50, 75, 95])
    return score_percentiles.tolist()



def ComputeReturnability(adjacency_matrices):
    """
    Compute node-wise returnability in a temporal network.

    For each node i and time t, define:
      - neighbors_t = current neighbors of i (excluding self).
      - past_neighbors = union of all neighbors of i seen before t.
    The instantaneous returnability at time t is:
      |neighbors_t ∩ past_neighbors| / |neighbors_t|, when neighbors_t ≠ ∅.
    The node’s score is the mean of these values across time.

    Parameters
    ----------
    adjacency_matrices : np.ndarray, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1). Symmetric, diagonal ignored.

    Returns
    -------
    score_percentiles : np.ndarray, shape (5,)
        [5, 25, 50, 75, 95] percentiles of node returnability scores (0–1).
    """
    T, N, _ = adjacency_matrices.shape
    returnability_scores = np.full(N, np.nan)

    for i in range(N):
        past_neighbors = set()
        vals = []

        for t in range(T):
            neighbors_t = set(np.where(adjacency_matrices[t, i] > 0)[0])
            neighbors_t.discard(i)  # exclude self

            if not neighbors_t:
                continue

            if past_neighbors:
                old_friends = neighbors_t & past_neighbors
                vals.append(len(old_friends) / len(neighbors_t))

            past_neighbors |= neighbors_t

        if vals:
            returnability_scores[i] = float(np.mean(vals))

    score_percentiles = np.nanpercentile(returnability_scores, [5, 25, 50, 75, 95])
    return score_percentiles.tolist()

def LinkBurstiness(adjacency_matrices):
    """
    Compute burstiness index B = (σ - μ) / (σ + μ) per link over time.

    Parameters
    ----------
    adjacency_matrices : array-like, shape (T, N, N)
        Sequence of binary (0/1) adjacency matrices for an undirected network.
        Diagonal is ignored if present.

    Returns
    -------
    link_burstiness : np.ndarray, shape (N, N)
        Burstiness per link; NaN where a link has fewer than 3 activations.

    Notes
    -----
    - Complexity is proportional to the number of *active* links with ≥3 events,
      not to N^2. This is typically much faster than triple nesting.
    - If your data are extremely sparse across time, consider using the
      `LinkBurstinessSparse` version below for further speed/memory wins.
    """
    A = np.asarray(adjacency_matrices, dtype=bool)  # (T, N, N)
    T, N, _ = A.shape
    # Zero diagonal just in case
    if N == A.shape[2]:
        A[:, np.arange(N), np.arange(N)] = False

    # Count activations per link and select only those with >= 3 events (need 2 gaps)
    counts = A.sum(axis=0)                      # (N, N)
    mask = (counts >= 3)
    # Upper triangle indices of links to process
    iu, ju = np.triu_indices(N, k=1)
    sel = mask[iu, ju]
    iu, ju = iu[sel], ju[sel]

    # Flatten time dimension to (T, N*N) to access columns quickly
    A2 = A.reshape(T, N * N)
    cols = (iu * N + ju)

    out = np.full((N, N), np.nan, dtype=float)

    # Process only selected columns 
    for c, i, j in zip(cols, iu, ju):
        times = np.flatnonzero(A2[:, c])
        # times.size >= 3 by construction
        gaps = np.diff(times)
        mu = gaps.mean()
        sigma = gaps.std(ddof=0)
        denom = (sigma + mu)
        if denom > 0:
            b = (sigma - mu) / denom
            out[i, j] = b
            out[j, i] = b  # symmetric

    return out

     
     