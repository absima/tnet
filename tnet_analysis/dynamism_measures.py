import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

def compute_transition_probability(tnet):
    """
    Estimate global switching probability in a temporal network.

    Parameters
    ----------
    tnet : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    switch_probability : float
        Probability of edge transitions (link flips) between consecutive snapshots.
    """
    T, N, _ = tnet.shape

    src, tgt = np.tril_indices(N, -1)
    flattened_tnet = tnet[:, src, tgt]  # Extract lower triangle over time

    diff_tnet = flattened_tnet[:-1, :] - flattened_tnet[1:, :]
    
    denom = (T - 1) * N * (N - 1)

    if denom == 0:
        switch_probability = 0
    else:
        switch_probability = 2 * len(np.where(diff_tnet)[0]) / denom

    return switch_probability
    

def compute_adjusted_entropy(tnet):
    """
    Compute adjusted entropy and dynamism indices of a temporal network.

    This combines:
    - Global entropy of link dynamics
    - Mean cosine similarity between successive snapshots
    - A combined dynamism index (entropy × (1 - similarity))

    Parameters
    ----------
    tnet : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    global_entropy : float
        Normalized entropy of link activations.
    mean_similarity : float
        Mean cosine similarity between successive time points.
    dynamism_index : float
        Combined measure: entropy scaled by (1 - similarity).
    """
    T, N, _ = tnet.shape

    src, tgt = np.tril_indices(N, -1)
    flattened_edges = tnet[:, src, tgt]

    # Global entropy calculation
    p_edge = np.sum(flattened_edges, axis=0) / T
    p_edge = p_edge[(p_edge > 0) & (p_edge < 1)]  # Exclude always-off and always-on edges

    gH = -(p_edge * np.log(p_edge) + (1 - p_edge) * np.log(1 - p_edge))
    max_entropy = -np.log(0.5) * (N * (N - 1) / 2)
    global_entropy = np.sum(gH) / max_entropy

    # Cosine similarity calculation
    norms = np.linalg.norm(flattened_edges, axis=1, keepdims=True)
    normalized_edges = np.divide(flattened_edges, norms, where=norms != 0)
    cosine_sim_matrix = np.dot(normalized_edges, normalized_edges.T)

    zero_norm_mask = (norms == 0).flatten()
    cosine_sim_matrix[np.outer(zero_norm_mask, zero_norm_mask)] = 1
    cosine_sim_matrix[zero_norm_mask, :] = 0
    cosine_sim_matrix[:, zero_norm_mask] = 0

    similarity_across_time = np.diag(cosine_sim_matrix, k=1)
    mean_similarity = np.mean(similarity_across_time)

    # Combined dynamism index
    dynamism_index = global_entropy * (1 - mean_similarity)

    return global_entropy, mean_similarity, dynamism_index  
    
def compute_temporal_mutual_information(adjacency_matrices):
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
     
     