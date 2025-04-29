import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score


def stable_label_propagation(A, max_iter=100):
    """
    label propagation with random visiting order and fair tie-breaking.

    Parameters
    ----------
    A : array, shape (N, N)
        Binary adjacency matrix.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    labels : array, shape (N,)
        Detected community labels.
    """
    n = A.shape[0]
    labels = np.arange(n)
    rng = np.random.default_rng(42)  # Fixed random seed for reproducibility

    for _ in range(max_iter):
        changed = False
        nodes = rng.permutation(n)  # Visit nodes in random order each round

        for i in nodes:
            neighbors = np.where(A[i])[0]
            if len(neighbors) == 0:
                continue

            neighbor_labels = labels[neighbors]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            candidate_labels = unique[counts == max_count]

            # Random tie-breaking
            new_label = rng.choice(candidate_labels)

            if labels[i] != new_label:
                labels[i] = new_label
                changed = True

        if not changed:
            break  # Early stopping

    return labels
    


def compute_stability(adjacency_matrices):
    """
    Compute node-level and global label stability across consecutive snapshots.

    Parameters
    ----------
    partitions : list of arrays
        List of label arrays, one per snapshot (length T).

    Returns
    -------
    nodewise_stability : array, shape (N,)
        Fraction of times each node kept the same label between snapshots.
    global_stability : float
        Mean of nodewise stability scores.
    """
    partitions = [stable_label_propagation(A) for A in adjacency_matrices]
    T = len(partitions)
    N = len(partitions[0])
    nodewise_stability = np.full(N, np.nan)

    transitions = np.zeros(N)

    for t in range(T-1):
        same = partitions[t] == partitions[t+1]
        transitions += same.astype(int)

    if T > 1:
        nodewise_stability = transitions / (T-1)
        global_stability = np.nanmean(nodewise_stability)
    else:
        global_stability = np.nan
    
    score_percentile = np.nanpercentile(nodewise_stability, [5, 25, 50, 75, 95])
    return score_percentile#, global_stability
        

def compute_node_fidelity(adjacency_matrices):
    """
    Compute node-wise and global fidelity across partitions.

    Parameters
    ----------
    partitions : list of arrays
        List of label arrays, one per snapshot.

    Returns
    -------
    nodewise_fidelity : array, shape (N,)
        Fraction of times each node kept its most frequent label.
    global_fidelity : float
        Mean of nodewise fidelity scores.
    """
    partitions = [stable_label_propagation(A) for A in adjacency_matrices]
    n_nodes = len(partitions[0])
    T = len(partitions)
    nodewise_fidelity = np.full(n_nodes, np.nan)

    for i in range(n_nodes):
        labels = [partitions[t][i] for t in range(T)]
        mode_label = max(set(labels), key=labels.count)
        fidelity = labels.count(mode_label) / T
        nodewise_fidelity[i] = fidelity

    global_fidelity = np.nanmean(nodewise_fidelity)
    score_percentile = np.nanpercentile(nodewise_fidelity, [5, 25, 50, 75, 95])
    return score_percentile#, global_fidelity