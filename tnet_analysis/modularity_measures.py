import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

def detect_communities_per_snapshot(adjacency_matrices):
    """
    Detect community partitions at each snapshot using Louvain method.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of adjacency matrices.

    Returns
    -------
    labels_per_snapshot : list of arrays
        List of community labels for each node at each time point.
    """
    T, N, _ = adjacency_matrices.shape
    labels_per_snapshot = []

    for t in range(T):
        adj = adjacency_matrices[t]
        G = nx.from_numpy_array(adj)
        partition = community_louvain.best_partition(G, weight='weight')
        labels = np.array([partition[i] for i in range(N)])
        labels_per_snapshot.append(labels)

    return labels_per_snapshot

def compute_modularity(adj, labels):
    """
    Compute modularity score of a partition for a given adjacency matrix.

    Parameters
    ----------
    adj : array, shape (N, N)
        Adjacency matrix.
    labels : array, shape (N,)
        Community label assignment per node.

    Returns
    -------
    modularity_score : float
        Modularity of the given partition (range roughly -0.5 to 1).
    """
    N = adj.shape[0]
    degrees = np.sum(adj, axis=1)
    m = np.sum(degrees) / 2  # Total number of edges

    Q = 0
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:  # Same community
                Q += (adj[i, j] - degrees[i]*degrees[j]/(2*m))
    
    return Q / (2*m)
    
def compute_modularity_over_time(adjacency_matrices, labels_per_snapshot):
    """
    Compute modularity scores across all time points.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of adjacency matrices.
    labels_per_snapshot : list of arrays
        Community labels at each snapshot.

    Returns
    -------
    modularity_scores : array, shape (T,)
        Modularity at each snapshot.
    """
    T = adjacency_matrices.shape[0]
    modularity_scores = []

    for t in range(T):
        adj = adjacency_matrices[t]
        labels = labels_per_snapshot[t]
        Q = compute_modularity(adj, labels)
        modularity_scores.append(Q)

    return np.array(modularity_scores)
    
def compute_allegiance_matrix(labels_per_snapshot):
    """
    Compute allegiance matrix across time.

    Allegiance[i,j] = fraction of time nodes i and j were in the same community.

    Parameters
    ----------
    labels_per_snapshot : list of arrays
        Community labels at each time point.

    Returns
    -------
    allegiance_matrix : array, shape (N, N)
        Allegiance values between node pairs.
    """
    T = len(labels_per_snapshot)
    N = len(labels_per_snapshot[0])
    allegiance = np.zeros((N, N))

    for t in range(T):
        labels = labels_per_snapshot[t]
        for i in range(N):
            for j in range(i+1, N):
                if labels[i] == labels[j]:
                    allegiance[i, j] += 1

    allegiance = allegiance / T  # Normalize
    allegiance += allegiance.T  # Symmetrize
    return allegiance
    
def gAllegiance_and_gModularity(adjacency_matrices):
    """
    Compute global allegiance score and modularity percentiles.

    Returns allegiance and [5, 25, 50, 75, 95]-percentile modularity scores.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of adjacency matrices.

    Returns
    -------
    segregation_profile : list
        [allegiance_score, modularity_5th, modularity_25th, modularity_50th, modularity_75th, modularity_95th]
    """
    labels_per_snapshot = detect_communities_per_snapshot(adjacency_matrices)
    modularity_scores = compute_modularity_over_time(adjacency_matrices, labels_per_snapshot)
    modularity_percentiles = list(np.percentile(modularity_scores, [5, 25, 50, 75, 95]))
    
    allegiance_matrix = compute_allegiance_matrix(labels_per_snapshot)
    N = allegiance_matrix.shape[0]
    global_allegiance_score = np.sum(allegiance_matrix) / (N * (N-1))

    return [global_allegiance_score] + modularity_percentiles