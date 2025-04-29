import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

def compute_node_persistence(tnet, confidence=.95):
    """
    Compute node-level persistence scores based on theoretical chance threshold.

    Parameters
    ----------
    tnet : array, shape (T, N, N)
        Sequence of binary or weighted adjacency matrices.
    confidence : float
        Confidence level for significance threshold (e.g., 0.95, 0.99).

    Returns
    -------
    persistence_scores : array, shape (N,)
        Persistence score per node (mean persistence probability across ever-connected neighbors).
    persistent_nodes : array
        Indices of nodes classified as significantly persistent.
    """
    
    def compute_theoretical_threshold(adjacency_matrices, confidence):
        T, N, _ = adjacency_matrices.shape
        k  = norm.ppf(1 - (1 - confidence)/2)
    
        link_counts = np.sum(adjacency_matrices > 0, axis=(1,2))  # Sum over nodes
        max_links = N * (N - 1)  # Undirected
    
        p_t = link_counts / max_links  # Probability at each time
        p_avg = np.mean(p_t)  # Average link probability
    
        std_persistence = np.sqrt(p_avg * (1 - p_avg) / T)
    
        threshold = p_avg + k * std_persistence
    
        return threshold
        
    
    T, N, _ = tnet.shape
    
    # Binarize if weighted
    binary_matrices = (tnet > 0).astype(int)
    
    # Persistence counts
    persistence_counts = np.sum(binary_matrices, axis=0)  # (N, N)
    
    # Persistence probabilities
    persistence_probs = persistence_counts / T  # (N, N)
    
    persistence_scores = np.zeros(N)
    
    for i in range(N):
        # Neighbors that ever connected
        ever_connected = persistence_counts[i] > 0
        if np.sum(ever_connected) == 0:
            persistence_scores[i] = np.nan
        else:
            persistence_scores[i] = np.mean(persistence_probs[i][ever_connected])
    
    threshold = compute_theoretical_threshold(tnet, confidence)
    
    # Find significantly persistent nodes
    persistent_nodes = np.where(persistence_scores > threshold)[0]
    
    score_percentiles = np.nanpercentile(persistence_scores, [5, 25, 50, 75, 95])
    n_persistent_nodes = len(persistent_nodes)
    return score_percentiles, n_persistent_nodes
    
def compute_partner_stability(adjacency_matrices):
    """
    Compute partner stability for each node based on connection recurrence over time.

    Stability is higher when neighbor interactions recur more frequently (smaller gaps).

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    stability_scores : array, shape (N,)
        Partner stability score per node (range 0 to 1).
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
                gaps = np.diff(times_connected)
                gaps_all.extend(gaps)

        if gaps_all:
            mean_gap = np.mean(gaps_all)
            stability_scores[i] = 1 - (mean_gap / T)  # Inverse: smaller mean gap = higher stability
        else:
            stability_scores[i] = 0  # No recurring neighbors = 0 stability
    score_percentile = np.nanpercentile(stability_scores, [5, 25, 50, 75, 95])
    return score_percentile
    

def compute_partner_diversity(adjacency_matrices, window_size=10):
    """
    Compute normalized entropy-based partner diversity for each node.

    Diversity is higher when a node's partners are spread across multiple temporal windows.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).
    window_size : int, optional
        Size of the temporal window for grouping connections.

    Returns
    -------
    diversity_scores : array, shape (N,)
        Partner diversity score per node (range 0 to 1).
    """
    T, N, _ = adjacency_matrices.shape
    diversity_scores = np.zeros(N)
    n_windows = T // window_size

    for i in range(N):
        neighbor_time_activity = dict()

        for t in range(T):
            neighbors = np.where(adjacency_matrices[t, i] > 0)[0]
            for j in neighbors:
                if j not in neighbor_time_activity:
                    neighbor_time_activity[j] = set()
                neighbor_time_activity[j].add(t // window_size)  # add window index

        if neighbor_time_activity:
            temporal_spans = np.array([
                len(windows) / n_windows for windows in neighbor_time_activity.values()
            ])
            probs = temporal_spans / np.sum(temporal_spans)
            probs = probs[probs > 0]  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs))
            x = np.log(len(probs))
            diversity_scores[i] = entropy / x if  x not in [0, np.nan] else np.nan # Normalize
        else:
            diversity_scores[i] = 0
    score_percentile = np.nanpercentile(diversity_scores, [5, 25, 50, 75, 95])
    return score_percentile
    

def compute_mean_degree(adjacency_matrices):
    """
    Compute the mean degree for each node across all snapshots.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    mean_degrees : array, shape (N,)
        Mean degree per node across time.
    """
    T, N, _ = adjacency_matrices.shape
    degrees_per_snapshot = np.sum(adjacency_matrices > 0, axis=2)  # Degree at each snapshot
    mean_degrees = np.mean(degrees_per_snapshot, axis=0)
    score_percentile = np.nanpercentile(mean_degrees, [5, 25, 50, 75, 95])
    return score_percentile
    

def compute_neighborhood_memory(adjacency_matrices, lag=1):
    """
    Compute neighborhood memory for each node based on Jaccard similarity
    of neighbor sets separated by a given lag.

    Memory is higher when a node tends to maintain similar neighbors over time.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).
    lag : int, optional
        Temporal lag to compare neighbor sets (default is 1, i.e., consecutive snapshots).

    Returns
    -------
    memory_scores : array, shape (N,)
        Neighborhood memory score per node (range 0 to 1).
    """
    T, N, _ = adjacency_matrices.shape
    memory_scores = np.zeros(N)

    for i in range(N):
        overlaps = []

        for t in range(T - lag):
            neighbors_t = set(np.where(adjacency_matrices[t, i] > 0)[0])
            neighbors_tlag = set(np.where(adjacency_matrices[t + lag, i] > 0)[0])

            if neighbors_t or neighbors_tlag:  # Avoid division by zero
                intersection = neighbors_t.intersection(neighbors_tlag)
                union = neighbors_t.union(neighbors_tlag)
                jaccard_similarity = len(intersection) / len(union)
                overlaps.append(jaccard_similarity)

        if overlaps:
            memory_scores[i] = np.mean(overlaps)
        else:
            memory_scores[i] = 0
    score_percentile = np.nanpercentile(memory_scores, [5, 25, 50, 75, 95])
    return score_percentile
    

def compute_temporal_clustering(adjacency_matrices):
    """
    Compute average temporal clustering coefficient per node across snapshots.

    Clustering is defined as the fraction of realized links among a node's neighbors
    compared to the maximum possible, averaged over time.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    clustering_scores : array, shape (N,)
        Average clustering coefficient per node across time (range 0 to 1).
    """
    T, N, _ = adjacency_matrices.shape
    clustering_trajectories = np.zeros((T, N))

    for t in range(T):
        adj = adjacency_matrices[t]
        
        for i in range(N):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)

            if k >= 2:
                # Count number of links between neighbors
                links = 0
                for idx1, u in enumerate(neighbors):
                    for v in neighbors[idx1+1:]:
                        if adj[u, v] > 0:
                            links += 1
                max_links = k * (k - 1) / 2
                clustering_trajectories[t, i] = links / max_links
            else:
                clustering_trajectories[t, i] = 0

    # Average clustering coefficient over time
    clustering_scores = np.mean(clustering_trajectories, axis=0)
    
    score_percentile = np.nanpercentile(clustering_scores, [5, 25, 50, 75, 95])
    return score_percentile
    