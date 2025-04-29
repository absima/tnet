import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

# experimental and advanced temporal network measures

def compute_circulation_latency(adjacency_matrices, max_latency=None):
    """
    Compute node-wise circulation latency in a temporal network.
    
    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of adjacency matrices (binary, 0/1), with waiting (diagonal ones).
    max_latency : int, optional
        Maximum number of steps to search. If None, set to T.

    Returns
    -------
    result : dict
        {
            'circulation_matrix': (N_active, T) latency matrix (NaN if no circulation),
            'nodewise_mean_latency': (N_active,) mean latency per node,
            'global_mean_latency': float,
            'global_std_latency': float,
            'global_percentiles_latency': (5,25,50,75,95) percentiles
        }
    """
    T, N, _ = adjacency_matrices.shape

    if max_latency is None:
        max_latency = T

    # identify active nodes
    node_activity = np.sum(adjacency_matrices, axis=(0, 1)) + np.sum(adjacency_matrices, axis=(0, 2))
    active_nodes = np.where(node_activity > 0)[0]
    N_active = len(active_nodes)

    if N_active == 0:
        raise ValueError("No active nodes in the temporal network.")

    # trim to active nodes for efficiency
    adj = adjacency_matrices[:, active_nodes][:, :, active_nodes]

    # Step 3: Force binary matrices, ensure diagonals=1 (waiting)
    adj = (adj > 0).astype(int)
    for t in range(T):
        np.fill_diagonal(adj[t], 1)

    # Step 4: Initialize circulation latency matrix
    circulation_latency = np.full((N_active, T), np.nan)

    # Step 5: For each node and each starting time
    for node_idx in range(N_active):
        for start_time in range(T-1):
            A = adj[start_time].copy()

            if np.sum(A[node_idx]) <= 1:
                continue  # Only waiting, no outgoing links

            path_matrix = A.copy()
            prev_diag = path_matrix[node_idx, node_idx]

            for latency in range(1, min(max_latency, T-start_time)):
                path_matrix = path_matrix @ adj[start_time + latency]
                new_diag = path_matrix[node_idx, node_idx]

                if new_diag > prev_diag:
                    # Circulation detected
                    if np.isnan(circulation_latency[node_idx, start_time]):
                        circulation_latency[node_idx, start_time] = latency + 1  # latency = steps + 1
                    break

                prev_diag = new_diag

    # Step 6: Global summaries
    valid_rows = ~np.all(np.isnan(circulation_latency), axis=1)
    nodewise_mean_latency = np.full(circulation_latency.shape[0], np.nan)
    nodewise_mean_latency[valid_rows] = np.nanmean(circulation_latency[valid_rows], axis=1)
    # nodewise_mean_latency = np.nanmean(circulation_latency, axis=1)
    global_mean_latency = np.nanmean(nodewise_mean_latency)
    global_std_latency = np.nanstd(nodewise_mean_latency)
    global_percentiles_latency = np.nanpercentile(nodewise_mean_latency, [5, 25, 50, 75, 95])

    # result = {
    #     'circulation_matrix': circulation_latency,
    #     'nodewise_mean_latency': nodewise_mean_latency,
    #     'global_mean_latency': global_mean_latency,
    #     'global_std_latency': global_std_latency,
    #     'global_percentiles_latency': global_percentiles_latency
    # }

    return global_percentiles_latency#result
    
    
def compute_returnability(adjacency_matrices):
    """
    Compute node-wise returnability scores in a temporal network.
    
    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of adjacency matrices (binary, 0/1).

    Returns
    -------
    returnability_scores : array, shape (N,)
        Mean returnability per node.
    """
    T, N, _ = adjacency_matrices.shape
    returnability_scores = np.full(N, np.nan)

    for i in range(N):
        past_neighbors = set()
        returnability_over_time = []

        for t in range(T):
            neighbors_t = set(np.where(adjacency_matrices[t, i] > 0)[0]) - {i}  # Exclude self

            if not neighbors_t:
                continue  # No neighbors, skip

            if past_neighbors:
                old_friends = neighbors_t.intersection(past_neighbors)
                returnability = len(old_friends) / len(neighbors_t)
                returnability_over_time.append(returnability)

            past_neighbors.update(neighbors_t)

        if returnability_over_time:
            returnability_scores[i] = np.mean(returnability_over_time)

    score_percentile = np.nanpercentile(returnability_scores, [5, 25, 50, 75, 95])
    return score_percentile

def compute_temporal_conductance(adjacency_matrices):
    """
    Compute node-wise temporal conductance scores (ego-group based).

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    conductance_scores : array, shape (N,)
        Mean conductance per node over time.
    """
    T, N, _ = adjacency_matrices.shape
    conductance_scores = np.full(N, np.nan)

    for i in range(N):
        conductance_over_time = []

        for t in range(T):
            neighbors = set(np.where(adjacency_matrices[t, i] > 0)[0]) - {i}

            if not neighbors:
                continue  # No group to define, skip

            ego_group = neighbors.union({i})
            ego_indices = np.array(list(ego_group))

            # Internal connections: count edges inside ego-group
            subgraph = adjacency_matrices[t][np.ix_(ego_indices, ego_indices)]
            internal_connections = np.sum(subgraph) - np.trace(subgraph)  # exclude self-loops if any
            internal_connections /= 2  # since undirected, each edge counted twice

            # Outgoing connections: from ego-group to outside
            total_outgoing = np.sum(adjacency_matrices[t][ego_indices, :]) - np.sum(subgraph)

            # Total connections
            denom = internal_connections + total_outgoing

            if denom > 0:
                conductance = total_outgoing / denom
                conductance_over_time.append(conductance)

        if conductance_over_time:
            conductance_scores[i] = np.mean(conductance_over_time)

    return conductance_scores



def compute_core_periphery_stability(adjacency_matrices, method='degree', core_fraction=0.2):
    """
    Compute temporal core-periphery stability for each node.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).
    method : str, optional
        Method to define core: 'degree', 'kcore', or 'modularity'.
    core_fraction : float, optional
        Fraction of nodes to label as core (only used for 'degree').

    Returns
    -------
    result : dict
        {
            'core_labels': (N, T) binary matrix (1=core, 0=periphery),
            'stability_scores': (N,) fraction of times role stayed the same
        }
    """
    T, N, _ = adjacency_matrices.shape
    core_labels = np.zeros((N, T), dtype=int)

    for t in range(T):
        adj = adjacency_matrices[t]
        G = nx.from_numpy_array(adj)

        if method == 'degree':
            degrees = np.array([d for n, d in G.degree()])
            threshold = np.percentile(degrees, 100 * (1 - core_fraction))
            core_nodes = np.where(degrees >= threshold)[0]

        elif method == 'kcore':
            core_indices = nx.core_number(G)
            max_kcore = max(core_indices.values())
            core_nodes = [node for node, k in core_indices.items() if k == max_kcore]

        elif method == 'modularity':
            if G.number_of_edges() == 0:
                core_nodes = []
            else:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G))
                largest_community = max(communities, key=len)
                core_nodes = list(largest_community)

        else:
            raise ValueError(f"Unknown method: {method}")

        core_labels[core_nodes, t] = 1

    # Stability: how often role stays the same across consecutive snapshots
    stability_scores = np.full(N, np.nan)

    for i in range(N):
        transitions = np.diff(core_labels[i])
        num_switches = np.sum(transitions != 0)
        possible_transitions = T - 1

        if possible_transitions > 0:
            stability_scores[i] = 1 - (num_switches / possible_transitions)

    result = {
        'core_labels': core_labels,
        'stability_scores': stability_scores
    }

    return result
    

def compute_link_burstiness(adjacency_matrices):
    """
    Compute burstiness index for each link in the temporal network.

    Parameters
    ----------
    adjacency_matrices : array, shape (T, N, N)
        Sequence of binary adjacency matrices (0/1).

    Returns
    -------
    link_burstiness : array, shape (N, N)
        Burstiness index per link (NaN if insufficient data).
    """
    T, N, _ = adjacency_matrices.shape
    link_burstiness = np.full((N, N), np.nan)

    for i in range(N):
        for j in range(i+1, N):  # Only upper triangle (i < j)
            activation_times = []

            for t in range(T):
                if adjacency_matrices[t, i, j] > 0:
                    activation_times.append(t)

            if len(activation_times) >= 3:  # Need at least 2 inter-event gaps
                inter_event_times = np.diff(activation_times)
                mu = np.mean(inter_event_times)
                sigma = np.std(inter_event_times)

                if mu + sigma > 0:
                    burstiness = (sigma - mu) / (sigma + mu)
                    link_burstiness[i, j] = burstiness
                    link_burstiness[j, i] = burstiness  # symmetric network

    return link_burstiness 
    
    
    