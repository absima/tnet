import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score



def compute_minimal_temporal_distances(tnet):
    """
    Estimate minimal step distances and reachability in a temporal network 
    using a smart deterministic walker.

    Parameters
    ----------
    tnet : array, shape (T, N, N)
        Sequence of binary adjacency matrices over time.

    Returns
    -------
    ave_reachability : array, shape (N,)
        Average number of nodes reachable from each node.
    minimal_distances : array, shape (N, N)
        Minimal number of steps needed to reach each node from another node.
        (Infinity if unreachable.)
    """
    T, N, _ = tnet.shape

    dmtx = np.full((N, N), np.inf)
    np.fill_diagonal(dmtx, 0)

    prod = np.identity(N)

    for t in range(T):
        prod = np.matmul(prod, tnet[t])
        prod[prod > 0] = 1  # Avoid accumulation
        
        cprod = prod.copy()
        np.fill_diagonal(cprod, 0)
        cprod[cprod > 0] = t + 1
        cprod[cprod == 0] = np.inf

        dmtx = np.minimum(dmtx, cprod)

    # Reachability calculation
    reachable = 1 - np.isinf(dmtx)
    ave_reachability = np.sum(reachable, axis=1) - 1  # Exclude self-links

    return ave_reachability, dmtx
    
    

def compute_random_walk_temporal_efficiency(tnet, n_reps=100):
    """
    Estimate random-walk-based first passage times and reachability in a temporal network.

    Parameters
    ----------
    tnet : array, shape (T, N, N)
        Sequence of binary adjacency matrices over time.
    n_reps : int, optional
        Number of random walk repetitions (default: 100).

    Returns
    -------
    ave_reachability : array, shape (N,)
        Average number of reachable nodes per starting node.
    mean_first_passage_times : array, shape (N, N)
        Mean first passage times between node pairs.
    feasible_paths : array, shape (N, N)
        Proportion of random walks that successfully reached a given destination.
    """
    T, N, _ = tnet.shape

    mFPT = np.full((N, N), np.inf)
    feasible_paths = np.zeros((N, N))
    reachable_counts = np.zeros((n_reps, N))

    for rep in range(n_reps):
        for start in range(N):
            first_pass_times = np.full(N, np.inf)
            first_pass_times[start] = 0
            current_node = start
            visited = 1

            for t in range(1, T + 1):
                neighbors = np.where(tnet[t - 1, current_node, :] == 1)[0]
                if len(neighbors) == 0:
                    break

                current_node = np.random.choice(neighbors)

                if t < first_pass_times[current_node]:
                    first_pass_times[current_node] = t
                    visited += 1

                if visited == N:
                    break

            for j in range(N):
                if not np.isinf(first_pass_times[j]):
                    if np.isinf(mFPT[start, j]):
                        mFPT[start, j] = first_pass_times[j]
                    else:
                        mFPT[start, j] = ((mFPT[start, j] * rep) + first_pass_times[j]) / (rep + 1)

                    feasible_paths[start, j] = ((feasible_paths[start, j] * rep) + 1) / (rep + 1)

            reachable_counts[rep, start] = np.sum(~np.isinf(first_pass_times))

    ave_reachability = np.mean(reachable_counts, axis=0)

    return ave_reachability, mFPT, feasible_paths


def summarize_temporal_distance_matrix(dmtx):
    """
    Summarize a temporal distance matrix into integration and eccentricity measures.

    Parameters
    ----------
    dmtx : array, shape (N, N)
        Matrix of minimal distances between nodes (inf for unreachable pairs).

    Returns
    -------
    mean_distance : float
        Mean distance between reachable node pairs.
    mean_efficiency : float
        Mean of (1 / distance) over reachable pairs.
    diameter : float
        Longest minimal distance in the network.
    radius : float
        Smallest maximal distance from any node.
    mean_eccentricity : float
        Average of node eccentricities.
    """
    mff = np.copy(dmtx)
    mff[mff == np.inf] = np.nan
    mff[mff == 0] = np.nan  # Exclude self-distances

    N = dmtx.shape[0]

    if np.sum(np.isnan(mff)) == N**2:
        return [np.nan] * 5

    # Efficiency and mean distance
    mean_distance = np.nanmean(mff)
    mean_efficiency = np.nanmean(1 / mff)

    # Remove fully disconnected rows/columns
    valid_rows = ~np.isnan(mff).all(axis=1)
    valid_cols = ~np.isnan(mff).all(axis=0)
    mff_trimmed = mff[np.ix_(valid_rows, valid_cols)]

    # Eccentricity, diameter, radius
    max_dist_row = np.nanmax(mff_trimmed, axis=1)
    max_dist_col = np.nanmax(mff_trimmed, axis=0)
    all_max_distances = np.concatenate((max_dist_row, max_dist_col))

    radius = np.nanmin(all_max_distances)
    diameter = np.nanmax(all_max_distances)
    mean_eccentricity = np.nanmean(all_max_distances)

    return mean_distance, mean_efficiency, diameter, radius, mean_eccentricity