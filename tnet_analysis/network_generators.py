import numpy as np
import networkx as nx
# import community as community_louvain
import community.community_louvain as community_louvain

from scipy.stats import norm
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

def generate_temporal_link_counts(target_mean, target_sigma, n_snapshots, max_attempts=2000):
    """
    Generate an array of link counts per snapshot with a specified mean and standard deviation.

    The function attempts to match the target standard deviation by iterative redistribution
    while preserving the total number of links and avoiding negative link counts.

    Parameters
    ----------
    target_mean : float
        Desired average number of links per snapshot.
    target_sigma : float
        Desired standard deviation of link counts across snapshots.
    n_snapshots : int
        Number of time snapshots (length of output array).
    max_attempts : int, optional
        Maximum number of adjustment attempts (default: 2000).

    Returns
    -------
    link_counts : array, shape (n_snapshots,)
        Array of link counts per snapshot.
    """
    adjust_value = int(np.round(target_mean / 4))
    total_links = int(np.round(target_mean * n_snapshots))
    base_count = total_links // n_snapshots
    remainder = total_links % n_snapshots

    # Start from uniform base distribution
    link_counts = np.full(n_snapshots, base_count, dtype=int)
    
    # Distribute remainder randomly
    remainder_indices = np.random.choice(np.arange(n_snapshots), remainder, replace=False)
    link_counts[remainder_indices] += 1

    initial_sum = np.sum(link_counts)

    # Save the best solution encountered
    closest_arr = link_counts.copy()
    closest_std_diff = abs(np.std(link_counts) - target_sigma)

    for attempt in range(max_attempts):
        current_std = np.std(link_counts)
        current_std_diff = abs(current_std - target_sigma)

        if current_std_diff < closest_std_diff:
            closest_arr = link_counts.copy()
            closest_std_diff = current_std_diff

        if np.isclose(current_std, target_sigma, atol=0.05):
            return link_counts

        indices = np.arange(n_snapshots)
        np.random.shuffle(indices)
        half = len(indices) // 2
        gain_indices = indices[:half]
        lose_indices = indices[half:]

        # Adjust link counts
        link_counts[gain_indices] += adjust_value
        link_counts[lose_indices] -= adjust_value

        # Ensure non-negative values
        negative_sum = np.abs(np.sum(link_counts[link_counts < 0]))
        link_counts[link_counts < 0] = 0

        if negative_sum > 0:
            non_negative_indices = np.where(link_counts > 0)[0]

            while negative_sum > 0 and len(non_negative_indices) > 0:
                compensation = min(negative_sum // len(non_negative_indices), max(link_counts[non_negative_indices]))
                remainder = negative_sum % len(non_negative_indices)

                new_negative_sum = 0
                for idx in non_negative_indices:
                    if link_counts[idx] >= compensation:
                        link_counts[idx] -= compensation
                    else:
                        new_negative_sum += compensation - link_counts[idx]
                        link_counts[idx] = 0

                picking = np.random.choice(non_negative_indices, remainder, replace=False)
                for idx in picking:
                    if link_counts[idx] > 0:
                        link_counts[idx] -= 1
                    else:
                        new_negative_sum += 1

                negative_sum = new_negative_sum
                non_negative_indices = np.where(link_counts > 0)[0]

        current_sum = np.sum(link_counts)
        if current_sum != initial_sum:
            break  # safety check

    return closest_arr

def generate_temporal_network_with_density_variation(tnet0, variation_type, sigma=0, scale=1):
    """
    Generate a cloned version of a temporal network with controlled density variability.

    The output temporal network has the same number of nodes and snapshots,
    but the number of active edges per snapshot varies according to the specified scheme.

    Parameters
    ----------
    tnet0 : array, shape (T, N, N)
        Reference temporal network to clone structure from.
    variation_type : str
        Type of variability:
        - 'desiredSigmaRho': use a user-specified sigma
        - 'sig_isMean': set sigma = mean number of edges
        - 'sig_halfMean': set sigma = 0.5 * mean number of edges
        - 'sig_quarterMean': set sigma = 0.25 * mean number of edges
    sigma : float, optional
        Standard deviation for edge count variability if using 'desiredSigmaRho' (default: 0).
    scale : float, optional
        Scaling factor to apply to mean when using proportional types (default: 1).

    Returns
    -------
    new_tnet : array, shape (T, N, N)
        Temporal network with modified density variability.
    """
    T, N, _ = tnet0.shape
    
    # Compute original number of edges per snapshot
    n_edges_per_snapshot = (np.sum(tnet0, axis=(1, 2)) - N) // 2  # undirected graph correction
    mean_edges = np.mean(n_edges_per_snapshot)
    std_edges = np.std(n_edges_per_snapshot)

    # Set desired standard deviation
    if variation_type == 'desiredSigmaRho':
        target_std = sigma
    elif variation_type in ['sig_isMean', 'sig_halfMean', 'sig_quarterMean']:
        target_std = scale * mean_edges
    else:
        raise ValueError(f"Unknown variation_type: {variation_type}")

    # Generate new edge counts
    new_edge_counts = generate_temporal_link_counts(mean_edges, target_std, T)

    # Generate new temporal network
    new_tnet = np.zeros((T, N, N), dtype=int)
    for t in range(T):
        new_tnet[t] = symmRandNetWithDiag(N, new_edge_counts[t], mode='nEdges')

    return new_tnet


def generate_symmetric_random_network_with_diag(N, x, flag):
    """
    Generate a symmetric random network with diagonal ones (self-loops).

    Parameters
    ----------
    N : int
        Number of nodes.
    x : float or int
        Depending on 'flag', controls density, degree, or number of edges.
    flag : str
        Interpretation of 'x':
        - 'density': x is density (0-1)
        - 'degree': x is average degree
        - 'nEdges': x is total number of edges

    Returns
    -------
    adjacency_matrix : array, shape (N, N)
        Symmetric binary adjacency matrix with self-loops.
    """
    if flag == 'density':
        n_edges = int(x * N * (N - 1) / 2)
    elif flag == 'degree':
        n_edges = int(x * (N - 1) / 2)
    elif flag == 'nEdges':
        n_edges = x
    else:
        raise ValueError(f"Unknown flag: {flag}")

    src, tgt = np.tril_indices(N, -1)
    edges = np.column_stack((src, tgt))
    np.random.shuffle(edges)
    selected_src, selected_tgt = edges[:n_edges].T

    adj = np.zeros((N, N))
    adj[selected_src, selected_tgt] = 1
    adj = adj + adj.T
    np.fill_diagonal(adj, 1)

    return adj
    
    
def generate_scale_free_network_with_diag(N, x, flag):
    """
    Generate a symmetric scale-free network with diagonal ones (self-loops).

    Parameters
    ----------
    N : int
        Number of nodes.
    x : float or int
        Depending on 'flag', controls density, degree, or number of edges.
    flag : str
        Interpretation of 'x':
        - 'density': x is density (0-1)
        - 'degree': x is average degree
        - 'nEdges': x is total number of edges

    Returns
    -------
    adjacency_matrix : array, shape (N, N)
        Symmetric binary adjacency matrix with self-loops.
    """
    if flag == 'degree':
        k = x
    elif flag == 'nEdges':
        k = 2 * x / (N - 1)
    elif flag == 'density':
        k = x * N
    else:
        raise ValueError(f"Unknown flag: {flag}")

    gamma = 2.5  # Typical for many real-world networks
    degrees = generateDegrees(N, k, gamma)

    stubs = []
    for node, degree in enumerate(degrees):
        stubs.extend([node] * degree)

    np.random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs) - 1, 2):
        if stubs[i] != stubs[i + 1]:  # Avoid self-loops
            edges.append((stubs[i], stubs[i + 1]))

    adj = np.zeros((N, N))
    if len(edges) > 0:
        src, tgt = np.array(edges).T
        adj[src, tgt] = 1
    adj = adj + adj.T
    np.fill_diagonal(adj, 1)

    return adj
    


def generate_symmetric_network_with_diag(N, x, flag, kind):
    """
    Generate a symmetric network (random or scale-free) with diagonal ones (self-loops).

    Parameters
    ----------
    N : int
        Number of nodes.
    x : float or int
        Parameter controlling density/degree/edges.
    flag : str
        Interpretation of 'x' ('density', 'degree', or 'nEdges').
    kind : str
        Type of network:
        - 'er': random (Erdős-Rényi)
        - 'sf': scale-free

    Returns
    -------
    adjacency_matrix : array, shape (N, N)
        Symmetric binary adjacency matrix.
    """
    if kind == 'er':
        adj = generate_symmetric_random_network_with_diag(N, x, flag)
    elif kind == 'sf':
        adj = generate_scale_free_network_with_diag(N, x, flag)
    else:
        raise ValueError(f"Unknown network kind: {kind}")

    return adj
    

def partial_randomizer(conn_mat, p, preserve_first_snapshot=False):
    """
    Partially randomize a symmetric connectivity matrix while preserving self-loops.

    A proportion 'p' of existing edges are rewired randomly among non-existent edges.

    Parameters
    ----------
    conn_mat : array, shape (N, N)
        Input binary adjacency matrix (symmetric).
    p : float
        Proportion of existing edges to rewire (0 = no randomization, 1 = full randomization).
    preserve_first_snapshot : bool
        If True, no randomization is applied (returns conn_mat unchanged).

    Returns
    -------
    randomized_mat : array, shape (N, N)
        Partially randomized symmetric matrix with self-loops on the diagonal.
    """
    N = len(conn_mat)

    if preserve_first_snapshot:
        return conn_mat

    if np.sum(conn_mat) == N:  # Only diagonal ones
        return conn_mat

    # Work on lower triangle excluding diagonal
    mtx = np.tril(conn_mat)
    np.fill_diagonal(mtx, 0)

    existing_edges = list(zip(*np.where(mtx)))
    
    # Matrix where missing links are zeros (after filling upper triangle)
    fmtx = mtx + np.triu(np.ones((N, N)))
    nonexisting_edges = list(zip(*np.where(fmtx == 0)))

    howmany = int(np.round(len(existing_edges) * p))
    np.random.shuffle(existing_edges)
    np.random.shuffle(nonexisting_edges)

    if howmany > 0:
        new_edges = existing_edges[:-howmany] + nonexisting_edges[:howmany]
    else:
        new_edges = existing_edges

    rsrc, rtgt = np.array(new_edges).T
    new_mat = np.zeros_like(conn_mat)
    new_mat[rsrc, rtgt] = 1

    randomized_mat = new_mat + new_mat.T
    np.fill_diagonal(randomized_mat, 1)

    return randomized_mat


def generate_randomized_temporal_network(initial_matrix, T, prnd):
    """
    Generate a randomized temporal network starting from a static connectivity matrix.

    Parameters
    ----------
    initial_matrix : array, shape (N, N)
        Base adjacency matrix (binary, symmetric).
    T : int
        Number of snapshots to generate.
    prnd : float
        Randomization proportion at each snapshot (0 = no randomization, 1 = full randomization).

    Returns
    -------
    tnet : array, shape (T, N, N)
        Temporal network with partially randomized snapshots.
    """
    N = len(initial_matrix)
    tnet = np.zeros((T, N, N))

    for it in range(T):
        preserve_first = (it == 0)
        tnet[it] = partial_randomizer(initial_matrix, prnd, preserve_first_snapshot=preserve_first)

    return tnet
    

def nullify_temporal_network(tnet0, tag, scale=1):
    """
    Generate null models of a temporal network.

    Parameters
    ----------
    tnet0 : array, shape (T, N, N)
        Original temporal network.
    tag : str
        Type of null model:
        - 'empirical': unchanged
        - 'static': unchanged
        - 'time': shuffle time order
        - 'edges': preserve number of edges per snapshot
        - 'edgetime': shuffle time first, then re-randomize edges
        - 'desiredSigmaRho', 'sig_isMean', 'sig_halfMean', 'sig_quarterMean': control density variability
        - 'linkActivation': preserve edge activation rates but shuffle over time
    scale : float, optional
        Scaling factor if applicable (default: 1).

    Returns
    -------
    tnet : array, shape (T, N, N)
        Nullified temporal network.
    """
    T, N, _ = tnet0.shape

    if tag in ['original', 'static']:
        tnet = tnet0
    elif tag == 'time':
        time_idx = np.arange(T)
        np.random.shuffle(time_idx)
        tnet = tnet0[time_idx, :, :]
    elif tag == 'edges':
        n_edges = (np.sum(np.sum(tnet0, axis=1), axis=1) - N) / 2
        n_edges = n_edges.astype(int)
        tnet = np.zeros((T, N, N))
        for t in range(T):
            tnet[t] = generate_symmetric_random_network_with_diag(N, n_edges[t], 'nEdges')
    elif tag == 'edgetime':
        tnet_shuffled = nullify_temporal_network(tnet0.copy(), 'time')
        tnet = nullify_temporal_network(tnet_shuffled.copy(), 'edges')
    elif tag in ['desiredSigmaRho', 'sig_isMean', 'sig_halfMean', 'sig_quarterMean']:
        tnet = generate_temporal_network_with_density_variation(tnet0, tag, 0, scale)
    elif tag == 'linkActivation':
        lower_indices = np.column_stack(np.tril_indices(N, -1))
        row, col = lower_indices.T
        edge_activation = tnet0[:, row, col]

        num_activation = np.sum(edge_activation, axis=0).astype(int)

        tnet = np.repeat([np.eye(N)], T, axis=0)

        for (i, j), activations in zip(lower_indices, num_activation):
            selected_times = np.random.choice(T, activations, replace=False)
            tnet[selected_times, i, j] = 1
            tnet[selected_times, j, i] = 1
    else:
        raise ValueError(f"Unrecognized tag: {tag}")

    return tnet
    