import numpy as np


def GenerateSymmetricRandomNetwork(n_nodes, n_edges):
    """
    Create a symmetric random network with a full diagonal.

    This function generates an undirected random adjacency matrix with 
    a specified number of edges. The diagonal is fully filled with ones, 
    representing self-connections.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network (size of the adjacency matrix).
    n_edges : int
        Number of undirected edges to include (not counting the diagonal).

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix of shape (n_nodes, n_nodes), 
        with `n_edges` undirected edges and a full diagonal of ones.
    """
    src, tgt = np.tril_indices(n_nodes, -1)  # lower-triangular indices
    # all_edges = np.column_stack((src, tgt))
    selected_edges = np.random.permutation(np.column_stack((src, tgt)))[:n_edges]
    # np.random.shuffle(all_edges)

    selected_src, selected_tgt = selected_edges.T
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    adj_matrix[selected_src, selected_tgt] = 1

    # Make symmetric
    adj_matrix = adj_matrix + adj_matrix.T

    # Fill diagonal with ones
    np.fill_diagonal(adj_matrix, 1)

    return adj_matrix


def GenerateSymmetricSmallWorldNetwork(n_nodes, n_edges, rewire_prob=0.02, seed=None):
    """
    Create a symmetric small-world-like random network with a full diagonal.

    This constructs an undirected adjacency matrix by:
      1) Enumerating all unordered node pairs sorted by their circular distance
         (nearby pairs first), which approximates a lattice-style backbone.
      2) Selecting the first `n_edges` pairs as the initial edge set.
      3) Rewiring a fraction (`rewire_prob`) of those edges to uniformly
         random non-chosen pairs, while preserving symmetry.
      4) Filling the diagonal with ones (self-connections).

    Notes
    -----
    - This is a lightweight small-world *approximation* that emulates
      "local-first, then some random shortcuts" behavior. It is not a
      canonical Watts–Strogatz generator, but it preserves your original
      ordering-by-offset approach and behavior.
    - The resulting matrix is symmetric with a full diagonal of ones.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network (size of the adjacency matrix).
    n_edges : int
        Number of undirected edges to include (not counting the diagonal).
    rewire_prob : float, optional (default=0.02)
        Fraction of the initially selected edges to rewire uniformly at random.
    seed : int or None, optional (default=None)
        Random seed for reproducibility. If None, the global RNG state is used.

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix of shape (n_nodes, n_nodes), with `n_edges`
        undirected edges and a full diagonal of ones.

    Raises
    ------
    ValueError
        If `n_edges` exceeds the maximum possible number of undirected pairs
        for `n_nodes`.
    """
    if seed is not None:
        np.random.seed(seed)

    max_possible_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_possible_edges:
        raise ValueError("n_edges is too large for the given n_nodes.")

    # Enumerate all unordered pairs (i < j), sorted by increasing offset k.
    # This yields "nearby" pairs first, approximating a lattice-like backbone.
    pairs = np.empty((0, 2), dtype=int)
    for k in range(1, n_nodes):
        i_idx = np.arange(n_nodes - k, dtype=int)
        j_idx = i_idx + k
        pairs = np.concatenate((pairs, np.column_stack((i_idx, j_idx))), axis=0)

    # Initial edge set: take the first n_edges "local" pairs.
    retained_edges = pairs[:n_edges].copy()

    # Number of edges to rewire.
    n_rewire = int(rewire_prob * n_edges)

    if n_rewire > 0:
        # Choose which of the retained edges to drop.
        drop_idx = np.random.permutation(n_edges)[:n_rewire]
        discarded = retained_edges[drop_idx]
        keep_mask = np.ones(n_edges, dtype=bool)
        keep_mask[drop_idx] = False
        retained_edges = retained_edges[keep_mask]

        # Candidate pool to rewire into: pairs not in the initial local set.
        pool = pairs[n_edges:]

        # If the pool is smaller than needed (rare), extend with discarded to ensure enough.
        if len(pool) < n_rewire:
            pool = np.vstack((pool, discarded))

        choose_idx = np.random.permutation(len(pool))[:n_rewire]
        rewired = pool[choose_idx]

        # Final edge set after rewiring.
        retained_edges = np.vstack((retained_edges, rewired))

    # Build symmetric adjacency with full diagonal.
    adj_matrix = np.eye(n_nodes, dtype=int)
    if len(retained_edges) > 0:
        i, j = retained_edges.T
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    np.fill_diagonal(adj_matrix, 1)

    return adj_matrix


import numpy as np

def GenerateSymmetricScaleFreeNetwork(n_nodes, n_edges, model='linear', gamma=2.5, a=0.1, alpha=0.5):
    """
    Create a symmetric scale-free-like random network with a full diagonal.

    This constructs an undirected adjacency matrix by assigning probabilities 
    to nodes based on the chosen preferential attachment model. Edges are then 
    sampled according to these preferences.

    Models supported
    ----------------
    - 'sf_linear'       : Linear decay preference (higher index = lower prob).
    - 'sf_exponential'  : Exponential decay with parameter `a`.
    - 'sf_powerlaw'     : Power-law decay with exponent `gamma`.
    - 'sf_hybrid'       : Combination of linear and power-law controlled by `alpha`.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network (size of the adjacency matrix).
    n_edges : int
        Number of undirected edges to include (not counting the diagonal).
    model : str, optional (default='linear')
        Preference model ('sf_linear', 'sf_exponential', 'sf_powerlaw', 'sf_hybrid').
    gamma : float, optional (default=2.5)
        Exponent for the power-law preference model.
    a : float, optional (default=0.1)
        Decay parameter for the exponential preference model.
    alpha : float, optional (default=0.5)
        Mixing parameter for the hybrid model (0 = pure linear, 1 = pure power-law).

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix of shape (n_nodes, n_nodes), 
        with `n_edges` undirected edges and a full diagonal of ones.

    Raises
    ------
    ValueError
        If the provided `model` is not recognized.
    """
    indices = np.arange(1, n_nodes + 1)  # 1-based indexing for preference formulas

    # Generate node preference weights
    if model == 'sf_linear':
        pref_values = n_nodes - indices + 1
    elif model == 'sf_exponential':
        pref_values = np.exp(-a * (indices - 1))
    elif model in ['sf_powerlaw', 'sf']:
        pref_values = indices ** (-gamma)
    elif model == 'sf_hybrid':
        linear = n_nodes - indices + 1
        powerlaw = indices ** (-gamma)
        pref_values = alpha * powerlaw + (1 - alpha) * linear
    else:
        raise ValueError("Unknown model type. Choose from "
                         "'sf_linear', 'sf_exponential', 'sf_powerlaw', 'sf_hybrid'.")

    prefs = pref_values / pref_values.sum()  # Normalize to a probability distribution

    # Generate all possible undirected edges
    possible_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    # Compute probabilities for edges based on node preferences
    edge_probs = np.array([prefs[i] * prefs[j] for i, j in possible_edges])
    edge_probs /= edge_probs.sum()  # Normalize to sum to 1

    # Sample n_edges unique edges
    sampled_indices = np.random.choice(
        len(possible_edges), size=n_edges, replace=False, p=edge_probs
    )
    sampled_edges = [possible_edges[i] for i in sampled_indices]

    # Create symmetric adjacency matrix with diagonal filled
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in sampled_edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    np.fill_diagonal(adj_matrix, 1)

    return adj_matrix




def GenerateSymmetricNetwork(n_nodes, n_edges, kind, seed=None):
    """
    Generate a symmetric network with a full diagonal, based on the specified model type.

    This function serves as a wrapper to construct different network topologies:
    - Erdős–Rényi random network
    - Small-world network
    - Scale-free network (various preference models)

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network (size of the adjacency matrix).
    n_edges : int
        Number of undirected edges to include (not counting the diagonal).
    kind : str
        Type of network to generate. Supported values:
        - 'er', 'er_static', 'static'  → Erdős–Rényi random network
        - 'sw', 'sw_static'            → Small-world network
        - 'sf', 'sf_linear', 
          'sf_exponential', 
          'sf_powerlaw', 
          'sf_hybrid'                  → Scale-free network

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix of shape (n_nodes, n_nodes),
        with the chosen topology and a full diagonal of ones.
    """
    if kind in ['er', 'er_static', 'static']:
        adj_matrix = GenerateSymmetricRandomNetwork(n_nodes, n_edges)
    elif kind in ['sw', 'sw_static']:
        adj_matrix = GenerateSymmetricSmallWorldNetwork(n_nodes, n_edges)
    elif kind in ['sf', 'sf_linear', 'sf_exponential', 'sf_powerlaw', 'sf_hybrid']:
        adj_matrix = GenerateSymmetricScaleFreeNetwork(n_nodes, n_edges, model=kind)
    else:
        raise ValueError(f"Unknown network type '{kind}'.")
    
    return adj_matrix


def GenerateStaticTemporalNetwork(tnet_init, kind, seed=None):
    """
    Generate a static temporal network of a specified topology with a full diagonal.

    This function constructs a temporal network where each snapshot 
    is identical, based on a chosen static network model 
    (Erdős–Rényi, small-world, or scale-free). 

    The number of edges per snapshot is estimated from the provided 
    temporal network template `tnet_init`.

    Parameters
    ----------
    tnet_init : np.ndarray
        Temporal network array of shape (T, N, N) used only to determine
        the number of time steps (T), number of nodes (N), 
        and approximate number of edges per snapshot.
    kind : str
        Type of network to generate. Supported values are the same as in 
        `GenerateSymmetricNetwork`:
        - 'er', 'er_static', 'static'  → Erdős–Rényi random network
        - 'sw', 'sw_static'            → Small-world network
        - 'sf', 'sf_linear', 
          'sf_exponential', 
          'sf_powerlaw', 
          'sf_hybrid'                  → Scale-free network
    seed : int or None, optional (default=None)
        Random seed for reproducibility. Passed to the underlying generator.

    Returns
    -------
    np.ndarray
        Temporal network of shape (T, N, N), where each snapshot is the same 
        symmetric adjacency matrix with a full diagonal of ones.
    """
    T, n_nodes, _ = tnet_init.shape

    num_edges_total = np.sum(tnet_init)
    num_upper_tri = (num_edges_total - n_nodes * T) // 2
    n_edges_per_snapshot = int(np.round(num_upper_tri / T))

    snapshot = GenerateSymmetricNetwork(n_nodes, n_edges_per_snapshot, kind, seed=seed)

    np.fill_diagonal(snapshot, 1)  # ensure diagonal is filled
    tnet = np.repeat(snapshot[np.newaxis, :, :], T, axis=0)

    return tnet





def GenerateTemporalLinkCounts(target_mean, target_sigma, n_snapshots, max_attempts=2000): 
    #generate_temporal_link_counts():
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

def GenerateTnetWithDensityVariation(tnet0, variation_type, sigma=0, scale=1): 
    #generate_temporal_network_with_density_variation
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
    new_edge_counts = GenerateTemporalLinkCounts(mean_edges, target_std, T)

    # Generate new temporal network
    new_tnet = np.zeros((T, N, N), dtype=int)
    for t, nedges in enumerate(new_edge_counts):
        new_tnet[t] = GenerateSymmetricNetwork(N, nedges, kind='er', seed=None)

    return new_tnet



def GenerateTemporalNetworkByLinkActivation(tnet_init, seed=None):
    """
    Generate a null temporal network by preserving per-edge activation counts.

    For each undirected edge (i < j), this function:
      1) Counts how many time frames the edge is active in `tnet_init`.
      2) Randomly selects exactly that many distinct time frames.
      3) Activates the edge at those selected frames.
    All snapshots have a full diagonal of ones. This preserves the
    marginal activation count (over time) of every edge while randomizing
    *when* the activations occur.

    Parameters
    ----------
    tnet_init : np.ndarray
        Input temporal network of shape (T, N, N). Assumed to be binary and
        symmetric per snapshot, with diagonal typically ones.
    seed : int or None, optional (default=None)
        Seed for reproducibility. If provided, uses a dedicated RNG.

    Returns
    -------
    np.ndarray
        Temporal network of shape (T, N, N) constructed by randomizing the
        timing of each edge's activations while preserving total counts.

    Notes
    -----
    - The procedure uses only the lower-triangular part (i < j) to avoid
      double-counting undirected edges, then mirrors to enforce symmetry.
    - If `tnet_init` contains only 0/1 entries, each edge's activation
      count is guaranteed to be in [0, T], so sampling without replacement
      is always valid.
    """
    rng = np.random.default_rng(seed)

    T, n_nodes, _ = tnet_init.shape

    # Extract all undirected edge indices (lower triangle)
    lower_tri_pairs = np.column_stack(np.tril_indices(n_nodes, k=-1))
    row, col = lower_tri_pairs.T

    # Edge activation over time for each (i, j) with i < j
    edge_activation = tnet_init[:, row, col]  # shape: (T, n_edges)

    # Number of active frames per edge
    num_activations = edge_activation.sum(axis=0).astype(int)

    # Initialize temporal network with identity (full diagonal) at each time
    tnet = np.repeat(np.eye(n_nodes, dtype=int)[np.newaxis, :, :], T, axis=0)

    # For each undirected edge, sample the time indices to activate it
    for (i_src, i_tgt), count in zip(lower_tri_pairs, num_activations):
        if count == 0:
            continue
        selected_times = rng.choice(T, size=count, replace=False)
        tnet[selected_times, i_src, i_tgt] = 1
        tnet[selected_times, i_tgt, i_src] = 1

    return tnet



def TrimIsolatedNodes(tnet):
    """
    Remove isolated nodes (across all time frames) from a temporal network.

    A node is considered isolated if, over the entire time horizon, it has
    no incident edges to any other node (self-loops on the diagonal are ignored).
    The function trims such nodes from the adjacency cubes and returns the
    reduced temporal network along with the number of active (non-isolated) nodes.

    Parameters
    ----------
    tnet : np.ndarray
        Temporal network of shape (T, N, N). Each snapshot is assumed to be
        symmetric and typically has ones on the diagonal.

    Returns
    -------
    tnet_trimmed : np.ndarray
        Temporal network of shape (T, M, M), where M ≤ N is the number of
        non-isolated nodes retained.
    n_active : int
        The number of active (non-isolated) nodes after trimming.

    Notes
    -----
    - Isolation is determined by aggregating adjacency over time (sum over T),
      zeroing the diagonal, and checking which nodes have any nonzero incident
      links. This matches the original behavior.
    """
    tnet_copy = tnet.copy()
    n_nodes = tnet_copy.shape[-1]

    # Aggregate across time and ignore diagonal to assess isolation
    aggregated = tnet_copy.sum(axis=0)         # shape (N, N)
    np.fill_diagonal(aggregated, 0)
    incident_sum = aggregated.sum(axis=0).astype(int)

    active_indices = np.where(incident_sum != 0)[0]
    n_active = len(active_indices)

    if n_active < n_nodes:
        isolated = np.setdiff1d(np.arange(n_nodes), active_indices, assume_unique=True)
        tnet_copy = np.delete(tnet_copy, isolated, axis=1)
        tnet_copy = np.delete(tnet_copy, isolated, axis=2)

    return tnet_copy, n_active


def GenerateNullModel(tnet, tag, scale=1, seed=None):
    """
    Generate a null-model temporal network from a given temporal network.

    This wrapper supports multiple null-model types (selected via `tag`) that
    randomize different aspects of the input network while preserving certain
    constraints (e.g., number of edges per snapshot, activation counts, etc.).
    If `trimIsolated` removes isolated nodes, the result is padded back to the
    original size with identity blocks on the diagonal.

    Supported tags
    --------------
    'original'       : Return the trimmed input as-is (no changes).
    'static'         : Same static snapshot repeated across time (ER by default).
    'er_static'      : Alias for 'static' (explicit ER static).
    'sw_static'      : Static small-world snapshot repeated across time.
    'sf_linear'      : Static scale-free (linear preference) snapshot repeated.
    'sf_exponential' : Static scale-free (exponential preference) snapshot repeated.
    'sf_powerlaw'    : Static scale-free (power-law) snapshot repeated.
    'sf_hybrid'      : Static scale-free (hybrid) snapshot repeated.
    'time'           : Time-reshuffle snapshots (random permutation of frames).
    'edges'          : For each frame, regenerate an ER snapshot preserving that
                       frame's edge count.
    'edgetime'       : First 'time' reshuffle, then 'edges'.
    'linkActivation' : Preserve per-edge activation counts across time but
                       randomize the specific time indices of activation.
    'desiredSigmaRho', 'sig_isMean', 'sig_halfMean', 'sig_quarterMean'
                     : Re-generate ER snapshots with per-frame edge counts drawn
                       to match the original mean and a chosen variability (σ).

    Parameters
    ----------
    tnet : np.ndarray
        Input temporal network, shape (T, N, N), assumed binary and symmetric
        with ones on the diagonal.
    tag : str
        Specifies which null-model to generate (see "Supported tags").
    scale : float, optional (default=1)
        Scale factor used when `tag` is one of the sigma-modes. Ignored otherwise.
    seed : int or None, optional (default=None)
        Random seed for reproducibility. Passed to underlying generators and used
        for shuffles in-place here.

    Returns
    -------
    tnet_out : np.ndarray
        Output temporal network, shape (T, N, N), padded back to the original N
        if isolated nodes were trimmed.
    n_core_nodes : int
        The size of the core (non-isolated) node set after trimming.

    Notes
    -----
    - Requires auxiliary functions:
        * trimIsolatedNodes(tnet) -> (tnet_trimmed, n_core_nodes)
        * GenerateSymmetricNetwork(n_nodes, n_edges, kind, seed=None)
        * GenerateStaticTemporalNetwork(tnet_init, kind, seed=None)
        * GenerateTemporalNetworkByLinkActivation(tnet_init, seed=None)
        * GenerateTemporalNetworkWithDensityVariability(tnet_init, sigma_mode, scale, sigma=0)
    - The diagonal is kept as ones in all outputs.
    """
    # Preserve original dimensions for padding at the end
    T, n_nodes, _ = np.shape(tnet)

    # Remove isolated nodes; work on the core, pad back later
    tnet_core, n_core_nodes = TrimIsolatedNodes(tnet)

    # Tag groups
    erand_group = ['erand_er', 'erand_sws', 'erand_sfs', 'erand_swr', 'erand_sfr']
    sfsw_static = ['sf_linear', 'sf_exponential', 'sf_powerlaw', 'sf_hybrid', 'sw_static', 'static', 'er_static']
    sigma_modes = ['desiredSigmaRho', 'sig_isMean', 'sig_halfMean', 'sig_quarterMean']
    sigma_scales = [1.0, 2.0, 0.5, 0.25]  # parallel to sigma_modes
    
    rng = np.random.default_rng(seed)

    if tag == 'original':
        tnet_result = tnet_core

    elif tag == 'time':
        # Permute time order of snapshots
        time_idx = np.arange(T)
        rng.shuffle(time_idx)
        tnet_result = tnet_core[time_idx, :, :]

    elif tag in erand_group: 
        # Preserve per-frame edge counts, regenerate ER snapshot per frame
        # Count edges per frame (exclude diagonal)
        identifier = tag.split('_')[1]
        kind = identifier[:2]
        edge_counts = ((tnet_core.sum(axis=(1, 2)) - n_core_nodes) // 2).astype(int)
        tnet_result0 = np.zeros((T, n_core_nodes, n_core_nodes), dtype=int)
        for t_idx, n_edges in enumerate(edge_counts):
            tnet_result0[t_idx] = GenerateSymmetricNetwork(n_core_nodes, int(n_edges), kind=kind, seed=seed)
        if identifier[0]== 's' and identifier[-1]=='r': # reindex/relable to distroy persistence by construction 
            tnet_result = np.empty_like(tnet_result0)
            for t, mtx in enumerate(tnet_result0):
                perm = np.random.permutation(n_core_nodes)
                tnet_result[t] = mtx[perm][:,perm] 
        else:
            tnet_result = tnet_result0
    elif tag == 'edgetime':
        # First shuffle time, then re-generate edges per frame
        tnet_time, _ = GenerateNullModel(tnet_core.copy(), 'time', seed=seed)
        tnet_result, _ = GenerateNullModel(tnet_time.copy(), 'edges', seed=seed)

    elif tag == 'linkActivation':
        tnet_result = GenerateTemporalNetworkByLinkActivation(tnet_core, seed=seed)

    elif tag in sigma_modes:
        # Map tag to its default scale if caller didn't supply a custom one
        default_scale = sigma_scales[sigma_modes.index(tag)]
        use_scale = scale if scale is not None else default_scale
        tnet_result = GenerateTnetWithDensityVariation(tnet_core, variation_type=tag, sigma=0, scale=use_scale)

    elif tag in sfsw_static:
        # Static small-world / scale-free snapshots repeated
        tnet_result = GenerateStaticTemporalNetwork(tnet_core, kind=tag, seed=seed)

    else:
        raise ValueError("Unrecognized tag value for null-model generation.")

    tnet_final_core, n_final_core_nodes = TrimIsolatedNodes(tnet_result)
    
    # If isolated nodes were trimmed away, pad result back to original size
    
    if n_nodes != n_final_core_nodes:
        tnet_out = np.tile(np.eye(n_nodes, dtype=int), (T, 1, 1))
        tnet_out[:, :n_final_core_nodes, :n_final_core_nodes] = tnet_final_core
    else:
        tnet_out = tnet_result

    return tnet_out, n_final_core_nodes




def generateRandomTemporalNetwork(t, n, pconn):
    """
    Generate a symmetric random temporal network.

    This function creates a temporal network represented as a 3D NumPy array of shape (t, n, n),
    where each n x n slice along the time axis is a symmetric adjacency matrix representing the
    network at a given time step. Edges between distinct node pairs are included independently
    with probability `pconn`. All diagonal entries (self-loops) are set to 1.

    Parameters:
        t (int): Duration of the temporal network (number of time steps).
        n (int): Number of nodes in the network.
        pconn (float): Probability of connection between distinct nodes at each time step (0 ≤ pconn ≤ 1).

    Returns:
        np.ndarray: A temporal network of shape (t, n, n), where each entry is 1 if a connection exists,
                    and 0 otherwise. Each n x n slice is symmetric with diagonal entries set to 1.
    """
    # Generate upper triangular random connections (excluding diagonal)
    upper = np.triu(np.random.rand(t, n, n) < pconn, k=1)
    
    # Mirror the upper triangle to the lower triangle to make it symmetric
    tnet = upper + np.transpose(upper, axes=(0, 2, 1))
    
    # Convert to int (0 or 1)
    tnet = tnet.astype(int)
    
    # Add self-loops (diagonal = 1)
    idx = np.arange(n)
    tnet[:, idx, idx] = 1

    return tnet


def PartiallyRandomizeMatrix(adj_matrix, rewire_prob, preserve_first_snapshot=False, seed=None):
    """
    Partially randomize a symmetric connectivity matrix while preserving self-loops.

    A proportion `rewire_prob` of existing edges are rewired randomly among
    non-existent edges. Self-loops (diagonal ones) are always preserved.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Input binary adjacency matrix (symmetric).
    rewire_prob : float
        Proportion of existing edges to rewire (0 = no randomization,
        1 = full randomization).
    preserve_first_snapshot : bool, optional (default=False)
        If True, no randomization is applied (returns `adj_matrix` unchanged).
    seed : int or None, optional (default=None)
        Random seed for reproducibility.

    Returns
    -------
    randomized_matrix : np.ndarray, shape (N, N)
        Partially randomized symmetric adjacency matrix with self-loops
        on the diagonal.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(adj_matrix)

    if preserve_first_snapshot:
        return adj_matrix.copy()

    if np.sum(adj_matrix) == n_nodes:  # Only diagonal ones
        return adj_matrix.copy()

    # Work on lower triangle excluding diagonal
    lower_triangle = np.tril(adj_matrix)
    np.fill_diagonal(lower_triangle, 0)

    existing_edges = list(zip(*np.where(lower_triangle == 1)))

    # Candidate pool for rewiring: non-existing edges (excluding diag)
    mask_existing = lower_triangle + np.triu(np.ones((n_nodes, n_nodes), dtype=int))
    nonexisting_edges = list(zip(*np.where(mask_existing == 0)))

    n_rewire = int(np.round(len(existing_edges) * rewire_prob))
    rng.shuffle(existing_edges)
    rng.shuffle(nonexisting_edges)

    if n_rewire > 0:
        new_edges = existing_edges[:-n_rewire] + nonexisting_edges[:n_rewire]
    else:
        new_edges = existing_edges

    r_src, r_tgt = np.array(new_edges).T
    new_matrix = np.zeros_like(adj_matrix)
    new_matrix[r_src, r_tgt] = 1

    randomized_matrix = new_matrix + new_matrix.T
    np.fill_diagonal(randomized_matrix, 1)

    return randomized_matrix


def GenerateRandomizedTemporalNetwork(initial_matrix, n_snapshots, rewire_prob, seed=None):
    """
    Generate a randomized temporal network from a static connectivity matrix.

    At each snapshot, the base adjacency matrix is partially randomized by
    rewiring a fraction of its edges while preserving the diagonal. The first
    snapshot can be preserved exactly.

    Parameters
    ----------
    initial_matrix : np.ndarray, shape (N, N)
        Base binary adjacency matrix (symmetric).
    n_snapshots : int
        Number of snapshots to generate.
    rewire_prob : float
        Randomization proportion at each snapshot (0 = no randomization,
        1 = full randomization).
    seed : int or None, optional (default=None)
        Random seed for reproducibility.

    Returns
    -------
    tnet : np.ndarray, shape (n_snapshots, N, N)
        Temporal network with partially randomized snapshots.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(initial_matrix)
    tnet = np.zeros((n_snapshots, n_nodes, n_nodes), dtype=int)

    for t_idx in range(n_snapshots):
        preserve_first = (t_idx == 0)
        tnet[t_idx] = PartiallyRandomizeMatrix(
            initial_matrix, rewire_prob,
            preserve_first_snapshot=preserve_first,
            seed=rng.integers(1e9)  # independent seed per snapshot
        )

    return tnet



    
