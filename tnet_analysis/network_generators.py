import numpy as np


def GenerateSymmetricRandomNetwork(n_nodes, n_edges):
    """
    Build a symmetric random adjacency with a full diagonal.

    Args:
        n_nodes: Number of nodes.
        n_edges: Number of undirected off-diagonal edges.

    Returns:
        Symmetric `(n_nodes, n_nodes)` adjacency with diagonal ones.
    """
    src, tgt = np.tril_indices(n_nodes, -1)  # lower-triangular indices
    selected_edges = np.random.permutation(np.column_stack((src, tgt)))[:n_edges]

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
    Build a symmetric small-world-style adjacency with diagonal ones.

    Args:
        n_nodes: Number of nodes.
        n_edges: Number of undirected off-diagonal edges.
        rewire_prob: Fraction of initially local edges to rewire.
        seed: Optional RNG seed.

    Returns:
        Symmetric `(n_nodes, n_nodes)` adjacency with diagonal ones.
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


def GenerateSymmetricScaleFreeNetwork(n_nodes, n_edges, gamma=2.5):
    """
    Build a symmetric power-law scale-free adjacency with diagonal ones.

    Args:
        n_nodes: Number of nodes.
        n_edges: Number of undirected off-diagonal edges.
        gamma: Power-law exponent used for node preference weights.

    Returns:
        Symmetric `(n_nodes, n_nodes)` adjacency with diagonal ones.
    """
    indices = np.arange(1, n_nodes + 1)  # 1-based indexing for preference formulas
    pref_values = indices ** (-gamma)
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
    Dispatch to an ER, small-world, or scale-free generator.

    Args:
        n_nodes: Number of nodes.
        n_edges: Number of undirected off-diagonal edges.
        kind: One of `er`, `sw`, or `sf`.
        seed: Optional RNG seed.

    Returns:
        Symmetric `(n_nodes, n_nodes)` adjacency with diagonal ones.
    """
    if kind == 'er':
        adj_matrix = GenerateSymmetricRandomNetwork(n_nodes, n_edges)
    elif kind == 'sw':
        adj_matrix = GenerateSymmetricSmallWorldNetwork(n_nodes, n_edges)
    elif kind == 'sf':
        adj_matrix = GenerateSymmetricScaleFreeNetwork(n_nodes, n_edges)
    else:
        raise ValueError(f"Unknown network type '{kind}'.")
    
    return adj_matrix


def GenerateStaticTemporalNetwork(tnet_init, kind, seed=None):
    """
    Repeat one static snapshot across all time points.

    Args:
        tnet_init: Reference temporal network of shape `(T, N, N)`.
        kind: One of `er`, `sw`, or `sf`.
        seed: Optional RNG seed.

    Returns:
        Temporal network of shape `(T, N, N)`.
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
    """
    Match per-snapshot link counts to a target mean and spread.

    Args:
        target_mean: Desired mean link count.
        target_sigma: Desired standard deviation.
        n_snapshots: Number of snapshots.
        max_attempts: Maximum redistribution attempts.

    Returns:
        Integer array of length `n_snapshots`.
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


def GenerateTemporalNetworkWithDensityVariation(tnet0, variation_type="uniform"):
    """
    Regenerate a temporal network with uniform edge counts across time.

    Args:
        tnet0: Reference temporal network of shape `(T, N, N)`.
        variation_type: Must be `uniform`.

    Returns:
        Temporal network of shape `(T, N, N)`.
    """
    if variation_type != "uniform":
        raise ValueError("variation_type must be 'uniform'.")

    T, N, _ = tnet0.shape

    # Compute original number of edges per snapshot
    n_edges_per_snapshot = (np.sum(tnet0, axis=(1, 2)) - N) // 2  # undirected graph correction
    mean_edges = np.mean(n_edges_per_snapshot)

    # Generate new edge counts
    new_edge_counts = GenerateTemporalLinkCounts(mean_edges, 0, T)

    # Generate new temporal network
    new_tnet = np.zeros((T, N, N), dtype=int)
    for t, nedges in enumerate(new_edge_counts):
        new_tnet[t] = GenerateSymmetricNetwork(N, nedges, kind='er', seed=None)

    return new_tnet


def GenerateTemporalNetworkByLinkActivation(tnet_init, seed=None):
    """
    Shuffle edge activation times while preserving per-edge totals.

    Args:
        tnet_init: Binary temporal network of shape `(T, N, N)`.
        seed: Optional RNG seed.

    Returns:
        Temporal network of shape `(T, N, N)`.
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


def GenerateTemporalNetworkFromEdgeTemplate(
    otnet,
    tag,
    p_rewire=0.02,
    seed=None,
    alpha_rank=1.0,
    exclude_never_active=True,
    eta_readd=1.0,
):
    """
    Build a temporal network from an empirical or random ranked edge template.

    Args:
        otnet: Reference temporal network of shape `(T, N, N)`.
        tag: `eActTemplate` or `eRndTemplate`.
        p_rewire: Fraction of template edges rewired per frame.
        seed: Optional RNG seed.
        alpha_rank: Rank-decay exponent.
        exclude_never_active: If `True`, `eActTemplate` ignores never-active edges.
        eta_readd: Weight multiplier for re-adding removed edges.

    Returns:
        Tuple `(tnet_out, n_final_core_nodes)`.
    """
    if not (0.0 <= p_rewire <= 1.0):
        raise ValueError("`p_rewire` must be in [0, 1].")
    if eta_readd <= 0:
        raise ValueError("`eta_readd` must be > 0.")
    if alpha_rank < 0:
        raise ValueError("`alpha_rank` must be >= 0.")
    if otnet.ndim != 3 or otnet.shape[1] != otnet.shape[2]:
        raise ValueError("`otnet` must have shape (T, N, N).")
    if tag not in {"eActTemplate", "eRndTemplate"}:
        raise ValueError('tag must be "eActTemplate" or "eRndTemplate".')

    rng = np.random.default_rng(seed)

    T, N, _ = otnet.shape
    work = np.array(otnet, copy=True)
    diag = np.arange(N)
    work[:, diag, diag] = 0

    lower_pairs = np.column_stack(np.tril_indices(N, k=-1))
    r_idx, c_idx = lower_pairs.T
    n_pair_edges = len(lower_pairs)

    edge_activation = work[:, r_idx, c_idx]
    nedges = edge_activation.sum(axis=1).astype(int)
    ne_max = int(nedges.max()) if T else 0

    if ne_max == 0:
        out = np.tile(np.eye(N, dtype=np.uint8), (T, 1, 1))
        return out, N

    total_per_edge = edge_activation.sum(axis=0).astype(int)
    if tag == "eActTemplate":
        order = np.argsort(total_per_edge)[::-1]
    else:
        order = rng.permutation(n_pair_edges)

    rank = np.empty(n_pair_edges, dtype=int)
    rank[order] = np.arange(n_pair_edges)

    main_ids = order[:ne_max]
    rem_ids = order[ne_max:]

    unique_k = np.unique(nedges)
    unique_k = unique_k[unique_k > 0]

    non_selected_by_k = {}
    w_non_by_k = {}

    for k in unique_k:
        k = int(k)
        non_sel = np.concatenate((main_ids[k:], rem_ids), axis=0)
        non_selected_by_k[k] = non_sel

        if len(non_sel) == 0:
            w_non_by_k[k] = None
            continue

        weights = 1.0 / ((rank[non_sel].astype(float) + 1.0) ** alpha_rank)
        if tag == "eActTemplate" and exclude_never_active:
            weights = weights * (total_per_edge[non_sel] > 0)

        w_non_by_k[k] = weights

    out = np.zeros((T, N, N), dtype=np.uint8)

    for t in range(T):
        k = int(nedges[t])

        if k <= 0:
            np.fill_diagonal(out[t], 1)
            continue

        selected_ids = main_ids[:k]

        r = int(p_rewire * k)

        if r > 0:
            r = min(r, k)

            replace_pos = rng.permutation(k)[:r]
            to_replace_ids = selected_ids[replace_pos]

            keep_mask = np.ones(k, dtype=bool)
            keep_mask[replace_pos] = False
            kept_ids = selected_ids[keep_mask]

            non_sel = non_selected_by_k.get(k)
            w_non = w_non_by_k.get(k)

            if non_sel is None or w_non is None:
                pool_ids = to_replace_ids.copy()
                w_pool = np.ones(len(pool_ids), dtype=float)

            else:
                w_non_local = w_non.astype(float)

                w_rep = 1.0 / ((rank[to_replace_ids].astype(float) + 1.0) ** alpha_rank)
                if tag == "eActTemplate" and exclude_never_active:
                    w_rep = w_rep * (total_per_edge[to_replace_ids] > 0)

                pool_ids = np.concatenate((non_sel, to_replace_ids))
                w_pool = np.concatenate((w_non_local, eta_readd * w_rep)).astype(float)

            pos = np.flatnonzero(w_pool > 0)
            if len(pos) < r:
                add_pos = rng.choice(len(pool_ids), size=r, replace=False)
            else:
                w_pos = w_pool[pos].astype(float)
                s = w_pos.sum()
                if s <= 0 or not np.isfinite(s):
                    add_pos = rng.choice(len(pool_ids), size=r, replace=False)
                else:
                    p_pos = w_pos / s
                    add_pos = pos[rng.choice(len(pos), size=r, replace=False, p=p_pos)]

            add_ids = pool_ids[add_pos]
            selected_ids = np.concatenate((kept_ids, add_ids), axis=0)

        rr, cc = lower_pairs[selected_ids].T
        out[t, rr, cc] = 1
        out[t, cc, rr] = 1
        np.fill_diagonal(out[t], 1)

    T, N, _ = out.shape
    tnet_final_core, n_final_core_nodes = TrimIsolatedNodes(out)

    if N != n_final_core_nodes:
        tnet_out = np.tile(np.eye(N, dtype=np.uint8), (T, 1, 1))
        tnet_out[:, :n_final_core_nodes, :n_final_core_nodes] = tnet_final_core
    else:
        tnet_out = out

    return tnet_out, n_final_core_nodes
#### <<<<templates end










def TrimIsolatedNodes(tnet):
    """
    Remove nodes that never connect to any other node across time.
    """
    tnet_copy = tnet.copy()
    n_nodes = tnet_copy.shape[-1]

    aggregated = tnet_copy.sum(axis=0)
    np.fill_diagonal(aggregated, 0)
    incident_sum = aggregated.sum(axis=0).astype(int)

    active_indices = np.where(incident_sum != 0)[0]
    n_active = len(active_indices)

    if n_active < n_nodes:
        isolated = np.setdiff1d(np.arange(n_nodes), active_indices, assume_unique=True)
        tnet_copy = np.delete(tnet_copy, isolated, axis=1)
        tnet_copy = np.delete(tnet_copy, isolated, axis=2)

    return tnet_copy, n_active


def GenerateNullModel(tnet, tag, scale=1, seed=None, **kwargs):
    """
    Generate a temporal null model from `tnet`.

    Supported tags:
        `original`, `emp_static`, `er_static`, `sw_static`, `sf_static`, `time`, `eAct`, `uniform`, `eActTemplate`, `eRndTemplate`, `erand_er`, `erand_sws`, `erand_sfs`, `erand_swr`, `erand_sfr`.

    Args:
        tnet: Input temporal network of shape `(T, N, N)`.
        tag: Null-model identifier.
        scale: Unused legacy argument kept for compatibility.
        seed: Optional RNG seed.
        **kwargs: Extra options forwarded to template-based generators.

    Returns:
        Tuple `(tnet_out, n_final_core_nodes)`.
    """
    # Preserve original dimensions for padding at the end
    T, n_nodes, _ = np.shape(tnet)

    # Remove isolated nodes; work on the core, pad back later
    tnet_core, n_core_nodes = TrimIsolatedNodes(tnet)

    # Tag groups
    erand_group = ['erand_er', 'erand_sws', 'erand_sfs', 'erand_swr', 'erand_sfr']
    static_tag_to_kind = {
        'er_static': 'er',
        'sw_static': 'sw',
        'sf_static': 'sf',
    }
    rng = np.random.default_rng(seed)

    if tag == 'original' or tag == 'emp_static':
        tnet_result = tnet_core

    elif tag == 'time':
        # Permute time order of snapshots
        time_idx = np.arange(T)
        rng.shuffle(time_idx)
        tnet_result = tnet_core[time_idx, :, :]

    elif tag in erand_group: 
        # Preserve per-frame edge counts, regenerate snapshot per frame
        identifier = tag.split('_')[1]
        kind = identifier[:2]
        edge_counts = ((tnet_core.sum(axis=(1, 2)) - n_core_nodes) // 2).astype(int)
        tnet_result0 = np.zeros((T, n_core_nodes, n_core_nodes), dtype=int)
        for t_idx, n_edges in enumerate(edge_counts):
            tnet_result0[t_idx] = GenerateSymmetricNetwork(
                n_core_nodes, int(n_edges), kind=kind, seed=seed
            )
        if identifier[0] == 's' and identifier[-1] == 'r':  # relabel to destroy persistence by construction
            tnet_result = np.empty_like(tnet_result0)
            for t, mtx in enumerate(tnet_result0):
                perm = np.random.permutation(n_core_nodes)
                tnet_result[t] = mtx[perm][:, perm]
        else:
            tnet_result = tnet_result0

    elif tag == 'eAct':
        tnet_result = GenerateTemporalNetworkByLinkActivation(tnet_core, seed=seed)

    elif tag == 'uniform':
        tnet_result = GenerateTemporalNetworkWithDensityVariation(
            tnet_core,
            variation_type='uniform',
        )

    elif tag in static_tag_to_kind:
        # Static snapshot repeated across time.
        tnet_result = GenerateStaticTemporalNetwork(
            tnet_core, kind=static_tag_to_kind[tag], seed=seed
        )

    elif tag in ['eActTemplate', 'eRndTemplate']:
        tnet_result, _ = GenerateTemporalNetworkFromEdgeTemplate(
            tnet_core,
            tag=tag,
            p_rewire=kwargs.get('p_rewire', 0.02),
            seed=seed,
            alpha_rank=kwargs.get('alpha_rank', 1.0),
            exclude_never_active=kwargs.get('exclude_never_active', True),
            eta_readd=kwargs.get('eta_readd', 1.0),
        )

    else:
        raise ValueError("Unrecognized tag value for null-model generation.")

    tnet_final_core, n_final_core_nodes = TrimIsolatedNodes(tnet_result)
    
    if n_nodes != n_final_core_nodes:
        tnet_out = np.tile(np.eye(n_nodes, dtype=int), (T, 1, 1))
        tnet_out[:, :n_final_core_nodes, :n_final_core_nodes] = tnet_final_core
    else:
        tnet_out = tnet_result

    return tnet_out, n_final_core_nodes


def generateRandomTemporalNetwork(t, n, pconn):
    """
    Generate a symmetric random temporal network.
    This function creates a temporal network represented as a 3D NumPy array of shape (t, n, n), where each n x n slice along the time axis is a symmetric adjacency matrix representing the network at a given time step. Edges between distinct node pairs are included independently with probability `pconn`. All diagonal entries (self-loops) are set to 1.

    Args:
        t (int): Duration of the temporal network (number of time steps).
        n (int): Number of nodes in the network.
        pconn (float): Probability of connection between distinct node pairs.

    Returns:
        np.ndarray: A temporal network of shape (t, n, n).
    """
    upper = np.triu(np.random.rand(t, n, n) < pconn, k=1)
    tnet = upper + np.transpose(upper, axes=(0, 2, 1))
    tnet = tnet.astype(int)
    idx = np.arange(n)
    tnet[:, idx, idx] = 1

    return tnet


def PartiallyRandomizeMatrix(adj_matrix, rewire_prob, preserve_first_snapshot=False, seed=None):
    """
    Partially randomize a symmetric connectivity matrix while preserving self-loops.
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
    At each snapshot, the base adjacency matrix is partially randomized by rewiring a fraction of its edges while preserving the diagonal. The first snapshot can be preserved exactly.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(initial_matrix)
    tnet = np.zeros((n_snapshots, n_nodes, n_nodes), dtype=int)

    for t_idx in range(n_snapshots):
        preserve_first = (t_idx == 0)
        tnet[t_idx] = PartiallyRandomizeMatrix(
            initial_matrix, rewire_prob,
            preserve_first_snapshot=preserve_first,
            seed=rng.integers(1e9)
        )

    return tnet
