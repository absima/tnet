import numpy as np
from scipy import sparse

# ---------- To be imporved  ----------

def EnsureCsr(A):
    """
    Convert an adjacency matrix to a SciPy CSR matrix with zero diagonal.

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix, shape (N, N)
        Adjacency matrix (dense or any sparse format).

    Returns
    -------
    A_csr : scipy.sparse.csr_matrix, shape (N, N)
        CSR adjacency with dtype float, zeroed diagonal, and no explicit zeros.

    Notes
    -----
    - Intended for undirected (symmetric) graphs; does not enforce symmetry.
    - Keeps weights; negative weights are not handled elsewhere in this module.
    """
    if sparse.issparse(A):
        A = A.tocsr()
        A = A.astype(float, copy=False)
        A.setdiag(0.0)
        A.eliminate_zeros()
        return A
    A = np.asarray(A, dtype=float)
    np.fill_diagonal(A, 0.0)
    return sparse.csr_matrix(A)


def Modularity(A, labels, gamma=1.0):
    """
    Compute Newman–Girvan modularity (undirected) with resolution parameter γ.

    Q_γ = (1 / 2m) * sum_{ij} [ A_ij - γ * (k_i k_j / 2m) ] * δ(c_i, c_j)

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix, shape (N, N)
        Symmetric adjacency (binary or weighted). Diagonal is ignored.
    labels : array-like, shape (N,)
        Community labels per node.
    gamma : float, default=1.0
        Resolution parameter (γ>1 favors smaller communities; γ<1 merges more).

    Returns
    -------
    Q : float
        Modularity value.

    Notes
    -----
    - For sparse A, a dense temporary copy is used for the final block sum;
      for very large N prefer the snapshot scorers below that aggregate by community.
    """
    if sparse.issparse(A):
        k = np.asarray(A.sum(axis=1)).ravel()
    else:
        A = np.asarray(A, dtype=float)
        k = A.sum(axis=1)
    m2 = k.sum()  # = 2m
    if m2 <= 0:
        return 0.0
    m = m2 / 2.0
    labs = np.asarray(labels)
    same = labs[:, None] == labs[None, :]
    if sparse.issparse(A):
        Ad = A.toarray()
        expected = gamma * np.outer(k, k) / (2.0 * m)
        return float(((Ad - expected)[same]).sum() / (2.0 * m))
    else:
        expected = gamma * np.outer(k, k) / (2.0 * m)
        return float(((A - expected)[same]).sum() / (2.0 * m))


def CoarseGrain(A, comm, C):
    """
    Build the coarse (community-level) graph by summing weights between communities.

    Parameters
    ----------
    A : scipy.sparse.spmatrix (CSR preferred), shape (n, n)
        Current-level adjacency.
    comm : array-like, shape (n,)
        Community id (0..C-1) of each node at current level.
    C : int
        Number of communities at current level.

    Returns
    -------
    A_coarse : scipy.sparse.csr_matrix, shape (C, C)
        Coarse graph where entry (p, q) is the sum of weights between communities p and q.

    Notes
    -----
    - Self-loops on the coarse graph are retained (standard Louvain behavior).
    """
    A = A.tocsr()
    rows, cols = A.nonzero()
    data = A.data
    cr = comm[rows]
    cc = comm[cols]
    M = sparse.coo_matrix((data, (cr, cc)), shape=(C, C))
    return M.tocsr()


def AggregateGroups(groups, comm, C):
    """
    Merge lists of original-node indices according to new community assignments.

    Parameters
    ----------
    groups : list of 1D np.ndarray
        At the current level, each entry contains original-node indices of a super-node.
    comm : array-like, shape (len(groups),)
        Community id (0..C-1) assigned to each super-node.
    C : int
        Number of new communities.

    Returns
    -------
    merged : list of 1D np.ndarray (length C)
        Each array lists the original-node indices belonging to that community.
    """
    merged = [list() for _ in range(C)]
    for g_idx, c_id in enumerate(comm):
        merged[c_id].append(groups[g_idx])
    return [np.concatenate(members) if len(members) > 1 else members[0] for members in merged]


def GroupsToLabels(groups, N):
    """
    Convert grouped original-node indices to a dense labels vector.

    Parameters
    ----------
    groups : list of 1D np.ndarray
        Each array contains the original-node indices of one community.
    N : int
        Number of original nodes.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype=int
        Labels 0..C-1 per original node, consistent with `groups`.
    """
    labels = np.empty(N, dtype=int)
    for c_id, idxs in enumerate(groups):
        labels[idxs] = c_id
    return labels


# ---------- Louvain  ----------

def DetectCommunitiesLouvain(A, gamma=1.0, max_passes=100, max_levels=10, tol=1e-7, seed=None):
    """
    Louvain community detection (undirected, weighted) with resolution γ.

    Implements the two-phase Louvain algorithm (Blondel et al., 2008):
    (1) local greedy node moves that improve modularity; (2) aggregation into a
    coarse graph; repeat across levels until improvement < tol or max_levels hit.

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix, shape (N, N)
        Symmetric adjacency (binary or weighted). Diagonal is ignored at the finest level.
    gamma : float, default=1.0
        Modularity resolution parameter.
    max_passes : int, default=100
        Maximum full sweeps of nodes per level during local moving.
    max_levels : int, default=10
        Maximum number of aggregation levels.
    tol : float, default=1e-7
        Minimum modularity gain between levels to continue.
    seed : int or None, default=None
        RNG seed controlling node visitation order.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype=int
        Community assignment for each original node (0..C-1).
    Q : float
        Modularity (with γ) of the returned partition, computed on the original graph.

    Notes
    -----
    - Complexity per pass is O(m) where m is the number of edges.
    - Deterministic behavior requires setting `seed`; different seeds may yield
      different but typically similar-Q partitions.
    """
    rng = np.random.default_rng(seed)
    A = EnsureCsr(A)
    N = A.shape[0]
    if N == 0:
        return np.array([], dtype=int), 0.0

    A_curr = A.copy()
    node_groups_per_level = [[np.array([i], dtype=int) for i in range(N)]]
    prev_Q = -np.inf

    for _level in range(max_levels):
        n_curr = A_curr.shape[0]
        comm = np.arange(n_curr, dtype=int)

        k = np.asarray(A_curr.sum(axis=1)).ravel()
        m2 = k.sum()
        if m2 <= 0:
            final_labels = GroupsToLabels(node_groups_per_level[-1], N)
            return final_labels, 0.0
        m = m2 / 2.0
        tot = k.copy()

        indptr, indices, data = A_curr.indptr, A_curr.indices, A_curr.data
        improved_any, passes = True, 0
        while improved_any and passes < max_passes:
            improved_any = False
            passes += 1
            order = rng.permutation(n_curr)
            for i in order:
                ci = comm[i]
                start, end = indptr[i], indptr[i+1]
                neigh = indices[start:end]
                w = data[start:end]
                k_i = k[i]
                if k_i == 0:
                    continue

                # Aggregate weights to neighboring communities: k_i,in(C)
                k_to_comm = {}
                for nb, wij in zip(neigh, w):
                    cnb = comm[nb]
                    k_to_comm[cnb] = k_to_comm.get(cnb, 0.0) + wij

                # Temporarily remove i from its community
                tot[ci] -= k_i
                comm[i] = -1

                best_c, best_gain = ci, 0.0
                base_tot_ci = tot[ci]

                for c, k_i_in in k_to_comm.items():
                    gain = k_i_in - gamma * k_i * (tot[c]) / (2.0 * m)
                    if gain > best_gain + 1e-14:
                        best_gain, best_c = gain, c

                # Option to return to original community (robustness)
                gain_back = k_to_comm.get(ci, 0.0) - gamma * k_i * (base_tot_ci) / (2.0 * m)
                if gain_back > best_gain + 1e-14:
                    best_gain, best_c = gain_back, ci

                # Commit
                if best_c != ci:
                    comm[i] = best_c
                    tot[best_c] += k_i
                    improved_any = True
                else:
                    comm[i] = ci
                    tot[ci] += k_i

        # Relabel consecutive 0..C-1
        unique_comms, new_ids = np.unique(comm, return_inverse=True)
        C = unique_comms.size
        comm = new_ids

        # Stop if no aggregation occurs
        if C == n_curr:
            final_groups = AggregateGroups(node_groups_per_level[-1], comm, C)
            final_labels = GroupsToLabels(final_groups, N)
            Q = Modularity(A, final_labels, gamma=gamma)
            return final_labels, Q

        # Aggregate and iterate
        A_next = CoarseGrain(A_curr, comm, C)
        current_groups = node_groups_per_level[-1]
        new_groups = AggregateGroups(current_groups, comm, C)
        node_groups_per_level.append(new_groups)
        A_curr = A_next

        # Early stopping by level-wise improvement
        candidate_labels = GroupsToLabels(new_groups, N)
        Q = Modularity(A, candidate_labels, gamma=gamma)
        if Q < prev_Q + tol:
            return GroupsToLabels(current_groups, N), prev_Q if prev_Q > -np.inf else Q
        prev_Q = Q

    # Max levels reached
    final_labels = GroupsToLabels(node_groups_per_level[-1], A.shape[0])
    Q = Modularity(A, final_labels, gamma=gamma)
    return final_labels, Q


# ---------- Scoring helpers ----------

def SnapshotModularity(A, labels, gamma=1.0, directed=False):
    """
    Compute modularity of a single snapshot (undirected or directed).

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix, shape (N, N)
        Adjacency matrix. For undirected, should be symmetric; diagonal ignored.
    labels : array-like, shape (N,)
        Community labels per node.
    gamma : float, default=1.0
        Resolution parameter (undirected or directed variants).
    directed : bool, default=False
        If True, uses directed modularity:
        Q = (1/m) sum_{ij} [ A_ij - γ (k_i^out k_j^in / m) ] δ(c_i, c_j)

    Returns
    -------
    Q : float
        Modularity of the partition on this snapshot.
    """
    A = A if sparse.issparse(A) else np.asarray(A, dtype=float)
    labs = np.asarray(labels)
    if sparse.issparse(A):
        if not directed:
            A = A.tocsr(copy=True)
            A.setdiag(0.0); A.eliminate_zeros()
            k = np.asarray(A.sum(axis=1)).ravel()
            m = k.sum()/2.0
            if m <= 0: return 0.0
            Q = 0.0
            _, inv, counts = np.unique(labs, return_inverse=True, return_counts=True)
            for c in range(counts.size):
                idx = np.where(inv == c)[0]
                if idx.size == 0: continue
                Asub = A[idx][:, idx]
                ksub = k[idx]
                e_c = Asub.sum()
                a_c = gamma * (ksub.sum()**2) / (2.0*m)
                Q += (e_c - a_c)
            return float(Q/(2.0*m))
        else:
            A = A.tocsr(copy=False)
            kout = np.asarray(A.sum(axis=1)).ravel()
            kin  = np.asarray(A.sum(axis=0)).ravel()
            m = A.sum()
            if m <= 0: return 0.0
            Q = 0.0
            _, inv, counts = np.unique(labs, return_inverse=True, return_counts=True)
            for c in range(counts.size):
                idx = np.where(inv == c)[0]
                Asub = A[idx][:, idx]
                e_c = Asub.sum()
                a_c = gamma * (kout[idx].sum()*kin[idx].sum())/m
                Q += (e_c - a_c)
            return float(Q/m)
    else:
        if not directed:
            A = A.astype(float, copy=True)
            np.fill_diagonal(A, 0.0)
            k = A.sum(axis=1)
            m = k.sum()/2.0
            if m <= 0: return 0.0
            same = (labs[:, None] == labs[None, :])
            expected = gamma * np.outer(k, k) / (2.0*m)
            return float(((A - expected)[same]).sum()/(2.0*m))
        else:
            kout = A.sum(axis=1)
            kin  = A.sum(axis=0)
            m = A.sum()
            if m <= 0: return 0.0
            same = (labs[:, None] == labs[None, :])
            expected = gamma * np.outer(kout, kin) / m
            return float(((A - expected)[same]).sum()/m)


# ---------- Public APIs (PascalCase) ----------

def ComputeSnapshotModularity(adj_matrix, labels=None, *, gamma=1.0,
                              directed=False, detect_method='louvain',
                              detect_kwargs=None):
    """
    Compute modularity for a single snapshot; auto-detect communities if needed.

    Parameters
    ----------
    adj_matrix : array-like or scipy.sparse.spmatrix, shape (N, N)
        Adjacency of the snapshot. Symmetric for undirected. Weights allowed.
    labels : None or array-like, shape (N,), default=None
        If provided, these labels are scored. If None, communities are detected
        using Louvain (undirected only) with the given `gamma`.
    gamma : float, default=1.0
        Modularity resolution parameter.
    directed : bool, default=False
        If True, uses directed modularity scoring (no auto-detect; pass labels).
    detect_method : {'louvain'}, default='louvain'
        Detection algorithm when `labels` is None.
    detect_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the detector (e.g., {'seed': 42}).

    Returns
    -------
    Q : float
        Modularity value for this snapshot.
    used_labels : np.ndarray, shape (N,)
        Labels that were scored (provided or auto-detected).
    """
    if labels is None:
        if directed:
            raise NotImplementedError("Auto-detection for directed graphs not implemented; provide labels.")
        if detect_method != 'louvain':
            raise ValueError("Only 'louvain' is supported in this helper.")
        detect_kwargs = detect_kwargs or {}
        labs, _Q = DetectCommunitiesLouvain(adj_matrix, gamma=gamma, **detect_kwargs)
        Q = SnapshotModularity(adj_matrix, labs, gamma=gamma, directed=False)
        return float(Q), labs.astype(int)
    else:
        labs = np.asarray(labels, dtype=int)
        Q = SnapshotModularity(adj_matrix, labs, gamma=gamma, directed=directed)
        return float(Q), labs


def ComputeTemporalModularity(adjacency_matrices, labels_per_snapshot=None, *,
                              gamma=1.0, directed=False, detect_method='louvain',
                              detect_kwargs=None):
    """
    Compute mean modularity across time; score provided labels or auto-detect per snapshot.

    Parameters
    ----------
    adjacency_matrices : np.ndarray or list, shape (T, N, N) or list of (N, N)
        Sequence of T snapshots (binary or weighted). Symmetric for undirected.
    labels_per_snapshot : None, (N,), or (T, N), default=None
        - None: auto-detect (Louvain) labels independently at each snapshot (undirected only).
        - (N,): a single static partition reused for all T snapshots.
        - (T, N): per-snapshot labels supplied by the caller.
    gamma : float, default=1.0
        Modularity resolution parameter.
    directed : bool, default=False
        If True, uses directed modularity scoring (no auto-detect).
    detect_method : {'louvain'}, default='louvain'
        Detection algorithm when labels are not supplied (undirected only).
    detect_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the detector (e.g., {'seed': 42}).

    Returns
    -------
    avg_Q : float
        Mean modularity over snapshots.
    Qs : np.ndarray, shape (T,)
        Per-snapshot modularity values.
    out_labels_per_snapshot : np.ndarray, shape (T, N)
        Labels used/scored at each snapshot (detected or provided).

    Notes
    -----
    - For a persistence-aware objective (ω coupling across time), use a multilayer
      optimizer/scorer instead of averaging snapshot modularities.
    """
    mats = adjacency_matrices
    if isinstance(mats, np.ndarray) and mats.ndim == 3 and not sparse.issparse(mats):
        mats_list = [mats[t] for t in range(mats.shape[0])]
    else:
        mats_list = list(mats)
    T = len(mats_list)
    detect_kwargs = detect_kwargs or {}

    if labels_per_snapshot is None:
        if directed:
            raise NotImplementedError("Auto-detection not implemented for directed graphs; provide labels_per_snapshot.")
        L = []
        for t in range(T):
            labs, _ = DetectCommunitiesLouvain(mats_list[t], gamma=gamma, **detect_kwargs)
            L.append(labs.astype(int))
        L = np.vstack(L)  # (T, N)
    else:
        L = np.asarray(labels_per_snapshot)
        if L.ndim == 1:
            L = np.repeat(L[None, :], T, axis=0)
        elif L.ndim != 2 or L.shape[0] != T:
            raise ValueError("labels_per_snapshot must be None, (N,), or (T,N).")

    Qs = np.empty(T, dtype=float)
    for t in range(T):
        Qs[t] = SnapshotModularity(mats_list[t], L[t], gamma=gamma, directed=directed)
    return float(Qs.mean()), Qs, L
