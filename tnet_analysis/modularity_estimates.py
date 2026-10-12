import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from numba import njit


def EnsureCsr(A):
    """
    Copy an adjacency matrix into canonical CSR format with zero diagonal.

    Args:
        A: array-like or scipy.sparse.spmatrix, shape (N, N). Binary or nonnegative weighted adjacency; directed inputs are also accepted here.

    Returns:
        A_csr: scipy.sparse.csr_matrix. Float adjacency with zero diagonal, duplicate entries summed and explicit zeros removed.

    Notes:
        - Never modifies the caller's dense or sparse input.
        - Non-finite or negative off-diagonal weights are rejected; signed modularity requires a separate null model.
    """
    if sparse.issparse(A):
        adjacency = sparse.csr_matrix(A, dtype=float, copy=True)
    else:
        dense = np.asarray(A)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError("A must be a square adjacency matrix.")
        # CSR owns its data; no dense float copy is needed to protect the input.
        adjacency = sparse.csr_matrix(dense, dtype=float, copy=True)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    adjacency.sum_duplicates()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    if not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0):
        raise ValueError("Off-diagonal weights must be finite and nonnegative.")
    return adjacency


def _ValidateGamma(gamma):
    """Validate the nonnegative resolution parameter."""
    if np.ndim(gamma) != 0 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative scalar.")
    return float(gamma)


def _ValidateUndirected(A):
    """Reject asymmetric graphs rather than silently treating them as undirected."""
    difference = A - A.T
    if difference.nnz:
        scale = float(np.max(A.data)) if A.nnz else 1.0
        if np.max(np.abs(difference.data)) > 1e-12 * scale:
            raise ValueError("Undirected modularity requires symmetric adjacency; use directed=True for scoring.")


def _CommunityIds(labels, N):
    """Validate one label per node and encode arbitrary community identifiers."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size != N:
        raise ValueError("labels must have shape (N,), with one label per node.")
    if np.issubdtype(labels.dtype, np.number) and not np.isfinite(labels).all():
        raise ValueError("Community labels must be finite.")
    if labels.dtype == object and any(label is None for label in labels):
        raise ValueError("Community labels cannot be missing.")
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


@njit(cache=True)
def _InternalWeight(indptr, indices, data, labels):
    """Sum within-community CSR weights without an N-by-N temporary matrix."""
    internal = 0.0
    for i in range(labels.size):
        for position in range(indptr[i], indptr[i+1]):
            if labels[i] == labels[indices[position]]:
                internal += data[position]
    return internal


def _ModularityCsr(A, labels, gamma, directed=False):
    """Score canonical CSR; retain self-loops on internal coarse graphs."""
    ids = _CommunityIds(labels, A.shape[0])
    degree_out = np.asarray(A.sum(axis=1)).ravel()
    mass = float(degree_out.sum())
    if not np.isfinite(mass):
        raise ValueError("Total edge weight must be finite; rescale excessively large weights.")
    if mass <= 0:
        return 0.0
    out_totals = np.bincount(ids, weights=degree_out) / mass
    if directed:
        degree_in = np.asarray(A.sum(axis=0)).ravel()
        in_totals = np.bincount(ids, weights=degree_in) / mass
    else:
        in_totals = out_totals
    internal = _InternalWeight(A.indptr, A.indices, A.data, ids)
    return float(internal / mass - gamma * np.dot(out_totals, in_totals))


def CoarseGrain(A, comm, C):
    """
    Sum current graph weights between communities into a coarse graph.

    Args:
        A: scipy.sparse.spmatrix, shape (N, N). Current-level adjacency, including any coarse self-loops.
        comm: array-like, shape (N,). Integer community identifiers in 0..C-1.
        C: int. Number of communities.

    Returns:
        A_coarse: scipy.sparse.csr_matrix, shape (C, C). Canonical coarse graph, retaining self-loops and total weight.

    Notes:
        - COO coordinates are paired directly with their weights, including explicit-zero or duplicate input entries.
    """
    adjacency = sparse.coo_matrix(A, dtype=float, copy=True)
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A must be square.")
    comm = np.asarray(comm)
    if isinstance(C, (bool, np.bool_)) or not isinstance(C, (int, np.integer)) or C < 0:
        raise ValueError("C must be a nonnegative integer.")
    if comm.shape != (adjacency.shape[0],) or not np.issubdtype(comm.dtype, np.integer):
        raise ValueError("comm must contain one integer community id per node.")
    if comm.size and (comm.min() < 0 or comm.max() >= C):
        raise ValueError("Community ids must lie in 0..C-1.")
    coarse = sparse.coo_matrix((adjacency.data, (comm[adjacency.row], comm[adjacency.col])), shape=(C, C)).tocsr()
    coarse.sum_duplicates()
    coarse.eliminate_zeros()
    coarse.sort_indices()
    return coarse


@njit(cache=True)
def _LouvainMovePass(indptr, indices, data, degree, comm, totals, order, gamma, mass):
    """Apply only positive modularity-gain moves; coarse loops cancel in gains."""
    weights = np.zeros(comm.size)
    touched = np.empty(comm.size, dtype=np.int64)
    moves = 0
    for i in order:
        if degree[i] == 0:
            continue
        count = 0
        for position in range(indptr[i], indptr[i+1]):
            neighbor = indices[position]
            if neighbor == i:
                continue
            community = comm[neighbor]
            if weights[community] == 0:
                touched[count] = community
                count += 1
            weights[community] += data[position]
        current = comm[i]
        totals[current] -= degree[i]
        # Compare insertion into a candidate with reinsertion into the current
        # community. Comparing against zero can miss valid improving moves.
        best = current
        best_gain = weights[current] - gamma * degree[i] * (totals[current] / mass)
        for position in range(count):
            candidate = touched[position]
            gain = weights[candidate] - gamma * degree[i] * (totals[candidate] / mass)
            if gain > best_gain + 1e-14 * mass:
                best, best_gain = candidate, gain
        comm[i] = best
        totals[best] += degree[i]
        if best != current:
            moves += 1
        for position in range(count):
            weights[touched[position]] = 0.0
    return moves


def _SplitDisconnectedCommunities(A, labels):
    """Refine disconnected Louvain groups without lowering nonnegative-gamma Q."""
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    within = A.copy()
    within.data[labels[rows] != labels[within.indices]] = 0
    within.eliminate_zeros()
    return connected_components(within, directed=False, return_labels=True)[1]


def _RunLouvain(A, gamma, max_passes, max_levels, tol, rng):
    """One restart, tracking original-node labels without nested node-group lists."""
    labels = np.arange(A.shape[0], dtype=np.int64)
    best_labels = labels.copy()
    best_Q = _ModularityCsr(A, labels, gamma)
    current = A
    for _ in range(max_levels):
        size = current.shape[0]
        comm = np.arange(size, dtype=np.int64)
        degree = np.asarray(current.sum(axis=1)).ravel()
        mass = float(degree.sum())
        totals = degree.copy()
        for _ in range(max_passes):
            moves = _LouvainMovePass(current.indptr, current.indices, current.data,
                                    degree, comm, totals, rng.permutation(size), gamma, mass)
            if moves == 0:
                break
        _, comm = np.unique(comm, return_inverse=True)
        candidate = comm[labels]
        Q = _ModularityCsr(current, comm, gamma)
        improvement = Q - best_Q
        if Q >= best_Q:
            best_labels, best_Q = candidate.copy(), Q
        communities = int(comm.max()) + 1
        if communities == size or improvement <= tol:
            break
        labels = candidate
        current = CoarseGrain(current, comm, communities)
    # A known Louvain limitation is a disconnected returned community. Splitting
    # it preserves internal edge weight and cannot increase the null-model term.
    refined = _SplitDisconnectedCommunities(A, best_labels)
    refined_Q = _ModularityCsr(A, refined, gamma)
    return refined, refined_Q


def DetectCommunitiesLouvain(A, gamma=1.0, max_passes=100, max_levels=10,
                            tol=1e-7, seed=None, *, n_restarts=1):
    """
    Detect undirected weighted communities with sparse multilevel Louvain.

    Args:
        A: array-like or scipy.sparse.spmatrix, shape (N, N). Symmetric binary or nonnegative weighted adjacency; original diagonal ignored.
        gamma: float, optional (default=1.0). Nonnegative resolution parameter; larger values favor smaller communities.
        max_passes: int, optional (default=100). Maximum node sweeps at each level.
        max_levels: int, optional (default=10). Maximum aggregation levels per restart.
        tol: float, optional (default=1e-7). Minimum modularity improvement to continue to another level.
        seed: int or None, optional (default=None). Seed controlling node visitation order.
        n_restarts: int, keyword-only, optional (default=1). Independent visitation sequences; return the highest-Q partition found.

    Returns:
        labels: np.ndarray, shape (N,), dtype=int. Community assignments of original nodes.
        Q: float. Modularity of the returned partition on the original zero-diagonal graph.

    Notes:
        - Compiled sparse node sweeps avoid Python per-edge dictionaries; coarse self-loops are retained but excluded from move-edge totals.
        - Disconnected detected communities are split into connected components.
        - Louvain remains a heuristic with a modularity resolution limit; restarts do not guarantee a global optimum or constitute the Leiden algorithm.
        - A fixed seed gives reproducible results in this implementation.
        - Reference: Blondel et al. (2008), https://arxiv.org/abs/0803.0476.
    """
    gamma = _ValidateGamma(gamma)
    for name, value in (("max_passes", max_passes), ("max_levels", max_levels), ("n_restarts", n_restarts)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if np.ndim(tol) != 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative scalar.")
    adjacency = EnsureCsr(A)
    _ValidateUndirected(adjacency)
    if adjacency.nnz == 0:
        return np.arange(adjacency.shape[0], dtype=int), 0.0
    rng = np.random.default_rng(seed)
    best_labels, best_Q = None, -np.inf
    for _ in range(n_restarts):
        labels, Q = _RunLouvain(adjacency, gamma, max_passes, max_levels, tol, rng)
        if Q > best_Q:
            best_labels, best_Q = labels, Q
    return best_labels, float(best_Q)


def SnapshotModularity(A, labels, gamma=1.0, directed=False):
    """
    Score a supplied snapshot partition using undirected or directed modularity.

    Args:
        A: array-like or scipy.sparse.spmatrix, shape (N, N). Binary or nonnegative weighted adjacency; diagonal ignored in both modes.
        labels: array-like, shape (N,). Community identifiers per node.
        gamma: float, optional (default=1.0). Finite nonnegative resolution parameter.
        directed: bool, optional (default=False). Use out-strength times in-strength in the null model instead of the undirected strength product.

    Returns:
        Q: float. Snapshot modularity; zero for an edgeless graph.

    Notes:
        - Does not modify inputs or densify sparse graphs.
        - Negative weights need signed modularity and are rejected, not silently discarded.
    """
    gamma = _ValidateGamma(gamma)
    adjacency = EnsureCsr(A)
    if not directed:
        _ValidateUndirected(adjacency)
    return _ModularityCsr(adjacency, labels, gamma, directed)


def ComputeSnapshotModularity(adj_matrix, labels=None, *, gamma=1.0,
                              directed=False, detect_method='louvain',
                              detect_kwargs=None):
    """
    Score a snapshot partition, optionally detecting communities first.

    Args:
        adj_matrix: array-like or scipy.sparse.spmatrix, shape (N, N). Binary or nonnegative weighted adjacency; diagonal ignored.
        labels: None or array-like, shape (N,), optional (default=None). Supplied partition or automatic undirected detection.
        gamma: float, optional (default=1.0). Nonnegative modularity resolution parameter.
        directed: bool, optional (default=False). Directed scoring requires supplied labels.
        detect_method: {"louvain"}, optional (default="louvain"). Automatic community detection method.
        detect_kwargs: dict or None, optional (default=None). Detector options such as seed, max_passes or n_restarts.

    Returns:
        Q: float. Snapshot modularity.
        used_labels: np.ndarray, shape (N,). Detected integer labels or an unmodified copy of supplied identifiers.
    """
    if labels is None:
        if directed:
            raise NotImplementedError("Directed detection is not implemented; provide labels.")
        if detect_method != 'louvain':
            raise ValueError("Only 'louvain' detection is supported.")
        used, Q = DetectCommunitiesLouvain(adj_matrix, gamma=gamma, **(detect_kwargs or {}))
        return Q, used
    used = np.array(labels, copy=True)
    return SnapshotModularity(adj_matrix, used, gamma, directed), used


def ComputeTemporalModularity(adjacency_matrices, labels_per_snapshot=None, *,
                              gamma=1.0, directed=False, detect_method='louvain',
                              detect_kwargs=None):
    """
    Compute the arithmetic mean of independent snapshot modularities.

    Args:
        adjacency_matrices: np.ndarray or list, shape (T, N, N) or list of (N, N). Binary or nonnegative weighted snapshots, dense or sparse.
        labels_per_snapshot: None, (N,), or (T, N), optional (default=None). Detect independently per frame, reuse one supplied partition, or score frame-specific partitions.
        gamma: float, optional (default=1.0). Nonnegative modularity resolution parameter.
        directed: bool, optional (default=False). Directed scoring requires supplied partitions.
        detect_method: {"louvain"}, optional (default="louvain"). Automatic detection method.
        detect_kwargs: dict or None, optional (default=None). Detector options, e.g. {"seed": 42, "n_restarts": 5}.

    Returns:
        avg_Q: float. Mean snapshot modularity; NaN for an empty sequence.
        Qs: np.ndarray, shape (T,). Snapshot modularity values.
        out_labels_per_snapshot: np.ndarray, shape (T, N). Detected or supplied labels used for each snapshot.

    Notes:
        - This is not multilayer modularity: there is no inter-frame coupling or community-identity matching across snapshots.
        - Edgeless frames contribute zero to the mean. Detection and scoring share one path, avoiding redundant per-frame scoring.
        - The same explicitly supplied seed is forwarded to each snapshot for reproducibility.
    """
    gamma = _ValidateGamma(gamma)
    if isinstance(adjacency_matrices, np.ndarray):
        if adjacency_matrices.ndim != 3 or adjacency_matrices.shape[1] != adjacency_matrices.shape[2]:
            raise ValueError("adjacency_matrices must have shape (T, N, N).")
        matrices = adjacency_matrices
        N = matrices.shape[1]
    else:
        matrices = list(adjacency_matrices)
        N = matrices[0].shape[0] if matrices else 0
        if any(matrix.shape != (N,N) for matrix in matrices):
            raise ValueError("All snapshots must have the same square shape.")
    T = len(matrices)
    provided = None if labels_per_snapshot is None else np.asarray(labels_per_snapshot)
    if provided is not None:
        if provided.shape == (N,):
            _CommunityIds(provided,N)
        elif provided.shape != (T,N):
            raise ValueError("labels_per_snapshot must have shape (N,) or (T, N).")
    if provided is None and (directed or detect_method != 'louvain'):
        if directed:
            raise NotImplementedError("Directed detection is not implemented; provide labels_per_snapshot.")
        raise ValueError("Only 'louvain' detection is supported.")
    Qs = np.empty(T, dtype=float)
    labels = np.empty((T,N),dtype=int if provided is None else provided.dtype)
    for t in range(T):
        current_labels = None if provided is None else provided if provided.ndim == 1 else provided[t]
        Qs[t], labels[t] = ComputeSnapshotModularity(matrices[t], current_labels,
            gamma=gamma, directed=directed, detect_method=detect_method, detect_kwargs=detect_kwargs)
    return float(Qs.mean()) if T else np.nan, Qs, labels
