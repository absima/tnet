import numpy as np
import scipy.io
import random

def compute_dynamic_functional_connectivity(tseries, window_size, lag):
    """
    Compute dynamic functional connectivity (DFC) matrices from a timeseries.

    Parameters
    ----------
    tseries : array, shape (T_total, N)
        Time × node matrix (raw time series data).
    window_size : int
        Size of the sliding window for computing correlations.
    lag : int
        Step size for moving the sliding window.

    Returns
    -------
    fcs : array, shape (n_snapshots, N, N)
        Sequence of correlation matrices (dynamic functional connectivity).
    """
    T_total, N = tseries.shape
    
    n_snapshots = int(np.floor((T_total - window_size) / lag) + 1)
    
    fcs = np.zeros((n_snapshots, N, N))

    for i in range(n_snapshots):
        shift = i * lag
        windowed_data = tseries[shift : shift + window_size]
        corr_matrix = np.corrcoef(windowed_data.T)
        fcs[i] = corr_matrix

    return fcs


def binarize_connectivity_matrices(fcs, average_degree, binsize, mode='dynamic'):
    """
    Binarize functional connectivity matrices based on target average degree.

    Parameters
    ----------
    fcs : array, shape (..., N, N)
        Sequence of weight matrices (correlation or other).
    average_degree : float
        Desired average degree per node after binarization.
    binsize : int
        Number of bins for histogram thresholding.
    mode : str, optional
        - 'dynamic' (default): binarize each frame separately.
        - 'static': sum matrices first, binarize based on total.

    Returns
    -------
    tnet : array, shape same as fcs
        Binarized adjacency matrices (0/1).
    """
    mtx = np.copy(fcs)
    N = mtx.shape[-1]

    if mode == 'static':
        mtx = np.sum(mtx, axis=0)
        mtx = np.repeat([mtx], len(fcs), axis=0)

    src, tgt = np.tril_indices(N, -1)
    fcflat = mtx[..., src, tgt]
    fcflat = np.abs(fcflat.flatten())  # Absolute values (include strong anticorrelations)

    values, base = np.histogram(fcflat, bins=binsize)
    cumulative = np.cumsum(values)

    rho = average_degree / N
    cutoff_idx = np.where(cumulative > (1 - rho) * len(fcflat))[0][0]
    threshold = base[cutoff_idx]

    tnet = np.copy(mtx)
    tnet[tnet < threshold] = 0
    tnet[tnet >= threshold] = 1

    return tnet


def trim_isolated_nodes(tnet):
    """
    Remove nodes with no connections across the entire temporal network.

    Parameters
    ----------
    tnet : array, shape (T, N, N)
        Sequence of binary adjacency matrices.

    Returns
    -------
    trimmed_tnet : array
        Temporal network after removing isolated nodes.
    active_nodes : array
        Indices of active (non-isolated) nodes.
    total_node_links : array
        Total number of links for each node across time.
    """
    N = tnet.shape[-1]
    
    cumsum = np.sum(tnet, axis=0)
    np.fill_diagonal(cumsum, 0)  # Ignore self-links
    total_node_links = np.sum(cumsum, axis=0).astype(int)

    active_nodes = np.where(total_node_links != 0)[0]

    if len(active_nodes) < N:
        isolated_nodes = np.setdiff1d(np.arange(N), active_nodes)
        tnet = np.delete(tnet, isolated_nodes, axis=1)
        tnet = np.delete(tnet, isolated_nodes, axis=2)

    return tnet, active_nodes, total_node_links