"""Compiled statistics for partner, memory and burstiness measures."""
import numpy as np
from numba import njit
from scipy.fft import rfft, irfft, next_fast_len
from scipy.stats import binom
from .modularity_estimates import EnsureCsr


def ValidateTemporalAdjacency(adjacency, *, undirected=False):
    """Validate framewise to avoid a full T-by-N-by-N boolean temporary."""
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2] or adjacency.shape[1] == 0:
        raise ValueError("adjacency_matrices must have shape (T, N, N), with N > 0.")
    if adjacency.dtype.kind not in "buif":
        raise ValueError("Adjacency must be real numeric or boolean.")
    for frame in adjacency:
        if not np.isfinite(frame).all() or np.any(frame < 0):
            raise ValueError("Adjacency must be finite and nonnegative.")
        if undirected and not np.array_equal(frame > 0, frame.T > 0):
            raise ValueError("This measure requires symmetric edge presence.")
    return adjacency


def PositiveInteger(value, name):
    """Reject accidental fractional, boolean or nonpositive time parameters."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def SnapshotAdjacency(adjacency):
    """Canonical sparse, nonnegative undirected snapshot; diagonal ignored."""
    adjacency = EnsureCsr(adjacency)
    difference = adjacency - adjacency.T
    scale = np.max(adjacency.data) if adjacency.nnz else 1.0
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12 * scale:
        raise ValueError("This measure requires symmetric adjacency.")
    return adjacency


@njit(cache=True)
def SnapshotTransitivity(indptr, indices):
    """Count triangles once by intersecting sorted upper neighbor lists."""
    triangles = 0
    triplets = 0
    for i in range(len(indptr) - 1):
        degree = indptr[i + 1] - indptr[i]
        triplets += degree * (degree - 1) // 2
        for edge in range(indptr[i], indptr[i + 1]):
            j = indices[edge]
            if j <= i:
                continue
            left = edge + 1
            right = indptr[j]
            while left < indptr[i + 1] and right < indptr[j + 1]:
                a, b = indices[left], indices[right]
                if a == b:
                    triangles += 1
                    left += 1
                    right += 1
                elif a < b:
                    left += 1
                else:
                    right += 1
    return 3.0 * triangles / triplets if triplets else 0.0


@njit(cache=True)
def ParticipationScores(indptr, indices, data, labels, communities, eps):
    scores = np.full(len(indptr) - 1, np.nan)
    strengths = np.zeros(communities)
    for i in range(len(scores)):
        strengths[:] = 0.0
        total = 0.0
        for edge in range(indptr[i], indptr[i + 1]):
            weight = data[edge]
            strengths[labels[indices[edge]]] += weight
            total += weight
        if total > eps:
            squares = 0.0
            for strength in strengths:
                squares += (strength / total) ** 2
            scores[i] = max(0.0, 1.0 - squares)
    return scores


@njit(cache=True)
def PartnerCountsAndStability(adjacency, gap_normalization=False):
    T, N, _ = adjacency.shape
    counts = np.zeros((N, N), dtype=np.int64)
    first = np.full((N, N), -1, dtype=np.int64)
    last = first.copy()
    for t in range(T):
        for i in range(N):
            for j in range(N):
                if i != j and adjacency[t, i, j] > 0:
                    if counts[i, j] == 0:
                        first[i, j] = t
                    last[i, j] = t
                    counts[i, j] += 1
    scores = np.zeros(N)
    for i in range(N):
        intervals = 0
        total_gap = 0
        for j in range(N):
            if counts[i, j] > 1:
                intervals += counts[i, j] - 1
                total_gap += last[i, j] - first[i, j]
        if intervals:
            scores[i] = intervals / total_gap if gap_normalization else 1.0 - total_gap / (intervals * T)
    return counts, scores


@njit(cache=True)
def PartnerDiversityScores(adjacency, window_size, available):
    T, N, _ = adjacency.shape
    coverage = np.zeros((N, N), dtype=np.int64)
    active = np.zeros((N, N), dtype=np.bool_)
    for start in range(0, T, window_size):
        active[:] = False
        for t in range(start, min(start + window_size, T)):
            for i in range(N):
                for j in range(N):
                    if i != j and adjacency[t, i, j] > 0:
                        active[i, j] = True
        for i in range(N):
            for j in range(N):
                coverage[i, j] += active[i, j]
    scores = np.zeros(N)
    for i in range(N):
        partners = 0
        total = 0
        for j in range(N):
            if coverage[i, j]:
                partners += 1
                total += coverage[i, j]
        if partners > 1:
            entropy = 0.0
            for j in range(N):
                if coverage[i, j]:
                    p = coverage[i, j] / total
                    entropy -= p * np.log(p)
            scores[i] = min(1.0, entropy / np.log(N - 1 if available else partners))
    return scores


@njit(cache=True)
def EdgeOverlapAndPersistence(adjacency):
    T, N, _ = adjacency.shape
    overlap = persistence = 0.0
    overlap_count = persistence_count = 0
    for t in range(T - 1):
        union = intersection = active = 0
        for i in range(N):
            for j in range(i + 1, N):
                a = adjacency[t, i, j] > 0
                b = adjacency[t + 1, i, j] > 0
                union += a or b
                intersection += a and b
                active += a
        if union:
            overlap += intersection / union
            overlap_count += 1
        if active:
            persistence += intersection / active
            persistence_count += 1
    return (overlap / overlap_count if overlap_count else np.nan,
            persistence / persistence_count if persistence_count else np.nan)


@njit(cache=True)
def NeighborhoodMemoryScores(adjacency, lag):
    T, N, _ = adjacency.shape
    totals = np.zeros(N)
    observations = np.zeros(N, dtype=np.int64)
    for t in range(max(0, T - lag)):
        for i in range(N):
            intersection = union = 0
            for j in range(N):
                if j != i:
                    a = adjacency[t, i, j] > 0
                    b = adjacency[t + lag, i, j] > 0
                    intersection += a and b
                    union += a or b
            if union:
                totals[i] += intersection / union
                observations[i] += 1
    for i in range(N):
        if observations[i]:
            totals[i] /= observations[i]
    return totals


@njit(cache=True)
def LinkBurstinessScores(adjacency, onsets):
    T, N, _ = adjacency.shape
    last = np.full((N, N), -1, dtype=np.int64)
    gaps = np.zeros((N, N), dtype=np.int64)
    mean = np.zeros((N, N))
    m2 = np.zeros((N, N))
    for t in range(T):
        for i in range(N):
            for j in range(i + 1, N):
                if adjacency[t, i, j] > 0:
                    if onsets and t > 0 and adjacency[t - 1, i, j] > 0:
                        continue
                    if last[i, j] >= 0:
                        gap = t - last[i, j]
                        gaps[i, j] += 1
                        delta = gap - mean[i, j]
                        mean[i, j] += delta / gaps[i, j]
                        m2[i, j] += delta * (gap - mean[i, j])
                    last[i, j] = t
    result = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(i + 1, N):
            if gaps[i, j] >= 2:
                sigma = np.sqrt(max(0.0, m2[i, j] / gaps[i, j]))
                value = (sigma - mean[i, j]) / (sigma + mean[i, j])
                result[i, j] = result[j, i] = value
    return result


def ConditionalPersistencePValues(counts, T, probability):
    """Exact conditional tails under a specified iid Bernoulli edge model.

    Given K ever-active partners, each of their T-frame counts is a
    zero-truncated binomial. Convolve that distribution K times by FFT;
    shifting each count down by one keeps polynomial support smaller.
    """
    partners = np.count_nonzero(counts, axis=1)
    totals = counts.sum(axis=1)
    result = np.full(len(partners), np.nan)
    if T == 0:
        return result
    valid = partners > 0
    if probability == 0:
        result[valid] = 0.0
        return result
    if T == 1 or probability == 1:
        result[valid] = 1.0
        return result
    base = binom.pmf(np.arange(1, T + 1), T, probability)
    base /= base.sum()
    for K in np.unique(partners[valid]):
        length = int(K) * (T - 1) + 1
        nfft = next_fast_len(length)
        mass = irfft(rfft(base, nfft) ** int(K), nfft)[:length]
        mass = np.maximum(mass, 0.0)
        mass /= mass.sum()
        tail = np.cumsum(mass[::-1])[::-1]
        nodes = np.flatnonzero(partners == K)
        result[nodes] = np.minimum(1.0, tail[totals[nodes] - K])
    return result
