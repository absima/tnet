"""Shared reductions for nodewise measure scores."""
import numpy as np


def SummarizeScores(values, summary_stat=None):
    """
    Summarize node scores with percentiles or a requested statistic.

    Args:
        values: array-like. Numeric scores; NaNs are ignored by the reductions.
        summary_stat: {None, "percentiles", "mean", "median", "both"}, optional (default=None).

    Returns:
        summary: list[float] or float. Five percentiles (5,25,50,75,95),
            one scalar, or [mean, median].

    Notes:
        All-NaN inputs return NaN summaries and issue NumPy warnings.
    """
    if summary_stat is None or summary_stat == "percentiles":
        return np.nanpercentile(values, [5, 25, 50, 75, 95]).tolist()
    if summary_stat == "both":
        return [float(np.nanmean(values)), float(np.nanmedian(values))]
    if summary_stat == "median":
        return float(np.nanmedian(values))
    if summary_stat == "mean":
        return float(np.nanmean(values))
    raise ValueError(
        "summary_stat must be None, 'percentiles', 'mean', 'median', or 'both'."
    )
