"""Compatibility imports for the optimized return-latency implementation.

Import ComputeReturnLatency from latency_measures for the canonical entry point.
Both names below refer to that same optimized function.
"""
from .latency_measures import ComputeReturnLatency, ComputeCirculationLatency
from ._summary import SummarizeScores as _summarize_scores
