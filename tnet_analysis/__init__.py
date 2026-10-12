"""Temporal network analysis toolkit.

This package groups temporal-network utilities into five areas:

1. Signal surrogates, FC estimation, and temporal-network preparation.
2. Temporal network generators and null models.
3. Distance and return metrics.
4. Dynamism and memory metrics.
5. Segregation and cohesion metrics.

Pipeline:
    Time series -> optional signal surrogates -> FC -> temporal networks ->
    null or synthetic networks ->
    distance and return metrics -> dynamism and memory metrics ->
    segregation and cohesion metrics.
"""
