Output layouts and summaries
============================

Distance matrices
-----------------

``SmartWalker`` returns an (N,N) matrix: NaN on the diagonal, infinity for
unreachable pairs, and finite arrival times elsewhere.

``RandomWalker`` returns (matrix, trial-averaged metrics), or additionally
an (trials,N,N) tensor when ``return_fpt_matrix=True``. Its matrix averages
only successful first passages; a pair is infinite if no trial reaches it.
The metrics summarize each trial first, then average trials. They need not
equal a summary recomputed from the returned mean matrix.
Likewise, a drunk median entry is the average of trial-specific medians,
not the median of pooled first-passage observations.

Distance vectors
----------------

Smart/drunk measure functions default to ``summary_stat="mean"``:

1. impermeability
2. latency
3. resistance
4. inaccessibility
5. irrigation

With ``"both"``, the eight entries are:

1. impermeability mean
2. impermeability median
3. latency mean
4. latency median
5. resistance mean
6. resistance median
7. inaccessibility
8. irrigation

Counts occur once. Drunk counts can be fractional because they average trials.
The impermeability median uses finite source-normalized pair entries.
Do not interpret every median as a median over nodes.

Dynamism order
--------------

``ComputeDynamism`` returns transition probability, global entropy,
cosine similarity, net fluidity, mutual information and mean edge count.
These are six scalars, not six node-score distributions.

Node-score summaries
--------------------

For functions supporting all summary options:

* ``None`` or ``"percentiles"``: [p05,p25,p50,p75,p95].
* ``"mean"`` or ``"median"``: one scalar.
* ``"both"``: [mean,median].

Here the median of node scores is not necessarily the median of all
underlying frames or events. Return latency summarizes nodewise mean
latencies. Node persistence adds its exceedance count to the summary.
Return latency returns (summary,mean successful-return count), unless
nodewise details are requested. Burstiness returns a matrix, not this
summary schema. Modularity and participation have their own outputs;
consult the API rather than assuming one universal shape.

NaN, infinity and padding
-------------------------

NaN often represents an undefined score or excluded diagonal. Infinity in
distance matrices represents non-arrival within the horizon. Zero is a
valid measure value, not universally missing data.

Use NaN-aware population summaries only after recording missing-data counts.
Do not silently turn infinity into zero. Resistance already handles
non-arrivals via its explicit penalty.

The example ``funcIntSeg`` wrapper has seven metric groups and pads every
vector to width eight. Their meaningful widths are [8,8,6,2,2,3,2].
Padding zeros are not observations. Use ``METRIC_SPECS`` output names to
identify meaningful entries, and preserve those names with saved arrays.
Package functions themselves do not add wrapper padding.

Save node order, subject IDs, achieved degrees, tags, FC method, time
settings, seeds, units and trial counts alongside results.

Aliases
-------

``ComputeReturnability`` aliases ``ComputeRetention``;
``ComputeCirculationLatency`` aliases ``ComputeReturnLatency``;
``ComputeStaticClustering`` aliases ``ComputeSpatialClustering``;
``LinkBurstiness`` aliases ``ComputeLinkBurstiness``.
Each pair refers to the same implementation. The canonical names are used
throughout the guide.
