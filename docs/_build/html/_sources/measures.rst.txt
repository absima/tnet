Choosing and interpreting measures
==================================

Distance and integration
------------------------

Smart latency is earliest arrival from each source starting at the first
frame. It explores all available time-respecting options; it is not a
single sampled walk or an average over every possible start time.
Smart propagation is deterministic for a fixed network and parameter set;
its accepted random_seed argument does not affect the search.
Drunk latency samples random walks and first-passage times. More trials
reduce Monte Carlo variation but increase cost. A finite horizon can leave
pairs unvisited even if another possible path exists.

Distance summaries contain:

* **Impermeability:** distance divided by the source's number of reached
  targets, aggregated over finite pairs. It combines timing and coverage;
  it is not just inverse irrigation.
* **Latency:** finite pairwise arrival times only.
* **Resistance:** arrival times with unreachable pairs replaced by a penalty.
* **Inaccessibility:** number of unreachable ordered off-diagonal pairs.
* **Irrigation:** number of reached ordered off-diagonal pairs.

With T frames, the default penalty is T+1 for frames and frame fractions,
or h*(T+1) for substeps. Keep penalty conventions consistent across
comparisons. A larger penalty directly increases resistance when pairs
are unreachable. Latency alone excludes those pairs and can look small
in a poorly connected network. Read it alongside irrigation.

For N nodes, ordered-pair counts have denominator N*(N-1), not N*(N-1)/2.
Changing network size changes raw counts; report any normalization explicitly.

Return latency and retention
----------------------------

**Return latency** searches for first non-waiting returns to the starting
node, across eligible start frames. It summarizes each node's mean return
latency across successful starts. A waiting self-loop alone is not a
successful return. The accompanying count averages successful start-frame
returns over included nodes: it is **not a count of distinct routes**.
Use ``return_full=True`` for nodewise means, minima, hit counts and
included-node indices.

**Retention** is the fraction of current neighbors encountered in eligible
past frames, averaged over eligible nonempty observations per node.
It describes recurring partners, not travel back to a source.
Nodes with fewer than two nonempty observations have NaN retention.
Its lifetime parameter and return latency's search limit answer different
questions.

Dynamism and memory
-------------------

``ComputeDynamism`` returns six scalars in the order given in :doc:`outputs`:

* **Transition probability:** fraction of possible edges changing state
  between consecutive frames.
* **Global entropy:** normalized Bernoulli entropy of edge activation
  probabilities across time. Persistent and never-active edges contribute zero.
* **Cosine similarity:** similarity of consecutive edge-state vectors.
  Two empty frames have similarity one; an empty/nonempty pair has zero.
* **Net fluidity:** global entropy multiplied by (1-cosine similarity).
  Similarity and fluidity are related but not generally exact opposites.
* **Mutual information:** mean normalized mutual information of consecutive
  binary edge-state vectors; this is not mutual information between signals.
* **Mean edge count:** average number of undirected off-diagonal edges.

**Edge overlap** is consecutive edge-set Jaccard, skipping empty unions.
**Edge persistence rate** averages the fraction of source-frame edges
surviving into the next frame. Frames receive equal weight; this is not
a pooled edge-weighted survival fraction.
**Neighborhood memory** is nodewise neighborhood Jaccard at a chosen frame
lag. Empty unions are skipped; nodes with no eligible comparisons score zero.

**Link burstiness** is (gap standard deviation - mean gap)/
(gap standard deviation + mean gap), using population standard deviation.
Regular gaps give -1; variable gaps can give positive values.
``event_mode="activations"`` treats every active frame as an event;
``"onsets"`` treats a contiguous run as one event.
At least three events are needed; other entries and the diagonal are NaN.
Short recordings, censoring and few events limit interpretation.

Clustering, partners and communities
------------------------------------

**Spatial clustering** measures connections among each node's neighbors,
then averages each node across frames. Degree below two contributes zero.
**Temporal clustering** measures temporally ordered triangle closure,
with the delta rules in :doc:`data_and_parameters`. It is not spatial
clustering applied to an aggregated graph.
**Transitivity** is global triangle closure relative to connected triples;
it is not mean local clustering. Nonnegative weights define edge presence.

**Partner stability** defaults to 1-mean recurrence gap/T. All recurrence
intervals receive equal weight; nodes without recurring partners score zero.
Its duration normalization makes recording lengths important.
``normalization="gap"`` instead returns reciprocal mean gap per frame,
not a next-frame survival probability.

**Partner diversity** is entropy of partners' active-window coverage.
Default ``normalization="observed"`` measures evenness: two equally
covered partners can score one, just as many equally covered partners can.
``"available"`` divides by log(N-1) to incorporate partner breadth.
Zero/one-partner nodes score zero; the final partial window is included.

**Node persistence** averages occupancy over ever-connected partners.
It also returns a count exceeding an iid conditional-binomial null.
``confidence`` sets the upper-tail threshold; ``null_probability``
supplies p, otherwise density estimates p. ``return_details=True``
adds scores and tail probabilities. This null assumes independent,
homogeneous links and independent frames. Overlapping FC windows,
autocorrelation and estimated p do not yield calibrated significance.
There is no automatic multiple-testing correction.

**Participation** is 1-sum of squared community-strength fractions.
It measures spread of connections across the supplied partition; isolates
have NaN scores. **Modularity** scores within-community connectivity
relative to a strength-based null, with resolution parameter gamma.
Gamma defaults to 1; increasing it generally favors finer partitions,
although a heuristic search need not change monotonically on every run.
Louvain is heuristic: seeds and restarts matter. Temporal modularity is
the mean of independent snapshot modularities, not a multilayer model
with inter-frame community coupling.
