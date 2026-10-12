Inputs and time parameters
==========================

Shapes and conventions
----------------------

* One subject's signals: ``(L, N)`` — samples, nodes.
* Connectivity or adjacency: ``(T, N, N)`` — frames, nodes, nodes.
* One snapshot: ``(N, N)``.
* Subject batches: ``(subjects, L, N)``; analyze individual subjects unless
  the runner explicitly supports batching.

Move the time axis before passing a ``(nodes, nodes, frames)`` tensor.
Most measures assume symmetric, undirected frames. Not every function
validates every assumption: check finite values, symmetry and binary
edge values. Signed FC is not a binary adjacency tensor.

Structure measures should receive zero diagonals. Smart and random walkers
force waiting diagonals internally. Give return latency a waiting-diagonal
network. Give ``ComputeDynamism`` diagonal ones too: its edge-count
calculation subtracts one self-entry per node per frame. Its other five
components use off-diagonal edges.

Average degree counts neighbors, not weighted strength. Modularity and
participation support nonnegative weights. Directed modularity scoring
requires supplied labels and ``directed=True``; automatic Louvain detection
is undirected. This does not imply directed support for other measures.

Window and lag
--------------

For Pearson FC, ``window`` counts signal samples; ``lag`` is the step
between window starts. The dispatch function defaults to 20 and 1.
The lower-level Pearson function requires both arguments.
Use positive integers, at least two samples per correlation window and
a recording longer than the window.

This implementation returns ``int((L-window)/lag)`` Pearson frames.
It does not add an endpoint window: 60 samples, window 20 and lag 5 yield
eight frames, not nine. Instantaneous FC yields one frame per signal sample,
ignoring window and lag.

Neighborhood memory's lag counts **adjacency frames**. Partner diversity's
``window_size`` groups adjacency frames, independently of the FC window.

Hops and latency units
----------------------

``hops_per_frame=1`` is the default. Larger values permit multiple substeps
in each frame. Smart propagation explores reachability; a random walk
selects one available neighbor, including waiting, at each substep.

For first arrival at substep s with hop allowance h:

* ``latency_unit="frames"`` returns ceil(s/h).
* ``"frame_fraction"`` returns s/h.
* ``"substeps"`` returns s.

Substeps include waiting, not just non-self edge traversals. Fractional-frame
timing is a model convention, not measured within-frame physical timing.

.. doctest::

   >>> import numpy as np
   >>> from tnet_analysis.latency_measures import SmartWalker
   >>> snapshot = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
   >>> network = np.repeat(snapshot[None, :, :], 3, axis=0)
   >>> SmartWalker(network)[0, 2]
   2.0
   >>> SmartWalker(network, hops_per_frame=2)[0, 2]
   1.0
   >>> SmartWalker(network, hops_per_frame=2, latency_unit="substeps")[0, 2]
   2.0

Lifetime and search limits
--------------------------

``ComputeTemporalClustering(delta=1)`` uses consecutive central connections
with closure in the second frame. At larger delta, central connections can
be separated by up to delta frames; closure is accepted from the second
connection's frame through delta-1 frames later. Each eligible triplet is
counted once, even with repeated closing edges. Recording boundaries truncate
closure windows without wrapping.

``ComputeRetention(delta=None)`` considers all past frames. A positive delta
restricts history to max(0,t-delta) through t-1. Current contacts cannot be
their own history. The first nonempty observation is excluded. Later
nonempty frames can score zero when all earlier contacts expire; empty
neighborhoods are skipped.

Return latency's ``max_latency`` limits the **frames** searched from each
start, even with multiple hops. It defaults to the recording horizon.
Shortening it changes the estimate, not just runtime.
