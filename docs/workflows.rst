Signals, FC and null models
===========================

Methods versus tags
-------------------

``fc_method="original1"`` selects sliding-window Pearson FC;
``"original2"`` selects instantaneous z-score products.
These are methods, not null-model tags. ``tag="original"`` means no
randomization for either method. Select method and tag independently.

Instantaneous FC uses ``FC[t,i,j] = z[t,i]*z[t,j]``, standardizing each
region over the full recording. Products are signed and not bounded by
[-1,1]. With population standardization (ddof=0), their time average equals
whole-record Pearson correlation. Constant regions are rejected.

Complete signal workflow
------------------------

.. literalinclude:: examples/signal_workflow.py
   :language: python

Download :download:`signal_workflow.py <examples/signal_workflow.py>`.
``PrepareConnectivity`` transforms signals before FC.
``PrepareTemporalNetwork`` thresholds that FC and transforms the network.
Passing unrandomized FC with a phase tag into the second function does not
retroactively randomize signals.

Thresholding
------------

Temporal binarization finds one signed threshold for the FC tensor and
applies ``FC > threshold`` to every frame. It targets average degree
over the recording, not identical degree in every frame.
``FindThresholdsForAverageDegrees`` uses a finite candidate grid (resolution
100 by default); requested degree is approximate. Record the achieved
degree using ``ComputeAverageDegree``.

``emp_static`` averages FC over time, selects strongest **absolute-valued**
off-diagonal entries and repeats that static network. Ties can exceed the
requested edge count. This differs from temporal signed thresholding.

For several degrees, compute FC once, find all thresholds together and pass
the corresponding threshold into network preparation. Recompute thresholds
when signals or FC method change; phase surrogates need their own FC.

All supported analysis tags
---------------------------

There are 17 analysis tags. They act at different stages: phase surrogates
modify signals before FC, ``emp_static`` changes binarization, and network
nulls transform the thresholded adjacency. None selects an FC method.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Tag
     - Operation
   * - ``original``
     - Keep the signal and temporal network.
   * - ``emp_static``
     - Repeat a static network formed from mean FC.
   * - ``er_static``
     - Repeat one random snapshot with the mean reference edge count.
   * - ``sw_static``
     - Repeat one small-world-style snapshot with the mean reference edge count.
   * - ``sf_static``
     - Repeat one rank-weighted snapshot with the mean reference edge count.
   * - ``time``
     - Shuffle adjacency frames, not raw signals.
   * - ``uniform``
     - Generate random frames with approximately constant edge count matched to the mean input density.
   * - ``eAct``
     - Randomize each edge's active frames, preserving its activation total.
   * - ``eActTemplate``
     - Use an empirically ranked edge template with framewise rewiring.
   * - ``eRndTemplate``
     - Use a randomly ranked edge template with framewise rewiring.
   * - ``erand_er``
     - Randomize edge locations while retaining each frame's edge count.
   * - ``erand_sws``
     - Generate small-world-style frames with a shared node ordering.
   * - ``erand_sfs``
     - Generate rank-weighted frames with a shared node ordering.
   * - ``erand_swr``
     - Generate small-world-style frames, then independently relabel each frame.
   * - ``erand_sfr``
     - Generate rank-weighted frames, then independently relabel each frame.
   * - ``phase_rand0``
     - Add independent Fourier phases per region.
   * - ``phase_rand1``
     - Add shared Fourier phases across regions.

Reference network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``original`` retains the selected signals and temporal thresholding. It is
valid with either ``fc_method="original1"`` or ``"original2"``. In network
preparation, isolated-node trimming and padding can still change node order;
see the warning below.

Static controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``emp_static`` preserves a static representation of mean FC, not the original
sequence of framewise edges. All frames are identical. Its absolute-valued
edge selection can include strong negative FC; temporal thresholding instead
uses signed values. Calling ``GenerateNullModel(tnet, "emp_static")`` alone
does not average FC or construct this static graph: that step belongs to
``PrepareTemporalNetwork`` / ``BinarizeDFCFromConnectivity``.

``er_static``, ``sw_static`` and ``sf_static`` generate one snapshot using
the rounded mean reference edge count, then repeat it for all T frames.
They remove temporal topology changes and reference edge identities. They
do not preserve empirical node degrees, clustering or community structure.

* ``er_static`` samples the requested number of unordered pairs uniformly
  without replacement: a fixed-edge-count random graph, not G(N,p).
* ``sw_static`` first selects pairs in increasing node-index separation,
  then rewires a fraction (0.02 by default). This is a small-world-style
  local backbone, not an exact Watts-Strogatz ring-lattice implementation.
* ``sf_static`` samples pairs without replacement with probabilities
  proportional to the product of node-rank weights, where rank weight is
  proportional to rank**(-gamma), gamma=2.5 by default. This encourages
  hubs but does not guarantee a fitted power-law degree distribution.

To change generator-specific parameters, use the corresponding lower-level
generator. The preparation wrapper does not expose their rewiring fraction
or power-law exponent.

Time-order and uniform-density controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``time`` permutes whole adjacency frames. It preserves each frame's topology,
the distribution of framewise edge counts and each edge's total activations.
It changes temporal order and usually changes consecutive-frame similarity,
time-respecting paths and recurrence intervals. It does not shuffle raw
time-series samples or randomize edges within a frame.

``uniform`` generates fresh random edge sets with nearly equal counts across
frames, matching the mean reference edge count. Integer rounding can produce
one-edge differences. Neither the original count sequence nor per-edge
activation totals are preserved. This is not a degree-sequence-preserving
rewiring or a fixed-probability Bernoulli graph model.

Edge-activation control
~~~~~~~~~~~~~~~~~~~~~~~

``eAct`` independently chooses new active-frame positions for each edge,
without replacement, retaining exactly its number of active frames.
Thus edge identities, total density and the time-aggregated weighted graph
are preserved, but per-frame edge counts, simultaneous activations,
run lengths and temporal ordering are not. Always-active and never-active
edges remain so. It is not the same as ``time``, which permutes whole frames.

Ranked-template controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both template tags retain the original edge count in each frame. A frame
with k edges starts with the first k edges of a common ranked template;
floor(p_rewire*k) of those edges are selected for replacement. They model
shared topology with controlled variation rather than unconstrained random
frames. Selecting an edge for replacement does not guarantee that it changes:
removed edges can be selected again.

``eActTemplate`` ranks edges by their reference activation totals, highest
first. This retains an empirical preference ordering, not exact per-edge
activation totals. By default never-active edges are excluded from weighted
replacement candidates. ``eRndTemplate`` instead assigns one random ranking
to all possible edges; reference activation preferences are discarded, while
the shared template still induces temporal persistence.

The lower-level interface supports:

* ``p_rewire=0.02``: fraction selected for replacement per frame; 0 repeats
  template prefixes (identical graphs when edge counts are identical).
* ``alpha_rank=1.0``: replacement preference decays as
  1/(rank+1)**alpha_rank; 0 gives uniform rank weights.
* ``exclude_never_active=True``: suppresses never-active replacement
  candidates for ``eActTemplate`` only. A sparse candidate pool can trigger
  an unweighted fallback, so this is not an unconditional exclusion rule.
* ``eta_readd=1.0``: positive multiplier controlling the preference for
  re-adding edges selected for removal.

``GenerateNullModel`` forwards these options to the template generator;
``PrepareTemporalNetwork`` uses the defaults. For customized templates:

.. code-block:: python

   from tnet_analysis.network_generators import GenerateNullModel

   template_network, n_core = GenerateNullModel(
       reference_network, "eActTemplate", seed=123,
       p_rewire=0.1, alpha_rank=1.0,
       exclude_never_active=True, eta_readd=0.5,
   )

Use a binary reference network with diagonal ones. This call operates after
FC construction and thresholding; it does not accept raw signals.

Framewise synthetic controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All five ``erand_*`` tags regenerate frames with the corresponding reference
edge counts, retaining the count sequence but not empirical edge identities.

* ``erand_er`` samples a uniform fixed-size edge set in each frame.
* ``erand_sws`` generates small-world-style local edges using the same node
  ordering across frames. The ``s`` suffix indicates shared labels.
* ``erand_sfs`` generates rank-weighted edges using the same node ranks
  across frames. Shared ranks induce persistent hub preferences.
* ``erand_swr`` uses the small-world-style construction and independently
  permutes node labels in each frame afterwards.
* ``erand_sfr`` does the same for the rank-weighted construction. The ``r``
  suffix indicates randomized labels, not a different edge-count model.

Relabeling retains each generated frame's unlabeled topology but removes
fixed node identities across frames. It does not guarantee zero persistence
or equal node degrees in a particular realization. The small-world generator
currently resets NumPy's RNG when a seed is passed for each frame; equal
edge-count frames can therefore repeat the same graph before relabeling.
Do not interpret ``erand_sws`` as independent draws at fixed seed. These
tags use the same lower-level small-world and rank-weighted constructions
described for the static controls.

Signal-phase surrogates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both phase modes preserve means, Fourier magnitudes and circular
autocovariance per region. Shared phases also preserve cross-spectra and
sample covariance; independent phases generally do not. Neither guarantees
unchanged sliding-window FC or every time-domain feature.

``phase_rand0`` uses independent phase offsets for each region and frequency.
It disrupts inter-region phase relationships while retaining each signal's
power spectrum. ``phase_rand1`` uses the same frequencywise offsets for all
regions, retaining their relative phases and cross-spectra. Neither mode
preserves the original samples or their amplitude distribution in general.
Recompute FC and thresholds from the surrogate signals before analysis.

Choosing an interface and recording seeds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ResolveAnalysisTag`` recognizes all 17 tags and rejects unknown values.
``GenerateNullModel`` accepts the 15 network-level tags only: it does not
handle phase surrogates or FC-method selection. Its ``scale`` argument is
reserved and does not change generation. The preparation wrapper skips
network generation if thresholding yields no off-diagonal edges.

.. warning::

   Some generators trim isolated nodes, build a core and pad it into the
   leading node positions. Original node identities are not guaranteed.
   Use aggregate comparisons unless you verify an explicit node mapping.

Record separate seeds for surrogates, nulls, random walks and community
detection. Network preparation seeds NumPy's global RNG for generators
using it. Process workers isolate that state; concurrent threads do not.
