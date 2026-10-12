Parallel computation and troubleshooting
========================================

Start small
-----------

Run one subject, a few degrees and a modest trial count first. Check shapes,
achieved degrees and NaN fractions, then time sparse and dense cases.
Increase worker counts gradually, observing memory and scratch use.
Hardware threads are not equivalent to physical cores, and more workers
do not guarantee a faster job.

A dense adjacency tensor needs T*N*N*itemsize bytes. A 296-frame,
400-node uint8 tensor needs about 45 MiB; float64 needs about 361 MiB.
Worker copies, FC tensors, temporary arrays and walk trials add to this.
Random-walk first-hit storage scales with trials*N*N even when full FPT
output is disabled. Avoid requesting full per-trial matrices unnecessarily.
Its dense neighbor table also uses int32 storage proportional to T*N*N.
The dynamism implementation forms a T-by-T similarity matrix, so long
recordings can add time-axis memory costs as well.

Prepare FC once per subject/method/window/lag and threshold sets once per
FC tensor. Reuse each network across metrics. Compute hop-independent
metrics once rather than repeating them for every hop allowance.
One-hop return uses packed reachability; the multi-hop return branch uses
dense matrix propagation, so larger hop allowances can be expensive.

Process-worker example
----------------------

.. literalinclude:: examples/parallel_subjects.py
   :language: python

Download :download:`parallel_subjects.py <examples/parallel_subjects.py>`.
The worker analyzes one small subject and returns named results.
The main guard supports process-based execution. Run it as a script,
not by starting workers during module import.

Limit nested numerical-library threads when using process workers.
Joblib's Loky configuration below limits supported inner thread pools.
Benchmark on the target machine; no fixed worker count is optimal everywhere.

Scratch and checkpointing
-------------------------

For large shared arrays, Joblib can use disk-backed memory maps.
Choose an explicitly configured local scratch directory with enough space,
pass it via ``temp_folder``, and monitor the filesystem containing that
directory, not just the output disk.

Use a scoped temporary directory for one subject/chunk so normal completion
cleans it. An abrupt kill can leave files: only remove directories known
to belong to stopped runs. Never clear a shared scratch root indiscriminately.

Save completed subjects individually, preferably by writing a temporary
file and atomically renaming it. Resume only when the saved shape and
configuration match. Neither metric functions nor this simple example
automatically implement checkpointing.

Common problems
---------------

* **Non-finite Pearson FC:** check missing samples and constant regions
  within each window. Decide a documented exclusion or imputation policy;
  do not blindly replace undefined correlations with zero.
* **No space left on device / pickling error:** check Joblib scratch,
  available space and inode counts. The pickling message can wrap a
  disk-write failure.
* **First call is slow:** Numba compilation is expected; compare warmed
  timings. Set ``NUMBA_CACHE_DIR`` to a writable folder if needed.
* **Means and medians differ:** sparse node-score distributions can have
  positive means and zero medians. Inspect scores and eligible counts.
* **Counts look negative:** give dynamism waiting diagonals; do not change
  node axes or include wrapper padding in calculations.
* **Memory pressure:** lower worker count, use compact binary dtypes,
  avoid full outputs and keep subject processing bounded.
