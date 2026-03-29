import os

# Set before numpy/scipy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from joblib import Parallel, delayed

from funcIntSeg import *

N_JOBS = int(os.environ.get("INTSEG_N_JOBS", "44"))


if __name__ == "__main__":
    __spec__ = None

    rng = np.random.default_rng(12345)

    for isubj in range(nSubj):
        tasks = [(ideg, itag) for ideg in range(nDeg) for itag in range(nTag)]
        rng.shuffle(tasks)

        results = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            verbose=10,
            batch_size=1,
            pre_dispatch="2*n_jobs",
        )(
            delayed(eval_subject_deg_tag)(isubj, ideg, itag)
            for ideg, itag in tasks
        )

        # Flat metric layout:
        # yy[iprob, ideg, itag, :] stores the packed nOut-vector from funcIntSeg.py.
        yy = np.full((nProb, nDeg, nTag, nOut), np.nan, dtype=float)

        for ideg, itag, block in results:
            yy[:, ideg, itag, :] = block

        np.save(f"{outdir}/xIntSeg_isubj_{isubj:02}.npy", yy)
        print(f"saved subject {isubj:02} -> {yy.shape}")
