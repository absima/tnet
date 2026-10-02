import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))
for module_name in tuple(sys.modules):
    if module_name == "funcIntSeg" or module_name == "tnet_analysis" or module_name.startswith("tnet_analysis."):
        del sys.modules[module_name]

import funcIntSeg

from funcIntSeg import (
    BinarizeDFC, EvaluateMetric, GenerateNullModel,
    HOP_DEPENDENT_METRICS, METRIC_SPECS, hopList, lag,
    latency_unit, rand_seed, window,
)

print(f"Using {funcIntSeg.__file__}")

indir = "/Users/sima/Documents/MATLAB/empirical"
tsdata = loadmat(f"{indir}/ts_100subjs.mat")["tseries"]
degDist = np.logspace(np.log10(0.1), np.log10(83.04), 54)

isubj = 5
ideg = 10
avdeg = degDist[ideg]
tag = "original"
seed = rand_seed + 100000 * isubj + 1000 * ideg

tnet = BinarizeDFC(tsdata[isubj], avdeg, window, lag, tag)
tnet, _ = GenerateNullModel(tnet, tag, seed=seed)
tnet_nodiag = tnet.copy()
tnet_nodiag[:, np.arange(tnet.shape[1]), np.arange(tnet.shape[1])] = 0

for ifunc, (_, output_names, _, _, _) in enumerate(METRIC_SPECS):
    tested_hops = hopList if ifunc in HOP_DEPENDENT_METRICS else [1]
    for hops_per_frame in tested_hops:
        _, name, values = EvaluateMetric(
            ifunc, tnet, tnet_nodiag, random_seed=seed,
            hops_per_frame=hops_per_frame, latency_unit=latency_unit,
        )
        suffix = f" (hops/frame={hops_per_frame})" if ifunc in HOP_DEPENDENT_METRICS else ""
        print(f"\nfunc{ifunc}: {name}{suffix}")
        for label, value in zip(output_names, values):
            print(f"  {label}: {value}")
