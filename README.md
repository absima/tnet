# Temporal Network Analysis Toolkit

This toolkit provides functions for building, transforming, and analyzing **temporal networks**.  
It is organized into five functional groups that form a **pipeline**:

---

## 📂 Group 1: DFC → TNet
**Convert dynamic functional connectivity (DFC) into temporal networks.**

- `ComputeDynamicFunctionalConnectivity(time_series, window, lag)`  
  → Sliding-window Pearson correlation matrices.  
- `FindThresholdForAverageDegree(adj_tensor, target_degree, resolution=100)`  
  → Find threshold to match a target average degree.  
- `BinarizeDFC(time_series, target_degree, window, lag)`  
  → Pipeline: DFC → thresholding → binary adjacency.  
- `SetDiagonals(tensor, value=1)`  
  → Utility to enforce diagonal values.  

---

## 📂 Group 2: Temporal Network Generators
**Create synthetic and null temporal networks.**

- `GenerateSymmetricRandomNetwork`  
- `GenerateSymmetricSmallWorldNetwork`  
- `GenerateSymmetricScaleFreeNetwork`  
- `GenerateSymmetricNetwork` (dispatch by type)  
- `GenerateStaticTemporalNetwork`  
- `GenerateTemporalNetworkByLinkActivation`  
- `GenerateTemporalNetworkWithDensityVariability`  
- `GenerateTemporalLinkCounts`  
- `GenerateNullModel`  
- `PartiallyRandomizeMatrix`  
- `GenerateRandomizedTemporalNetwork`  
- `TrimIsolatedNodes`  

---

## 📂 Group 3: Distance & Circulation Metrics
**Temporal reachability, shortest paths, and circulation.**

- `ComputeAverageDegree`  
- `SmartWalker` → Earliest arrival latencies.  
- `RandomWalker` → Monte Carlo first-passage latencies.  
- `MeanLatencyMatrixAnalysis` → Summarize latency matrix.  
- `ComputeTemporalDistanceMeasures` → Bundle (degree + walkers).  
- `ComputeCirculationLatency`  
- `ComputeCirculationRate`  
- `ComputeCirculationLatencyAndRate` (efficient one-pass).  

---

## 📂 Group 4: Temporal Dynamism & Memory
**Quantify network volatility and temporal memory.**

- `ComputeTemporalMutualInformation`  
- `ComputeAdjustedEntropy`  
- `ComputeTemporalEdgeOverlap`  
- `ComputeEdgePersistenceRate`  
- `ComputeNeighborhoodMemory`  
- `ComputeReturnability`  
- `ComputeLinkBurstiness`  

---

## 📂 Group 5: Segregation & Cohesion Structure
**Measure clustering, modularity, and persistence of community structure.**

- `ComputeStaticClustering`  
- `ComputeTemporalClustering`  
- `ComputeSnapshotTransitivity`, `ComputeTemporalTransitivity`  
- `ComputeSnapshotParticipationCoefficient`, `ComputeTemporalParticipationCoefficient`  
- `ComputePartnerStability`  
- `ComputePartnerDiversity`  
- `ComputeNodePersistence`  


---

## 📂 Group 5.1: Modularity (to be revised)
- `ComputeSnapshotModularity`, `ComputeTemporalModularity`  

## 📌 Pipeline Overview

1. **Time series → TNet** (Group 1)  
2. **Null models & synthetic networks** (Group 2)  
3. **Distance & circulation metrics** (Group 3)  
4. **Dynamism & memory analysis** (Group 4)  
5. **Segregation & cohesion analysis** (Group 5)  

Together these provide a unified framework for studying both **integration** and **segregation** in temporal networks.

	
	
## 🚀 Author

Built by Simachew Mengiste (with Demian Bataglia)

Driven by curiosity, scientific clarity, and modular design.
FunSy - LNCA - Unistra