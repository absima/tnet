# tnet-analysis 
"""
Temporal Network Analysis Toolkit
=================================

This package provides functions for constructing and analyzing temporal networks.
Functions are organized into five logical groups:

1. DFC → TNet
   - Convert time series into temporal networks via sliding-window correlations
     and thresholding to match a target degree.
   - Key functions: ComputeDynamicFunctionalConnectivity, FindThresholdForAverageDegree,
     BinarizeDFC, SetDiagonals.

2. Temporal Network Generators
   - Build synthetic and null temporal networks of different types
     (Erdős–Rényi, small-world, scale-free, randomized, density-controlled).
   - Key functions: GenerateSymmetricRandomNetwork, GenerateSymmetricSmallWorldNetwork,
     GenerateSymmetricScaleFreeNetwork, GenerateSymmetricNetwork, GenerateStaticTemporalNetwork,
     GenerateNullModel, GenerateTemporalNetworkByLinkActivation, GenerateTemporalNetworkWithDensityVariability,
     PartiallyRandomizeMatrix, GenerateRandomizedTemporalNetwork, TrimIsolatedNodes.

3. Distance & Circulation Metrics
   - Estimate earliest arrival latencies, random-walk distances, and circulation times.
   - Key functions: ComputeAverageDegree, SmartWalker, RandomWalker, MeanLatencyMatrixAnalysis,
     ComputeTemporalDistanceMeasures, ComputeCirculationLatency, ComputeCirculationRate,
     ComputeCirculationLatencyAndRate.

4. Temporal Dynamism & Memory
   - Quantify network volatility, persistence of ties, and burstiness.
   - Key functions: ComputeTemporalMutualInformation, ComputeAdjustedEntropy,
     ComputeTemporalEdgeOverlap, ComputeEdgePersistenceRate, ComputeNeighborhoodMemory,
     ComputeReturnability, ComputeLinkBurstiness.

5. Segregation & Cohesion Structure
   - Characterize clustering, modularity, participation, partner stability/diversity,
     and persistence at the node/community level.
   - Key functions: ComputeStaticClustering, ComputeTemporalClustering,
     ComputeSnapshotTransitivity, ComputeTemporalTransitivity,
     ComputeSnapshotModularity, ComputeTemporalModularity,
     ComputeSnapshotParticipationCoefficient, ComputeTemporalParticipationCoefficient,
     ComputePartnerStability, ComputePartnerDiversity, ComputeNodePersistence,
     SummarizeSegregationStructure.

Pipeline
--------
The intended workflow is:

    Time series  →  Temporal Network (Group 1)
                  →  Null / Synthetic Networks (Group 2)
                  →  Distance & Circulation Metrics (Group 3)
                  →  Dynamism & Memory Metrics (Group 4)
                  →  Segregation & Cohesion Metrics (Group 5)

This provides a complete set of tools for studying both
integration (reachability, latency, circulation) and
segregation (clustering, modularity, persistence) in temporal networks.
"""

