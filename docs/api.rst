API reference
=============

The signatures and descriptions below come from package docstrings.
The guide examples deliberately import library modules, not runner scripts
that load external data when executed.

Connectivity
------------

.. automodule:: tnet_analysis.dfc_to_tnet
   :members: ComputeDynamicFunctionalConnectivity, ComputeInstantaneousFunctionalConnectivity, ComputeFunctionalConnectivity, FindThresholdsForAverageDegrees, FindThresholdForAverageDegree, BinarizeDFC, BinarizeDFCFromConnectivity

Preparation
-----------

.. automodule:: tnet_analysis.analysis_preparation
   :members: ResolveAnalysisTag, PrepareSignal, PrepareConnectivity, PrepareTemporalNetwork

Surrogates
----------

.. automodule:: tnet_analysis.signal_surrogates
   :members: PhaseRandSurrogates

Distance and return
-------------------

.. automodule:: tnet_analysis.latency_measures
   :members: SetDiagonals, ComputeAverageDegree, SmartWalker, RandomWalker, MeanLatencyMatrixAnalysis, ComputeSmartTemporalDistanceMeasures, ComputeDrunkTemporalDistanceMeasures, ComputeReturnLatency

Dynamism and memory
-------------------

.. automodule:: tnet_analysis.dynamism_and_memory
   :members: ComputeTemporalMutualInformation, ComputeDynamism, ComputeTemporalEdgeOverlap, ComputeEdgePersistenceRate, ComputeNeighborhoodMemory, ComputeRetention, ComputeLinkBurstiness

Structure
---------

.. automodule:: tnet_analysis.segregation_and_cohesion
   :members: ComputeSpatialClustering, ComputeTemporalClustering, ComputeSnapshotTransitivity, ComputeTemporalTransitivity, ComputeSnapshotParticipationCoefficient, ComputeTemporalParticipationCoefficient, ComputePartnerStability, ComputePartnerDiversity, ComputeNodePersistence

Modularity
----------

.. automodule:: tnet_analysis.modularity_estimates
   :members: EnsureCsr, CoarseGrain, DetectCommunitiesLouvain, SnapshotModularity, ComputeSnapshotModularity, ComputeTemporalModularity

Network generators
------------------

.. automodule:: tnet_analysis.network_generators
   :members: GenerateSymmetricRandomNetwork, GenerateSymmetricSmallWorldNetwork, GenerateSymmetricScaleFreeNetwork, GenerateSymmetricNetwork, GenerateStaticTemporalNetwork, GenerateTemporalLinkCounts, GenerateTemporalNetworkWithDensityVariation, GenerateTemporalNetworkByLinkActivation, GenerateTemporalNetworkFromEdgeTemplate, TrimIsolatedNodes, GenerateNullModel, generateRandomTemporalNetwork, PartiallyRandomizeMatrix, GenerateRandomizedTemporalNetwork
