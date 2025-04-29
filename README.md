# tnet_analysis

**Temporal Network Analysis Toolkit**  
An expandable Python package for analyzing dynamic (temporal) networks with a focus on measuring segregation, integration, and dynamical patterns.

---

## 📦 Package Structure

- `tnet_analysis/`
  - `preprocessing_tools.py` — preprocess time series to build functional connectivity.
  - `network_generators.py` — generate synthetic or randomized temporal networks.
  - `dynamism_measures.py` — quantify dynamic behavior over time.
  - `segregation_measures.py` — capture local segregation properties.
  - `efficiency_measures.py` — global integration and diffusion efficiency measures.
  - `modularity_measures.py` — dynamic community detection and allegiance matrices.
  - `fidelity_and_stability.py` — assess stability of node communities over time.
  - `advanced_temporal_measures.py` — advanced experimental dynamic features (e.g., circulation, returnability).
  
  one can import a module for example and use any function within the module as: 
  `from tnet_analysis import preprocessing_tools as pt 
  pt.compute_dynamic_functional_connectivity(tseries, window=60, lag=5)`

---

## ⚙️ Installation

Clone the repository or download it manually, then:

```bash
cd path/to/tnet_analysis/
pip install -e .
```
---

## 🧠 Project Motivation

Many real-world systems — brains, social networks, communication systems — evolve over time.  
Understanding **when**, **how**, and **where** segregation or integration patterns emerge is crucial.

**`tnet_analysis`** offers a clean, scalable, and expandable framework for:

- Capturing dynamic organization,
- Comparing empirical vs randomized temporal networks,
- Quantifying local and global features over time.

Built for scientific clarity and robustness.

---

## ✨ Key Features
- Persistence-based Segregation: Node-level significant persistence detection.
- Partner Diversity and Stability: Capture changes in neighborhoods across time.
- Temporal Modularity and Allegiance: Assess community dynamics.
- Advanced Temporal Properties: Circulation latency, returnability, conductance.
- Global Integration Metrics: Dynamic diffusion and random walk efficiency.
- Randomized Models and Null Networks: Time-shuffled and edge-randomized models.
- Memory-based Analysis: Node-wise neighborhood memory and clustering.
##🎯 Designed For
- Dynamic brain network studies (e.g., fMRI, EEG)
- Time-evolving social networks
- Communication or transportation systems
- Biological or ecological interaction networks
- Any domain requiring temporal graph analysis

## 🌌 Future Expansions
- Support for weighted networks
- Temporal motif analysis
- Node centrality evolution
- Improved null model generators
- Public distribution on GitHub or PyPI
	
	
## 🚀 Author

Built by Simachew Mengiste (with Demian Bataglia)
Driven by curiosity, scientific clarity, and modular design.
FunSy - LNCA - Unistra