import numpy as np


def PhaseRandSurrogates(TS, value=1, random_seed=None):
    """
    Generate phase-randomized signals following dFCwalk's PhaseRand_surrogates.m.

    Args:
        TS: np.ndarray, shape (time, regions). Finite, real-valued input signals.
        value: {0, 1}, optional (default=1). Independent phases per region (0),
            or shared phases across regions (1).
        random_seed: int or None, optional (default=None). Seed for a local RNG.

    Returns:
        TS_rnd: np.ndarray, shape (time, regions). Real-valued surrogate signals.

    Notes:
        - Both modes preserve means and each region's Fourier magnitudes, hence
          its circular autocovariance. DC and even-length Nyquist terms are unchanged.
        - Mode 1 also preserves cross-spectra and the sample covariance matrix.
        - Mode 0 does not preserve interregional covariance in general.
        - Seeded results are reproducible in Python, not seed-identical to MATLAB.
    """
    signals = np.asarray(TS)
    if signals.ndim != 2 or min(signals.shape) < 1:
        raise ValueError("TS must have nonempty shape (time, regions).")
    if np.iscomplexobj(signals) or not np.isfinite(signals).all():
        raise ValueError("TS must contain finite, real-valued signals.")
    if value not in (0, 1):
        raise ValueError("value must be 0 (independent) or 1 (shared phases).")
    n_time, n_regions = signals.shape
    spectrum = np.fft.rfft(signals, axis=0)
    n_random = (n_time - 1) // 2
    rng = np.random.default_rng(random_seed)
    columns = n_regions if value == 0 else 1
    phases = rng.uniform(-np.pi, np.pi, size=(columns, n_random)).T
    spectrum[1:1 + n_random] *= np.exp(1j * phases)
    return np.fft.irfft(spectrum, n=n_time, axis=0)
