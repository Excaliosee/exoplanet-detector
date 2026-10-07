import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic


def process_stellar_data(
    target_name: str,
    expected_period: float,
    bins: int = 200
):
    """
    Downloads and processes Kepler light curve data
    into a CNN-ready fixed-size vector.
    """

    print(f"\nProcessing target: {target_name}")

    # -------------------------------------------------
    # Download ALL available Kepler quarters
    # -------------------------------------------------
    search_result = lk.search_lightcurve(
        target_name,
        mission='Kepler'
    )

    lc_collection = search_result.download_all()

    # Stitch all quarters together
    lc = lc_collection.stitch()

    # -------------------------------------------------
    # Clean data
    # -------------------------------------------------
    lc = lc.remove_nans()

    lc = lc.remove_outliers(
        sigma_upper=5,
        sigma_lower=5
    )

    # Normalize flux
    lc = lc.normalize()

    # Remove stellar trends
    flattened_lc = lc.flatten(
        window_length=2001
    )

    # -------------------------------------------------
    # Fold on orbital period
    # -------------------------------------------------
    folded_lc = flattened_lc.fold(
        period=expected_period
    )

    # Sort by phase
    folded_lc = folded_lc.sort('phase')

    phases = folded_lc.phase.value
    fluxes = folded_lc.flux.value

    # -------------------------------------------------
    # Center transit at phase = 0
    # -------------------------------------------------
    min_idx = np.argmin(fluxes)

    transit_center = phases[min_idx]

    phases = phases - transit_center

    # Re-sort after shifting
    sort_idx = np.argsort(phases)

    phases = phases[sort_idx]
    fluxes = fluxes[sort_idx]

    # -------------------------------------------------
    # Bin into fixed-size ML vector
    # -------------------------------------------------
    binned_flux, _, _ = binned_statistic(
        phases,
        fluxes,
        statistic='median',
        bins=bins
    )

    # Replace NaN bins
    nan_mask = np.isnan(binned_flux)

    if np.any(nan_mask):
        binned_flux[nan_mask] = np.nanmedian(
            binned_flux
        )

    # Normalize around zero
    binned_flux = binned_flux - 1.0

    return binned_flux


# =====================================================
# Example Usage
# =====================================================

if __name__ == "__main__":

    ml_ready_vector = process_stellar_data(
        target_name="Kepler-10",
        expected_period=0.837495,
        bins=200
    )

    print("\n--- PIPELINE SUCCESSFUL ---")

    print(f"Vector Shape: {ml_ready_vector.shape}")

    print("\nFirst 10 Values:")
    print(ml_ready_vector[:10])

    # -------------------------------------------------
    # Plot processed signal
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))

    plt.plot(
        ml_ready_vector,
        'r.-',
        label='Processed Transit Signal'
    )

    plt.title("ML-Ready Exoplanet Transit")
    plt.xlabel("Bin Index")
    plt.ylabel("Normalized Flux")

    plt.grid(True)
    plt.legend()

    plt.show()