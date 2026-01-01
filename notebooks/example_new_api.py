"""
Example demonstrating the new improved API for visualize_pixel_map.

This example shows how to:
1. Load data from a neutron data folder (auto-detecting AssociatedResults)
2. Use filtering options for photons, events, and pixels
3. Use TOA vs TOT colormapping
4. Configure association settings
"""

import visualize_pixel_map as vpm

# ============================================================================
# Example 1: Simple usage - auto-detect and load AssociatedResults
# ============================================================================
print("Example 1: Loading data from folder with auto-detection")
print("=" * 60)

# Simply point to the neutron data folder
# This will automatically detect and load AssociatedResults/*.csv
data = vpm.Data("data/neutrons")

# Plot with default settings
data.plot(key=0, cmap='viridis', time_bins=60)

print("\n")

# ============================================================================
# Example 2: Auto-run association if no AssociatedResults exists
# ============================================================================
print("Example 2: Auto-running association")
print("=" * 60)

# If AssociatedResults folder doesn't exist, neutron_event_analyzer will run automatically
# You can specify which association mode to use:
# - 'pixel_photon': Associate pixels to photons
# - 'photon_event': Associate photons to events
# - 'full': Both pixel-photon and photon-event associations
data = vpm.Data(
    "data/neutrons",
    auto_associate=True,
    association_mode='full',  # Run full association
    verbosity=1
)

print("\n")

# ============================================================================
# Example 3: Using a settings.json file for association parameters
# ============================================================================
print("Example 3: Using settings.json")
print("=" * 60)

# You can provide a settings.json file or dictionary
# The settings.json will be automatically loaded if it exists in the data folder
data = vpm.Data(
    "data/neutrons",
    settings="path/to/settings.json",  # Or use a dict
    verbosity=1
)

# Or use a settings dictionary
settings = {
    "pixel2photon": {
        "dSpace": 2,
        "dTime": 100e-09,
        "nPxMin": 8,
        "nPxMax": 100
    },
    "photon2event": {
        "dSpace_px": 50.0,
        "dTime_s": 5e-08,
        "durationMax_s": 5e-07
    }
}

data = vpm.Data(
    "data/neutrons",
    settings=settings,
    verbosity=1
)

print("\n")

# ============================================================================
# Example 4: Filtering by photon range or IDs
# ============================================================================
print("Example 4: Filtering by photons")
print("=" * 60)

# Filter by photon index range (first 10 photons)
data.plot(
    photon_range=(0, 10),
    cmap='viridis',
    time_bins=60
)

# Filter by specific photon IDs
data.plot(
    photon_ids=[5, 10, 15, 20],
    cmap='magma',
    time_bins=60
)

print("\n")

# ============================================================================
# Example 5: Filtering by events
# ============================================================================
print("Example 5: Filtering by events")
print("=" * 60)

# Filter by event index range
data.plot(
    event_range=(0, 5),
    cmap='plasma',
    time_bins=60
)

# Filter by specific event IDs
data.plot(
    event_ids=[1, 2, 3],
    cmap='inferno',
    time_bins=60
)

print("\n")

# ============================================================================
# Example 6: Filtering by pixel range
# ============================================================================
print("Example 6: Filtering by pixel hits")
print("=" * 60)

# Filter by pixel index range (first 1000 pixel hits)
data.plot(
    pixel_range=(0, 1000),
    cmap='viridis',
    time_bins=60
)

print("\n")

# ============================================================================
# Example 7: Filtering by time of arrival (TOA) range
# ============================================================================
print("Example 7: Filtering by TOA range")
print("=" * 60)

# Filter by time range (0 to 0.001 seconds)
data.plot(
    toa_range=(0, 0.001),
    cmap='coolwarm',
    time_bins=60
)

print("\n")

# ============================================================================
# Example 8: Using TOT (Time Over Threshold) for colormapping
# ============================================================================
print("Example 8: TOA vs TOT colormapping")
print("=" * 60)

# Default: color by TOA (time of arrival)
data.plot(
    key=0,
    color_by='toa',  # Color based on time bins
    cmap='viridis',
    time_bins=60
)

# Color by TOT (time over threshold) - shows energy deposition
data.plot(
    key=0,
    color_by='tot',  # Color based on TOT values
    cmap='plasma',
    time_bins=60,
    show_labels=True  # Labels will show TOT values instead of time
)

print("\n")

# ============================================================================
# Example 9: Combined filters
# ============================================================================
print("Example 9: Combining multiple filters")
print("=" * 60)

# You can combine multiple filters
data.plot(
    photon_range=(0, 20),      # First 20 photons
    toa_range=(0, 0.005),      # Within first 5ms
    color_by='tot',            # Color by TOT
    cmap='viridis',
    time_bins=80,
    show_labels=True,
    show_legend=True
)

print("\n")

# ============================================================================
# Example 10: Backward compatibility - loading CSV files directly
# ============================================================================
print("Example 10: Backward compatibility")
print("=" * 60)

# The old API still works - you can pass a CSV file directly
data = vpm.Data("data/pixel_data.csv")
data.plot(key=800, cmap='viridis')

print("\n")
print("All examples completed successfully!")
