# New Features - visualize_pixel_map v2.0

## Overview

The visualize_pixel_map package has been significantly improved to integrate seamlessly with the [neutron_event_analyzer](https://github.com/TsvikiHirsh/neutron_event_analyzer) package. The new API simplifies data loading and adds powerful filtering and visualization options.

## Key Improvements

### 1. Simplified Data Loading API

**Before:**
```python
# Old API - required CSV file path
data = vpm.Data("data/pixel_data.csv")
```

**After:**
```python
# New API - just point to the data folder!
data = vpm.Data("data/neutrons")
```

The new API automatically:
- Detects and loads AssociatedResults CSV files
- Runs neutron_event_analyzer if needed
- Loads settings.json configuration
- Maintains backward compatibility with CSV file paths

### 2. Automatic Association with neutron_event_analyzer

If your data folder contains Exported folders but no AssociatedResults, the tool will automatically run the neutron_event_analyzer to create associations.

```python
data = vpm.Data(
    "data/neutrons",
    auto_associate=True,        # Run association automatically (default)
    association_mode='full',    # 'pixel_photon', 'photon_event', or 'full'
    settings="settings.json"    # Optional: custom settings
)
```

### 3. Advanced Filtering Options

#### Filter by Photons

```python
# Filter by photon index range (first 10 photons)
data.plot(photon_range=(0, 10))

# Filter by specific photon IDs
data.plot(photon_ids=[5, 10, 15, 20])
```

#### Filter by Events

```python
# Filter by event index range
data.plot(event_range=(0, 5))

# Filter by specific event IDs
data.plot(event_ids=[1, 2, 3])
```

#### Filter by Pixels

```python
# Filter by pixel hit index range
data.plot(pixel_range=(0, 1000))

# Filter by specific pixel indices
data.plot(pixel_ids=[0, 100, 200, 300])
```

#### Filter by Time of Arrival

```python
# Filter by TOA range (in seconds)
data.plot(toa_range=(0, 0.001))
```

### 4. TOA vs TOT Colormapping

You can now choose whether to color pixels based on Time of Arrival (TOA) or Time Over Threshold (TOT):

```python
# Color by TOA (default) - shows temporal evolution
data.plot(color_by='toa', cmap='viridis')

# Color by TOT - shows energy deposition
data.plot(color_by='tot', cmap='plasma', show_labels=True)
```

When using `color_by='tot'`, the labels will display TOT values instead of time bins.

### 5. Settings Configuration

#### Automatic settings.json Loading

If a `settings.json` file exists in your data folder, it will be loaded automatically:

```python
data = vpm.Data("data/neutrons")  # Automatically loads data/neutrons/settings.json
```

#### Custom Settings

You can provide custom settings as a file path or dictionary:

```python
# From file
data = vpm.Data("data/neutrons", settings="custom_settings.json")

# From dictionary
settings = {
    "pixel2photon": {
        "dSpace": 2,
        "dTime": 100e-09,
        "nPxMin": 8,
        "nPxMax": 100
    },
    "photon2event": {
        "dSpace_px": 50.0,
        "dTime_s": 5e-08
    }
}
data = vpm.Data("data/neutrons", settings=settings)
```

## Complete API Reference

### Data Class Constructor

```python
Data(
    data_path="data.empirphot.csv",  # Path to CSV file or folder
    start_time=0,                     # Start time in seconds
    end_time=0.01,                    # End time in seconds
    time_step=10,                     # Time step in nanoseconds
    sensor_size=8,                    # Sensor size in mm
    neutron_id=False,                 # Group by neutron_id
    verbosity=1,                      # Verbosity level (0 or 1)
    auto_associate=True,              # Auto-run association if needed
    association_mode='full',          # 'pixel_photon', 'photon_event', or 'full'
    settings=None                     # Settings file or dict
)
```

### Plot Method

```python
data.plot(
    key=None,                    # Starting time bin index
    neutron_id_filter=None,      # Filter by neutron ID

    # New filtering options
    photon_range=None,           # (start_idx, end_idx) for photons
    photon_ids=None,             # List of specific photon IDs
    pixel_range=None,            # (start_idx, end_idx) for pixels
    pixel_ids=None,              # List of specific pixel indices
    event_range=None,            # (start_idx, end_idx) for events
    event_ids=None,              # List of specific event IDs
    toa_range=None,              # (min_toa, max_toa) in seconds

    # New coloring option
    color_by='toa',              # 'toa' or 'tot'

    # Existing options
    zoom_region=None,            # (x_min, x_max, y_min, y_max)
    zoom_size=None,              # Square zoom size
    show_background=False,       # Show background histogram
    despine=True,                # Remove plot spines
    time_bins=None,              # Max time in ns (e.g., 40, 60)
    show_scale=False,            # Show pixel scale
    cmap=None,                   # Colormap name
    show_labels=True,            # Show labels on pixels
    show_legend=False,           # Show legend
    auto_zoom_margin=5           # Margin for auto zoom
)
```

## Data Folder Structure

The tool expects the following structure (as generated by neutron_event_analyzer):

```
data/neutrons/
├── AssociatedResults/       # Auto-detected and loaded
│   └── associated_data.csv
├── ExportedEvents/          # Used for association if no AssociatedResults
├── ExportedPhotons/
├── ExportedPixels/
├── settings.json            # Optional: auto-loaded if present
└── ... (other folders)
```

## Migration Guide

### Updating Existing Code

**Old code:**
```python
import visualize_pixel_map as vpm

data = vpm.Data("data/my_pixels.csv")
data.plot(key=800, cmap='viridis')
```

**New code (option 1 - use folder):**
```python
import visualize_pixel_map as vpm

data = vpm.Data("data/neutrons")  # Point to folder instead
data.plot(key=800, cmap='viridis')
```

**New code (option 2 - keep CSV, still works!):**
```python
import visualize_pixel_map as vpm

data = vpm.Data("data/my_pixels.csv")  # Still works!
data.plot(key=800, cmap='viridis')
```

### Taking Advantage of New Features

```python
import visualize_pixel_map as vpm

# Load from folder with auto-association
data = vpm.Data("data/neutrons", association_mode='full')

# Use advanced filtering and TOT coloring
data.plot(
    photon_range=(0, 20),
    toa_range=(0, 0.005),
    color_by='tot',
    cmap='plasma',
    time_bins=80,
    show_labels=True
)
```

## Examples

See [example_new_api.py](example_new_api.py) for comprehensive usage examples covering all new features.

## Requirements

- numpy >= 1.21.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- tqdm
- neutron_event_analyzer (optional, for auto-association)

## Backward Compatibility

All existing code using CSV file paths will continue to work without modifications. The new features are purely additive and opt-in.
