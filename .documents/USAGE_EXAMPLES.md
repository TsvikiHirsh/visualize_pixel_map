# visualize_pixel_map - Usage Examples

## Quick Start

```python
import visualize_pixel_map as vpm

# Simple: just point to your data folder
data = vpm.Data("data/neutrons")

# Plot with default settings
data.plot()
```

## Simplified Filtering API

The new API combines range and ID parameters into single flexible parameters:

### Photon Filtering

```python
# Range using tuple
data.plot(photons=(0, 10))  # First 10 photons

# Range using slice
data.plot(photons=slice(0, 10))  # First 10 photons

# Specific IDs using list
data.plot(photons=[5, 10, 15, 20])  # Specific photon IDs

# Single ID using int
data.plot(photons=5)  # Just photon ID 5
```

### Event Filtering

```python
# Range
data.plot(events=(0, 5))  # First 5 events

# Specific IDs
data.plot(events=[1, 2, 3])  # Specific event IDs

# Single event
data.plot(events=1)  # Just event ID 1
```

### Pixel Filtering

```python
# Range
data.plot(pixels=(0, 1000))  # First 1000 pixel hits

# Specific indices
data.plot(pixels=[0, 100, 200, 300])  # Specific pixels

# Slice
data.plot(pixels=slice(0, 1000))  # First 1000 pixels
```

## Colormap Options

### TOA (Time of Arrival) Coloring

```python
# Default: color by time of arrival
data.plot(
    color_by='toa',  # Default
    cmap='viridis',
    time_bins=60
)
```

### TOT (Time Over Threshold) Coloring

```python
# Color by energy deposition (TOT)
data.plot(
    color_by='tot',
    cmap='plasma',
    time_bins=60,
    show_labels=True  # Labels show TOT values
)
```

## Combined Filters

```python
# Combine multiple filters
data.plot(
    photons=(0, 20),           # First 20 photons
    toa_range=(0, 0.005),      # Within first 5ms
    color_by='tot',            # Color by TOT
    cmap='viridis',
    time_bins=80,
    show_labels=True
)
```

## Auto-Association

```python
# Automatically run neutron_event_analyzer if needed
data = vpm.Data(
    "data/neutrons",
    auto_associate=True,        # Default: True
    association_mode='full',    # 'pixel_photon', 'photon_event', or 'full'
    settings="settings.json"    # Optional: custom settings
)
```

## Settings Configuration

```python
# Auto-loaded from data folder if present
data = vpm.Data("data/neutrons")  # Loads data/neutrons/settings.json

# Custom settings file
data = vpm.Data("data/neutrons", settings="custom.json")

# Settings dictionary
settings = {
    "pixel2photon": {
        "dSpace": 2,
        "dTime": 100e-09
    },
    "photon2event": {
        "dSpace_px": 50.0,
        "dTime_s": 5e-08
    }
}
data = vpm.Data("data/neutrons", settings=settings)
```

## Complete Example

```python
import visualize_pixel_map as vpm

# Load data with auto-association
data = vpm.Data(
    "data/neutrons",
    association_mode='full',
    verbosity=1
)

# Plot first 20 photons with TOT coloring
fig, ax = data.plot(
    photons=(0, 20),        # First 20 photons
    toa_range=(0, 0.005),   # 0-5ms
    color_by='tot',         # Color by TOT
    cmap='plasma',
    time_bins=80,
    zoom_size=30,           # Zoom to 30x30 pixel region
    show_labels=True,
    despine=True
)

# Save the figure
fig.savefig('output.png', dpi=300, bbox_inches='tight')
```

## Backward Compatibility

Old CSV-based API still works:

```python
# Load from CSV file (old API)
data = vpm.Data("data/pixel_data.csv")
data.plot(key=800, cmap='viridis')
```

## All Available Parameters

```python
data.plot(
    key=None,                  # Starting time bin index
    neutron_id_filter=None,    # Filter by neutron ID

    # Simplified filtering
    photons=None,              # tuple/list/int/slice
    pixels=None,               # tuple/list/int/slice
    events=None,               # tuple/list/int/slice
    toa_range=None,            # (min, max) in seconds

    # Coloring
    color_by='toa',            # 'toa' or 'tot'
    cmap=None,                 # Matplotlib colormap
    custom_color=None,         # Deprecated

    # Zoom and display
    zoom_region=None,          # (x_min, x_max, y_min, y_max)
    zoom_size=None,            # Square zoom size
    auto_zoom_margin=5,        # Margin for auto zoom

    # Visual options
    show_background=False,     # Show background histogram
    despine=True,              # Remove plot spines
    show_labels=True,          # Show labels on pixels
    show_legend=False,         # Show legend
    show_scale=False,          # Show pixel scale

    # Time bins
    time_bins=None,            # Max time in ns (e.g., 40, 60)
)
```
