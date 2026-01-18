# visualize_pixel_map - Usage Examples

## Quick Start

```python
import visualize_pixel_map as vpm

# Simple: just point to your data folder
data = vpm.Data("data/neutrons")

# Check what data is available
data.info()

# Plot with default settings
data.plot()
```

## Inspecting Your Data

### Get Summary Information

```python
# Print comprehensive summary
data.info()
```

Output:
```
============================================================
Data Summary
============================================================
Total pixel hits: 98426
Data source: folder

Time range: 0.000000 - 0.010000 s
Time bins: 1000

Photons: 1234
  ID range: 0.0 - 1233.0

Events: 56
  ID range: 0.0 - 55.0

Available columns: x, y, toa, tot, tof, assoc_photon_id, ...
============================================================
```

### Check Available Ranges

```python
# Check photon information
print(data.photons)
# {'count': 1234, 'ids': [0, 1, 2, ...], 'range': (0.0, 1233.0)}

# Check event information
print(data.events)
# {'count': 56, 'ids': [0, 1, 2, ...], 'range': (0.0, 55.0)}

# Check pixel information
print(data.pixels)
# {'count': 98426, 'range': (0, 98425)}

# Check time information
print(data.times)
# {'range': (0.0, 0.01), 'bins': 1000, 'bin_keys': [0.0, 1e-8, ...]}

# Check neutron information (if available)
print(data.neutrons)
# {'count': 10, 'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
```

### Access the DataFrame Directly

```python
# Access the full associated dataframe
df = data.associated_df

# Now you can do pandas operations
print(df.head())
print(df.columns)
print(df.describe())

# Filter and analyze
high_tot = df[df['tot'] > 100]
print(f"High TOT pixels: {len(high_tot)}")
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

### Photon ID Coloring

```python
# Color by photon ID (discrete colors for each photon)
data.plot(
    photons=(0, 20),           # Filter to first 20 photons
    color_by='photons',        # or 'photon'
    cmap='viridis',
    show_labels=False,         # Hide labels to avoid clutter
    show_legend=True          # Show legend with photon IDs
)
```

### Event ID Coloring

```python
# Color by event ID (discrete colors for each event)
data.plot(
    events=(0, 10),            # Filter to first 10 events
    color_by='events',         # or 'event'
    cmap='tab20',              # Good for discrete categorical data
    show_labels=False,
    show_legend=True          # Legend shows "Event 1", "Event 2", etc.
)

# Show non-associated pixels as gray
data.plot(
    photons=(1, 20),           # Filter to photons 1-20
    color_by='events',
    cmap='viridis',
    show_assoc=False,          # Show ALL pixels, including non-associated
    show_legend=True          # Legend includes "Not associated" in gray
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

## Troubleshooting

### All Pixels Have Same Color When Using Event/Photon Coloring

If you see all pixels with the same color when using `color_by='events'` or `color_by='photons'`, this likely means:

1. **All filtered pixels belong to the same event/photon**:
   ```python
   # Check how many unique events are in your filtered data
   filtered_df = data._apply_filters(photons=(1, 20), verbosity=0)
   print(f"Unique events: {filtered_df[data._get_column_name('event_id')].nunique()}")
   ```

2. **Your filter is too restrictive**:
   ```python
   # Try expanding the filter range
   data.plot(
       photons=(0, 100),      # Wider range
       color_by='events',
       show_legend=True
   )
   ```

3. **Event/photon association might not be working correctly**:
   ```python
   # Check if association data is present
   data.info()  # Look for "Events:" and "Photons:" in output
   ```

### Gray Blobs Instead of Colors

If you see gray blobs instead of colors:

1. **Make sure you specify a colormap**:
   ```python
   # This will give you gray blobs (no colormap)
   data.plot(color_by='events')

   # This will give you colors
   data.plot(color_by='events', cmap='viridis')
   ```

2. **Non-associated pixels are shown in gray by default with show_assoc=False**:
   ```python
   # If you see gray pixels, they might be non-associated
   # By default, non-associated pixels are filtered out (show_assoc=True)
   # To see them as gray:
   data.plot(
       photons=(1, 20),
       color_by='events',
       cmap='viridis',
       show_assoc=False,      # Show non-associated pixels in gray
       show_legend=True       # Legend will show "Not associated"
   )

   # To hide non-associated pixels (default):
   data.plot(
       photons=(1, 20),
       color_by='events',
       cmap='viridis',
       show_assoc=True,       # Default - filters out non-associated pixels
       show_legend=True
   )
   ```

3. **Check that the data has the required columns**:
   ```python
   # Verify photon/event ID columns exist
   print(data.has_photon_id)  # Should be True for photon coloring
   print(data.has_event_id)   # Should be True for event coloring
   ```

### Legend Shows Wrong Values

The legend automatically adapts based on `color_by`:
- `color_by='toa'`: Shows time bins (e.g., "10 ns", "20 ns")
- `color_by='tot'`: Shows TOT values
- `color_by='photons'`: Shows "Photon 0", "Photon 1", etc.
- `color_by='events'`: Shows "Event 0", "Event 1", etc.

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
    color_by='toa',            # 'toa', 'tot', 'photon'/'photons', 'event'/'events'
    show_assoc=True,           # When using photon/event coloring, filter non-associated pixels
    cmap=None,                 # Matplotlib colormap

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
