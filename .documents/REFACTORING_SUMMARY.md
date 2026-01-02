# visualize_pixel_map Refactoring Summary

## Major Changes

### 1. **Lazy Histogram Computation** ✅
- **Before:** Histograms computed during `__init__`, causing issues with filtering
- **After:** Histograms created on-demand when `plot()` is called
- **Benefit:** Much more efficient, no pre-computation overhead

### 2. **Filter-Then-Histogram Approach** ✅
```python
# Old approach (problematic):
__init__ → compute histograms → filter data → recompute histograms → plot

# New approach (efficient):
plot() → filter data → compute histograms once → plot
```

### 3. **Pandas Query Support** ✅
```python
# New query parameter for maximum flexibility
data.plot(query="tot > 50 & x < 128")
data.plot(query="assoc_event_id.notna() & toa < 0.005")
```

### 4. **Verbosity Levels** ✅
```python
# 0 = QUIET - no output
data = vpm.Data("data/neutrons", verbosity=0)
data.plot(verbosity=0)

# 1 = BASIC - progress bars and minimal info (default)
data = vpm.Data("data/neutrons", verbosity=1)
data.plot(verbosity=1)

# 2 = ADVANCED - detailed filtering info
data = vpm.Data("data/neutrons", verbosity=2)
data.plot(verbosity=2)
```

## Usage Examples

### Basic Usage
```python
import visualize_pixel_map as vpm

# Load data (no histogram computation yet!)
data = vpm.Data("data/neutrons", verbosity=1)

# Plot with TOA range (filters first, then creates histograms)
data.plot(toa_range=(0, 0.006), cmap='viridis')
```

### Advanced Filtering
```python
# Combine multiple filters
data.plot(
    events=slice(0, 10),           # First 10 events
    toa_range=(0, 0.005),          # 0-5ms
    query="tot > 30",              # High energy hits
    color_by='tot',                # Color by TOT
    cmap='plasma',
    verbosity=2                    # Show detailed filtering info
)
```

Output with `verbosity=2`:
```
  - TOA range filter: 45231 rows remain
Filtered: 45,231 rows (from 98,426, 46.0%)
Creating histograms: 100%|████████| 500/500 [00:01<00:00]
Created 500 histograms covering 45231 pixels
```

### Query Examples
```python
# Filter by TOT threshold
data.plot(query="tot > 50")

# Complex boolean logic
data.plot(query="(tot > 30 & tot < 100) | (x > 200)")

# Use association data
data.plot(query="assoc_event_id.notna() & pixel_spatial_diff_px < 5")

# Combine with other filters
data.plot(
    photons=(0, 20),
    query="tot > 40",
    color_by='tot'
)
```

### Inspect Before Plotting
```python
# Check available data
data.info()

# Check specific ranges
print(f"Events: {data.events['count']}")
print(f"Time range: {data.times['range']}")

# Access dataframe for custom analysis
df = data.associated_df
high_tot = df[df['tot'] > 100]
print(f"High TOT hits: {len(high_tot)}")
```

## Removed Methods

The following methods were removed as they're no longer needed:

- `prepare_histograms()` - Replaced by `_create_histograms()` (internal)
- `get_neutron_ids()` - Use `data.neutrons` property instead
- `get_keys_for_neutron()` - Not needed with new filter approach
- `_do_plot()` - Merged into `plot()` method

## Performance Improvements

1. **No pre-computation:** Initialization is instant
2. **Filter-first:** Only histogram filtered data
3. **Pandas optimization:** Leverages pandas query engine
4. **Memory efficient:** Don't store unused histograms

## Migration Guide

### Old Code
```python
data = vpm.Data("data.csv", start_time=0, end_time=0.01)  # Slow - computes histograms
data.plot(key=800)
```

### New Code
```python
data = vpm.Data("data/neutrons")  # Fast - no histogram computation
data.plot(key=0, toa_range=(0.008, 0.01))  # Computes only needed histograms
```

### With Filters
```python
# Old (problematic)
data.plot(photon_range=(0, 10))  # Had to recompute histograms

# New (efficient)
data.plot(photons=(0, 10))  # Filters first, then histograms once
```

## Verbosity Levels in Detail

### Level 0 (QUIET)
- No output at all
- Silent operation
- Use for scripts/automation

### Level 1 (BASIC) - Default
- Shows progress bars
- Shows filtered row count
- Minimal essential info

### Level 2 (ADVANCED)
- Shows each filter operation
- Shows rows remaining after each filter
- Shows histogram creation details
- Useful for debugging

## Example with All Features
```python
import visualize_pixel_map as vpm

# Load with advanced verbosity
data = vpm.Data("tests/data/neutrons", verbosity=2)

# Check what's available
data.info()

# Plot with complex filtering
fig, ax = data.plot(
    # Filtering
    events=slice(5, 15),                       # Events 5-15
    toa_range=(0, 0.003),                      # 0-3ms
    query="tot > 40 & assoc_event_id.notna()", # Custom query

    # Visualization
    color_by='tot',                             # Color by TOT
    cmap='plasma',
    time_bins=80,
    show_labels=True,
    zoom_size=40,

    # Verbosity
    verbosity=2                                 # Show details
)

# Save
fig.savefig('output.png', dpi=300, bbox_inches='tight')
```

## Benefits Summary

✅ **Faster initialization** - No pre-computation
✅ **More flexible** - Query parameter for complex filters
✅ **Better error messages** - Clear feedback on what's wrong
✅ **Memory efficient** - Only create needed histograms
✅ **Cleaner code** - Simpler internal logic
✅ **Better verbosity** - 3 levels of output control
