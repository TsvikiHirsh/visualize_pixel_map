# visualize_pixel_map

A Python package to visualize pixel hit time development from empirphot CSV data.

## Installation

Install the package via pip:

```bash
pip install visualize_pixel_map
```

## Usage

### Prerequisites
- A CSV file exported from an empirphot file with columns: `x`, `y`, `toa`, `tof`.

### Example

```python
from visualize_pixel_map import Data, visualize

# Load and prepare data
data = Data.from_csv("2024-09-11T160534_000000.empirphot.csv")
data.prepare_histograms(bins=np.arange(0, 0.01, 1e-8))

# Visualize
fig, ax = visualize.plot_time_development(data.h, data.keys, start_key_idx=0, zoom_size=20, despine=True)
plt.show()
```

## Features
- Plots pixel hit time development with envelope contours.
- Supports custom colors and zoom regions.
- Option to remove axes spines (`despine=True`).

## License
MIT License