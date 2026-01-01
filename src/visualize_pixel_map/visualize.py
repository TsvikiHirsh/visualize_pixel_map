import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch

def find_contour_envelope(binary_mask):
    """
    Find the envelope (contour) around a binary mask of connected pixels.
    
    Parameters:
    binary_mask: 2D boolean array where True indicates pixel hits
    
    Returns:
    List of (x, y) coordinates forming the envelope
    """
    if not np.any(binary_mask):
        return []
    
    padded_mask = np.pad(binary_mask, 1, mode='constant', constant_values=0)
    boundary = np.zeros_like(padded_mask)
    
    for i in range(1, padded_mask.shape[0] - 1):
        for j in range(1, padded_mask.shape[1] - 1):
            if padded_mask[i, j]:
                neighbors = padded_mask[i-1:i+2, j-1:j+2]
                if not np.all(neighbors):
                    boundary[i, j] = True
    
    y_coords, x_coords = np.where(boundary)
    x_coords = x_coords - 1
    y_coords = y_coords - 1
    
    if len(x_coords) == 0:
        return []
    
    envelope_points = []
    pixel_corners = set()
    for x, y in zip(x_coords, y_coords):
        corners = [(x-0.5, y-0.5), (x+0.5, y-0.5), 
                  (x+0.5, y+0.5), (x-0.5, y+0.5)]
        pixel_corners.update(corners)
    
    return list(pixel_corners)

def calculate_center_of_gravity(h, keys, start_key_idx, num_bins=4):
    """
    Calculate center of gravity of the combined event.

    Parameters
    ----------
    h : dict of 2D arrays
        Histograms (image frames).
    keys : list
        List of keys (timestamps).
    start_key_idx : int
        Starting index.
    num_bins : int
        How many consecutive bins to combine.

    Returns
    -------
    (cog_x, cog_y) : tuple of floats
        Center of gravity coordinates, or (None, None) if empty.
    """
    # Validate indices
    if start_key_idx >= len(keys):
        return None, None
    
    combined_hist = np.zeros_like(h[keys[start_key_idx]], dtype=float)
    actual_bins_used = 0
    
    for i in range(num_bins):
        if start_key_idx + i < len(keys):
            combined_hist += h[keys[start_key_idx + i]]
            actual_bins_used += 1

    if np.sum(combined_hist) == 0 or actual_bins_used == 0:
        return None, None

    # Note: shape is (rows=y, cols=x)
    y_indices, x_indices = np.indices(combined_hist.shape)

    total_weight = np.sum(combined_hist)
    cog_x = np.sum(x_indices * combined_hist) / total_weight
    cog_y = np.sum(y_indices * combined_hist) / total_weight

    return int(cog_x), int(cog_y)

def calculate_optimal_zoom(h, keys, start_key_idx, num_bins=4, margin=5):
    """
    Calculate optimal zoom region based on actual data extent.
    
    Parameters
    ----------
    h : dict of 2D arrays
        Histograms (image frames).
    keys : list
        List of keys (timestamps).
    start_key_idx : int
        Starting index.
    num_bins : int
        How many consecutive bins to check.
    margin : int
        Extra pixels to add around the data.
    
    Returns
    -------
    (x_min, x_max, y_min, y_max) : tuple of ints
        Zoom region, or None if no data found.
    """
    # Combine histograms to find data extent
    combined_hist = np.zeros_like(h[keys[start_key_idx]], dtype=float)
    
    for i in range(num_bins):
        if start_key_idx + i < len(keys):
            combined_hist += h[keys[start_key_idx + i]]
    
    # Find all non-zero pixels
    y_indices, x_indices = np.where(combined_hist > 0)
    
    if len(x_indices) == 0:
        return None
    
    # Calculate bounding box
    x_min = max(0, np.min(x_indices) - margin)
    x_max = min(combined_hist.shape[1], np.max(x_indices) + margin + 1)
    y_min = max(0, np.min(y_indices) - margin)
    y_max = min(combined_hist.shape[0], np.max(y_indices) + margin + 1)
    
    # Ensure minimum size
    min_size = 10
    if x_max - x_min < min_size:
        center_x = (x_min + x_max) / 2
        x_min = max(0, int(center_x - min_size/2))
        x_max = min(combined_hist.shape[1], int(center_x + min_size/2))
    
    if y_max - y_min < min_size:
        center_y = (y_min + y_max) / 2
        y_min = max(0, int(center_y - min_size/2))
        y_max = min(combined_hist.shape[0], int(center_y + min_size/2))
    
    return (x_min, x_max, y_min, y_max)

def find_connected_clusters(pixel_coords):
    """
    Find connected clusters of pixels using 4-connectivity.
    """
    if len(pixel_coords) == 0:
        return []
    
    pixel_set = set(pixel_coords)
    visited = set()
    clusters = []
    
    def get_neighbors(x, y):
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    
    def flood_fill(start_pixel):
        cluster = []
        stack = [start_pixel]
        
        while stack:
            pixel = stack.pop()
            if pixel in visited or pixel not in pixel_set:
                continue
                
            visited.add(pixel)
            cluster.append(pixel)
            
            for neighbor in get_neighbors(*pixel):
                if neighbor in pixel_set and neighbor not in visited:
                    stack.append(neighbor)
        
        return cluster
    
    for pixel in pixel_coords:
        if pixel not in visited:
            cluster = flood_fill(pixel)
            if cluster:
                clusters.append(cluster)
    
    return clusters

def create_grid_envelope(cluster_pixels):
    """
    Create grid-aligned envelope using a simple rectangular approach.
    """
    if len(cluster_pixels) == 0:
        return []
    
    coords = np.array(cluster_pixels)
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    
    envelope = [
        (min_x - 0.5, min_y - 0.5),
        (max_x + 0.5, min_y - 0.5),
        (max_x + 0.5, max_y + 0.5),
        (min_x - 0.5, max_y + 0.5)
    ]
    
    return envelope

def create_pixel_boundary_envelope(cluster_pixels):
    """
    Create envelope by tracing pixel boundaries.
    """
    if len(cluster_pixels) == 0:
        return []
    
    pixel_set = set(cluster_pixels)
    
    # Find all external edge segments
    edge_segments = []
    for x, y in cluster_pixels:
        if (x, y-1) not in pixel_set:
            edge_segments.append(((x-0.5, y-0.5), (x+0.5, y-0.5)))
        if (x+1, y) not in pixel_set:
            edge_segments.append(((x+0.5, y-0.5), (x+0.5, y+0.5)))
        if (x, y+1) not in pixel_set:
            edge_segments.append(((x+0.5, y+0.5), (x-0.5, y+0.5)))
        if (x-1, y) not in pixel_set:
            edge_segments.append(((x-0.5, y+0.5), (x-0.5, y-0.5)))
    
    if not edge_segments:
        return create_grid_envelope(cluster_pixels)
    
    # Build a graph of edge connections
    point_connections = {}
    for p1, p2 in edge_segments:
        if p1 not in point_connections:
            point_connections[p1] = []
        if p2 not in point_connections:
            point_connections[p2] = []
        point_connections[p1].append(p2)
        point_connections[p2].append(p1)
    
    # Trace the boundary path
    if not point_connections:
        return create_grid_envelope(cluster_pixels)
    
    all_points = list(point_connections.keys())
    start_point = min(all_points, key=lambda p: (p[0], p[1]))
    
    path = []
    current = start_point
    prev = None
    
    for _ in range(len(edge_segments) * 2):
        path.append(current)
        next_options = [p for p in point_connections[current] if p != prev]
        if not next_options:
            break
        next_point = next_options[0]
        if next_point == start_point and len(path) > 2:
            break
        prev = current
        current = next_point
    
    if len(path) >= 3:
        return path
    return create_grid_envelope(cluster_pixels)

def plot_time_development(h, keys, start_key_idx=800, zoom_region=None, zoom_size=None,
                         custom_color=None, show_background=False, despine=True,
                         time_bins=None, show_scale=False, cmap=None,
                         show_labels=True, show_legend=False, auto_zoom_margin=5,
                         color_by='toa', df=None):
    """
    Clean publication-ready plot of pixel hit time development with envelope contours.

    Parameters:
    h: dictionary with histograms
    keys: list of dictionary keys (timestamps)
    start_key_idx: starting index for the time bins
    zoom_region: tuple (x_min, x_max, y_min, y_max) for zoom region, or None for auto
    zoom_size: if provided, creates a square region of this size around CoG (overrides auto zoom)
    custom_color: string color name (e.g., 'red', 'blue') or None for grayscale (deprecated, use cmap)
    show_background: whether to show the background histogram
    despine: whether to remove axes spines (default True)
    time_bins: maximum time in nanoseconds (e.g., 40 for [10, 20, 30, 40], 60 for [10, 20, 30, 40, 50, 60]),
               or None for default [10, 20, 30, 40]
    show_scale: whether to display a scale with pixel range at the bottom (default False)
    cmap: matplotlib colormap name or object for coloring timestamped pixels (default None for grayscale)
    show_labels: whether to display timestamp text labels on pixels (default True)
    show_legend: whether to display a legend with timestamp colors (default False)
    auto_zoom_margin: margin in pixels around data when auto-zooming (only used if zoom_size not set)
    color_by: 'toa' or 'tot' - which parameter to use for colormap (default 'toa')
    df: pandas DataFrame with pixel data (required if color_by='tot')

    Returns:
    Tuple of (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Control spine visibility based on despine parameter
    if despine and not show_scale:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Determine number of time bins to use
    if time_bins is None:
        time_bins_max = 40
    else:
        time_bins_max = time_bins
    num_bins = time_bins_max // 10
    
    # Calculate zoom region if not provided
    if zoom_region is None:
        # If zoom_size is specified, use CoG-based zoom with specified size
        if zoom_size is not None:
            cog_x, cog_y = calculate_center_of_gravity(h, keys, start_key_idx, num_bins)
            
            if cog_x is not None and cog_y is not None:
                half_size = zoom_size // 2
                zoom_region = (cog_x - half_size, cog_x + half_size,
                              cog_y - half_size, cog_y + half_size)
            else:
                # If CoG fails, fall back to optimal zoom
                zoom_region = calculate_optimal_zoom(h, keys, start_key_idx, num_bins, auto_zoom_margin)
        else:
            # Use automatic optimal zoom based on data extent
            zoom_region = calculate_optimal_zoom(h, keys, start_key_idx, num_bins, auto_zoom_margin)
        
        # If all methods fail, use default region
        if zoom_region is None:
            print(f"Warning: No data found at start_key_idx={start_key_idx}. Using default zoom.")
            zoom_region = (100, 150, 100, 150)
    
    x_min, x_max, y_min, y_max = zoom_region
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if show_background:
        combined_hist = np.zeros_like(h[keys[start_key_idx]], dtype=float)
        for i in range(num_bins):
            if start_key_idx + i < len(keys):
                combined_hist += (i + 1) * h[keys[start_key_idx + i]]
        ax.imshow(combined_hist, cmap=cmap if cmap else "gray", origin="lower", alpha=0.3)
    
    # Check if we can use TOT coloring
    use_tot_coloring = (color_by == 'tot' and df is not None and 'tot' in df.columns)

    # Define time bins based on maximum time
    time_bins_list = [{'time': t, 'face_color': str(t/100), 'edge_color': '0.1', 'alpha': 0.6 + 0.3 * (t/100)}
                      for t in range(10, time_bins_max + 1, 10)]

    # Prepare TOT-based coloring if requested
    if use_tot_coloring:
        # Get TOT range for normalization
        tot_values = df['tot'].dropna()
        if len(tot_values) > 0:
            tot_min, tot_max = tot_values.min(), tot_values.max()
            tot_norm = plt.Normalize(tot_min, tot_max)
            # We'll apply colors per-cluster based on actual TOT values
        else:
            use_tot_coloring = False  # Fall back to TOA if no TOT data

    # Apply cmap to face colors if provided (for TOA mode)
    if not use_tot_coloring:
        if cmap and len(time_bins_list) > 1:
            norm = plt.Normalize(min(tb['time'] for tb in time_bins_list), max(tb['time'] for tb in time_bins_list))
            for i, bin_info in enumerate(time_bins_list):
                rgba = plt.cm.get_cmap(cmap)(norm(bin_info['time']))
                bin_info['face_color'] = rgba[:3]  # Use RGB only, alpha handled separately
        elif custom_color:
            # Deprecation warning for custom_color
            import warnings
            warnings.warn("The 'custom_color' parameter is deprecated. Use 'cmap' instead.", DeprecationWarning)
            for bin_info in time_bins_list:
                bin_info['face_color'] = custom_color

    # Collect legend handles
    legend_patches = []
    for i, bin_info in enumerate(time_bins_list):
        hist_data = h[keys[start_key_idx + i]] if start_key_idx + i < len(keys) else np.zeros_like(h[keys[0]])
        zoom_hist = hist_data[y_min:y_max, x_min:x_max]
        y_indices, x_indices = np.where(zoom_hist > 0)

        if len(x_indices) == 0:
            continue

        pixel_coords = [(x + x_min, y + y_min) for x, y in zip(x_indices, y_indices)]
        clusters = find_connected_clusters(pixel_coords)

        for cluster in clusters:
            envelope_points = create_pixel_boundary_envelope(cluster)
            if not envelope_points or len(envelope_points) < 3:
                envelope_points = create_grid_envelope(cluster)

            # Initialize variables for TOT tracking
            cluster_tots = []
            mean_tot = 0

            if envelope_points and len(envelope_points) >= 3:
                # Determine color based on TOA or TOT
                if use_tot_coloring:
                    # Get TOT values for pixels in this cluster
                    for x, y in cluster:
                        # Find pixels in dataframe matching this position
                        # Round to integer pixel coordinates
                        pixel_mask = (df['x'].round().astype(int) == int(x)) & (df['y'].round().astype(int) == int(y))
                        pixel_tots = df.loc[pixel_mask, 'tot'].dropna()
                        if len(pixel_tots) > 0:
                            cluster_tots.extend(pixel_tots.tolist())

                    if len(cluster_tots) > 0:
                        # Use mean TOT for this cluster
                        mean_tot = np.mean(cluster_tots)
                        if cmap:
                            rgba = plt.cm.get_cmap(cmap)(tot_norm(mean_tot))
                            face_color = rgba[:3]
                        else:
                            # Grayscale based on TOT
                            face_color = str(tot_norm(mean_tot))
                    else:
                        # No TOT data, use default
                        face_color = bin_info['face_color']
                else:
                    # Use TOA-based color
                    face_color = bin_info['face_color']

                polygon = Polygon(envelope_points,
                                facecolor=face_color,
                                edgecolor=bin_info['edge_color'],
                                linewidth=1.5,
                                alpha=bin_info['alpha'])
                ax.add_patch(polygon)

            if show_labels:
                # Label shows TOT value if using TOT coloring, otherwise time
                if use_tot_coloring and len(cluster_tots) > 0:
                    label_text = f"{mean_tot:.0f}"
                else:
                    label_text = str(bin_info['time'])

                for x, y in cluster:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        ax.text(x, y, label_text,
                               ha='center', va='center',
                               fontsize=6, fontweight='normal',
                               color='black')

        # Add to legend if not already included
        if show_legend and not any(p.get_facecolor() == bin_info['face_color'] for p in legend_patches):
            legend_patch = plt.Rectangle((0,0), 1, 1, fc=bin_info['face_color'], ec=bin_info['edge_color'], alpha=bin_info['alpha'])
            legend_patches.append(legend_patch)

    # Add scale if requested
    if show_scale and not despine:
        ax.spines['bottom'].set_visible(True)
        pixel_range = int(x_max - x_min)
        ax.set_xticks([x_min, x_max])
        ax.set_xticklabels([f"-{pixel_range//2}", f"{pixel_range//2}"])
        ax.set_xlabel("Pixels from Center")

    # Add legend if requested
    if show_legend and legend_patches:
        ax.legend(legend_patches, [str(tb['time']) + ' ns' for tb in time_bins_list],
                 loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig, ax