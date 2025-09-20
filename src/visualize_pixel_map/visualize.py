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
    """
    combined_hist = np.zeros_like(h[keys[start_key_idx]])
    for i in range(num_bins):
        combined_hist += h[keys[start_key_idx + i]]
    
    if np.sum(combined_hist) == 0:
        return None, None
    
    y_indices, x_indices = np.meshgrid(np.arange(combined_hist.shape[0]), 
                                      np.arange(combined_hist.shape[1]), 
                                      indexing='ij')
    
    total_weight = np.sum(combined_hist)
    cog_x = np.sum(x_indices * combined_hist) / total_weight
    cog_y = np.sum(y_indices * combined_hist) / total_weight
    
    return int(cog_x), int(cog_y)

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

def plot_time_development(h, keys, start_key_idx=800, zoom_region=None, zoom_size=20,
                         custom_color=None, show_background=False, despine=True,
                         time_bins=None, show_scale=False, cmap=None,
                         show_labels=True, show_legend=False):
    """
    Clean publication-ready plot of pixel hit time development with envelope contours.
    
    Parameters:
    h: dictionary with histograms
    keys: list of dictionary keys (timestamps)
    start_key_idx: starting index for the time bins
    zoom_region: tuple (x_min, x_max, y_min, y_max) for zoom region, or None for auto
    zoom_size: size of square region around center of gravity
    custom_color: string color name (e.g., 'red', 'blue') or None for grayscale (deprecated, use cmap)
    show_background: whether to show the background histogram
    despine: whether to remove axes spines (default True)
    time_bins: maximum time in nanoseconds (e.g., 40 for [10, 20, 30, 40], 60 for [10, 20, 30, 40, 50, 60]),
               or None for default [10, 20, 30, 40]
    show_scale: whether to display a scale with pixel range at the bottom (default False)
    cmap: matplotlib colormap name or object for coloring timestamped pixels (default None for grayscale)
    show_labels: whether to display timestamp text labels on pixels (default True)
    show_legend: whether to display a legend with timestamp colors (default False)
    
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
    
    if zoom_region is None:
        cog_x, cog_y = calculate_center_of_gravity(h, keys, start_key_idx)
        if cog_x is None:
            zoom_region = (180, 200, 150, 170)
        else:
            half_size = zoom_size // 2
            zoom_region = (cog_x - half_size, cog_x + half_size,
                          cog_y - half_size, cog_y + half_size)
    
    x_min, x_max, y_min, y_max = zoom_region
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if show_background:
        combined_hist = (h[keys[start_key_idx]] + 
                        2 * h[keys[start_key_idx + 1]] + 
                        3 * h[keys[start_key_idx + 2]] + 
                        4 * h[keys[start_key_idx + 3]])
        ax.imshow(combined_hist, cmap=cmap if cmap else "gray", origin="lower", alpha=0.3)
    
    # Define time bins based on maximum time
    if time_bins is None:
        time_bins_max = 40
    else:
        time_bins_max = time_bins
    time_bins = [{'time': t, 'face_color': str(t/100), 'edge_color': '0.1', 'alpha': 0.6 + 0.3 * (t/100)} 
                 for t in range(10, time_bins_max + 1, 10)]
    
    # Apply cmap to face colors if provided
    if cmap and len(time_bins) > 1:
        norm = plt.Normalize(min(tb['time'] for tb in time_bins), max(tb['time'] for tb in time_bins))
        for i, bin_info in enumerate(time_bins):
            rgba = plt.cm.get_cmap(cmap)(norm(bin_info['time']))
            bin_info['face_color'] = rgba[:3]  # Use RGB only, alpha handled separately
    elif custom_color:
        # Deprecation warning for custom_color
        import warnings
        warnings.warn("The 'custom_color' parameter is deprecated. Use 'cmap' instead.", DeprecationWarning)
        for bin_info in time_bins:
            bin_info['face_color'] = custom_color

    # Collect legend handles
    legend_patches = []
    for i, bin_info in enumerate(time_bins):
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
            
            if envelope_points and len(envelope_points) >= 3:
                polygon = Polygon(envelope_points, 
                                facecolor=bin_info['face_color'],
                                edgecolor=bin_info['edge_color'],
                                linewidth=1.5,
                                alpha=bin_info['alpha'])
                ax.add_patch(polygon)
            
            if show_labels:
                for x, y in cluster:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        ax.text(x, y, str(bin_info['time']), 
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
        ax.legend(legend_patches, [str(tb['time']) + ' ns' for tb in time_bins],
                 loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig, ax