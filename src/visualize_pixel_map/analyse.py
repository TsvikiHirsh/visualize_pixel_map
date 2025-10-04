import pandas as pd
import numpy as np


class Data:
    """Class to hold and process pixel data from empirphot CSV files."""
    
    def __init__(self, csv_filepath="data.empirphot.csv", start_time=0, end_time=0.01, 
                 time_step=10, sensor_size=8, neutron_id=False, verbosity=1):
        """
        Initialize Data with CSV file and histogram bin parameters.
        
        Parameters:
            csv_filepath (str): Path to the empirphot CSV file.
            start_time (float): Start time in seconds for histogram bins.
            end_time (float): End time in seconds for histogram bins.
            time_step (float): Time step in nanoseconds for bin edges.
            sensor_size (float): Size of the sensor in mm (used for normalization).
            neutron_id (bool): If True, group histograms by neutron_id (for simulation data).
            verbosity (int): If 1, show progress bar during initialization. If 0, silent.
        """
        from tqdm.notebook import tqdm
        
        self.verbosity = verbosity
        
        if self.verbosity:
            print(f"Loading CSV file: {csv_filepath}")
        
        self.df = pd.read_csv(csv_filepath)
        self.neutron_id = neutron_id
        self.has_neutron_id = 'neutron_id' in self.df.columns
        
        if self.verbosity:
            print(f"Loaded {len(self.df)} rows")
        
        self._normalize_columns(sensor_size)
        self.h = {}
        self.keys = []
        
        # Convert time_step from nanoseconds to seconds
        time_step_sec = time_step * 1e-9
        bins = np.arange(start_time, end_time + time_step_sec, time_step_sec)
        self.prepare_histograms(bins)
    
    def _normalize_columns(self, sensor_size=8):
        """
        Detect the format and normalize column names to x, y, toa, tof.
        Supports two formats:
        1. Original format: x, y, toa, tof
        2. Alternative format: x2, y2, z2, id, neutron_id, toa2, photon_count, time_diff
        Parameters:
            sensor_size (float): Size of the sensor in mm (used for normalization).
        """
        columns = self.df.columns.tolist()
        
        # Check if this is the alternative format (has x2, y2, toa2)
        if 'x2' in columns and 'y2' in columns and 'toa2' in columns:
            # Alternative format - select relevant columns and drop NaNs
            cols_to_keep = ["x2", "y2", "toa2"]
            if 'neutron_id' in columns:
                cols_to_keep.append('neutron_id')
            
            self.df = self.df[cols_to_keep].dropna()
            
            # Convert toa2 from nanoseconds to seconds
            self.df["toa2"] *= 1e-9
            
            # Convert between mm and pixels (x2, y2 are in mm, range -sensor_size to +sensor_size)
            self.df["x"] = (self.df["x2"] + sensor_size) / sensor_size * 128
            self.df["y"] = (self.df["y2"] + sensor_size) / sensor_size * 128
            
            # Set toa and tof (duplicate toa2 as tof)
            self.df["toa"] = self.df["toa2"]
            self.df["tof"] = self.df["toa2"]
            
            # Filter: keep only data where time is between 0 and 20 second
            self.df = self.df.loc[(self.df["toa"] >= 0)]
            self.df = self.df.sort_values(by="toa")
            
            # Keep only the normalized columns
            final_cols = ['x', 'y', 'toa', 'tof']
            if 'neutron_id' in self.df.columns:
                final_cols.append('neutron_id')
            self.df = self.df[final_cols]
        
        elif 'x' in columns and 'y' in columns and 'toa' in columns:
            # Original format - just ensure column order
            if 'tof' not in columns:
                # If tof doesn't exist, duplicate toa
                self.df['tof'] = self.df['toa']
            
            final_cols = ['x', 'y', 'toa', 'tof']
            if 'neutron_id' in columns:
                final_cols.append('neutron_id')
            self.df = self.df[final_cols]
        
        else:
            raise ValueError(
                f"Unrecognized CSV format. Expected either "
                f"(x, y, toa, tof) or (x2, y2, toa2, ...). "
                f"Found columns: {columns}"
            )
    
    def prepare_histograms(self, bins):
        """Prepare 2D histograms from the data."""
        from tqdm.notebook import tqdm
        
        self.h = {}
        
        if self.neutron_id and self.has_neutron_id:
            # Group by both time bin and neutron_id
            self.df["tbin"] = pd.cut(self.df["toa"], bins, labels=bins[:-1])
            df1 = self.df.query("toa < @bins[-1]")
            
            # Create composite keys: (time_bin, neutron_id)
            grouped = df1.groupby(["tbin", "neutron_id"], observed=False)
            
            # Get total count for progress bar
            total = len(grouped) if self.verbosity else None
            
            # Iterate directly with tqdm wrapper
            if self.verbosity:
                iterator = tqdm(grouped, desc="Preparing histograms", total=total)
            else:
                iterator = grouped
            
            for (tbin, nid), group in iterator:
                try:
                    if not group.empty and not pd.isna(tbin):
                        key = (tbin, int(nid))
                        self.h[key], x_edges, y_edges = np.histogram2d(
                            group["x"], group["y"], bins=[np.arange(256), np.arange(256)]
                        )
                except Exception:
                    continue
            
            self.keys = sorted(self.h.keys())
        else:
            # Original behavior: group by time bin only
            self.df["tbin"] = pd.cut(self.df["toa"], bins, labels=bins[:-1])
            df1 = self.df.query("toa < @bins[-1]").set_index("tbin")
            
            unique_bins = df1.index.unique()
            
            # Iterate directly with tqdm wrapper
            if self.verbosity:
                iterator = tqdm(unique_bins, desc="Preparing histograms")
            else:
                iterator = unique_bins
            
            for tbin in iterator:
                try:
                    subset = df1.loc[tbin]
                    if not subset.empty:
                        self.h[tbin], x_edges, y_edges = np.histogram2d(
                            subset["x"], subset["y"], bins=[np.arange(256), np.arange(256)]
                        )
                except Exception:
                    continue
            
            self.keys = list(self.h.keys())
        
        if self.verbosity:
            print(f"Created {len(self.h)} histograms")
    
    def get_neutron_ids(self):
        """
        Get list of unique neutron IDs in the data.
        
        Returns:
            list: Sorted list of neutron IDs, or empty list if not available.
        """
        if not self.has_neutron_id:
            return []
        
        if self.neutron_id:
            # Extract unique neutron_ids from composite keys
            return sorted(set(key[1] for key in self.keys if isinstance(key, tuple)))
        else:
            # Get from dataframe
            return sorted(self.df['neutron_id'].unique().tolist())
    
    def get_keys_for_neutron(self, neutron_id):
        """
        Get all time bin keys for a specific neutron.
        
        Parameters:
            neutron_id (int): The neutron ID to filter by.
        
        Returns:
            list: List of keys (time bins) for the specified neutron.
        """
        if not self.neutron_id or not self.has_neutron_id:
            raise ValueError("neutron_id grouping was not enabled during initialization")
        
        return [key for key in self.keys if isinstance(key, tuple) and key[1] == neutron_id]

    def plot(self, key=None, neutron_id_filter=None, zoom_region=None, zoom_size=None,
             custom_color=None, show_background=False, despine=True,
             time_bins=None, show_scale=False, cmap=None,
             show_labels=True, show_legend=False, auto_zoom_margin=5):
        """
        Plot the pixel hit time development using the stored histograms.
        
        Parameters:
            key (int or tuple): Starting index for the time bins. If neutron_id is enabled,
                               can be tuple (time_index, neutron_id) or just time_index.
            neutron_id_filter (int): If provided, only plot data for this specific neutron.
            zoom_region (tuple): (x_min, x_max, y_min, y_max) for zoom region.
            zoom_size (int): If provided, creates a square zoom region of this size around CoG.
                           Overrides automatic zoom. If None, uses automatic zoom.
            custom_color (str): Custom color map for plotting (deprecated, use cmap).
            show_background (bool): Whether to show background pixels.
            despine (bool): Whether to remove plot spines.
            time_bins: maximum time in nanoseconds (e.g., 40 for [10, 20, 30, 40]).
            show_scale (bool): Whether to show scale on the plot.
            cmap (str): Colormap to use for the plot.
            show_labels (bool): Whether to display timestamp text labels on pixels.
            show_legend (bool): Whether to display a legend with timestamp colors.
            auto_zoom_margin (int): Margin in pixels around data when auto-zooming (if zoom_size not set).
        """
        from visualize_pixel_map.visualize import plot_time_development
        
        # Determine the starting key
        if key is None:
            key = 0
        
        # Handle neutron_id filtering or when using composite keys
        if self.neutron_id and self.has_neutron_id:
            if neutron_id_filter is not None:
                # Get keys for specific neutron
                neutron_keys = self.get_keys_for_neutron(neutron_id_filter)
                if not neutron_keys:
                    raise ValueError(f"No data found for neutron_id={neutron_id_filter}")
                
                # Find the appropriate starting key
                if isinstance(key, int):
                    if key >= len(neutron_keys):
                        raise ValueError(f"key index {key} out of range for neutron {neutron_id_filter}")
                    start_key = neutron_keys[key]
                else:
                    start_key = key
                
                # Create a filtered view of histograms and keys for this neutron
                filtered_h = {k: v for k, v in self.h.items() if isinstance(k, tuple) and k[1] == neutron_id_filter}
                filtered_keys = neutron_keys
                
                # Convert composite keys to indices for plotting
                start_key_idx = filtered_keys.index(start_key)
                
                plot_time_development(filtered_h, filtered_keys, start_key_idx, 
                                    zoom_region, zoom_size, custom_color, show_background, 
                                    despine, time_bins, show_scale, cmap, 
                                    show_labels, show_legend, auto_zoom_margin)
            else:
                # neutron_id is enabled but no filter
                if isinstance(key, tuple):
                    # User provided a composite key like (time, neutron_id)
                    if key not in self.keys:
                        raise ValueError(f"Key {key} not found in data")
                    
                    # Extract neutron_id from the key and filter
                    neutron_id_filter = key[1]
                    neutron_keys = self.get_keys_for_neutron(neutron_id_filter)
                    filtered_h = {k: v for k, v in self.h.items() if isinstance(k, tuple) and k[1] == neutron_id_filter}
                    start_key_idx = neutron_keys.index(key)
                    
                    plot_time_development(filtered_h, neutron_keys, start_key_idx, 
                                        zoom_region, zoom_size, custom_color, show_background, 
                                        despine, time_bins, show_scale, cmap, 
                                        show_labels, show_legend, auto_zoom_margin)
                elif isinstance(key, int):
                    # User provided index - treat keys as if they were a flat list
                    # This allows plotting across all neutrons by index
                    if key >= len(self.keys):
                        raise ValueError(f"key index {key} out of range. Valid range: 0-{len(self.keys)-1}")
                    
                    # Use the key directly as an index into the full keys list
                    # When plotting, we need to determine which neutron this belongs to
                    actual_key = self.keys[key]
                    if isinstance(actual_key, tuple):
                        neutron_id_from_key = actual_key[1]
                        neutron_keys = self.get_keys_for_neutron(neutron_id_from_key)
                        filtered_h = {k: v for k, v in self.h.items() if isinstance(k, tuple) and k[1] == neutron_id_from_key}
                        start_key_idx = neutron_keys.index(actual_key)
                        
                        if self.verbosity:
                            print(f"Note: Using key index {key} which corresponds to neutron_id={neutron_id_from_key}, time bin index {start_key_idx}")
                        
                        plot_time_development(filtered_h, neutron_keys, start_key_idx, 
                                            zoom_region, zoom_size, custom_color, show_background, 
                                            despine, time_bins, show_scale, cmap, 
                                            show_labels, show_legend, auto_zoom_margin)
                    else:
                        # Shouldn't happen but handle gracefully
                        plot_time_development(self.h, self.keys, key, 
                                            zoom_region, zoom_size, custom_color, show_background, 
                                            despine, time_bins, show_scale, cmap, 
                                            show_labels, show_legend, auto_zoom_margin)
        else:
            # Standard plotting without neutron grouping
            if isinstance(key, tuple):
                raise ValueError("Composite keys only valid when neutron_id=True")
            
            start_key_idx = key
            
            plot_time_development(self.h, self.keys, start_key_idx, 
                                zoom_region, zoom_size, custom_color, show_background, 
                                despine, time_bins, show_scale, cmap, 
                                show_labels, show_legend, auto_zoom_margin)