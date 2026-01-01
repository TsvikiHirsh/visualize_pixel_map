import pandas as pd
import numpy as np
import json
from pathlib import Path


class Data:
    """Class to hold and process pixel data from empirphot CSV files or neutron data folders."""

    def __init__(self, data_path="data.empirphot.csv", start_time=0, end_time=0.01,
                 time_step=10, sensor_size=8, neutron_id=False, verbosity=1,
                 auto_associate=True, association_mode='full', settings=None):
        """
        Initialize Data with CSV file or folder path and histogram bin parameters.

        Parameters:
            data_path (str): Path to CSV file or folder containing neutron data.
                            If folder, will auto-detect AssociatedResults or run association.
            start_time (float): Start time in seconds for histogram bins.
            end_time (float): End time in seconds for histogram bins.
            time_step (float): Time step in nanoseconds for bin edges.
            sensor_size (float): Size of the sensor in mm (used for normalization).
            neutron_id (bool): If True, group histograms by neutron_id (for simulation data).
            verbosity (int): If 1, show progress bar during initialization. If 0, silent.
            auto_associate (bool): If True, automatically run neutron_event_analyzer when needed.
            association_mode (str): 'pixel_photon', 'photon_event', or 'full' (both).
            settings (str or dict): Path to settings.json or settings dictionary for association.
        """
        self.verbosity = verbosity
        self.data_path = data_path
        self.sensor_size = sensor_size
        self.auto_associate = auto_associate
        self.association_mode = association_mode
        self.settings = settings

        # Detect if input is file or folder
        path = Path(data_path)

        if path.is_file():
            # Legacy behavior: load CSV file directly
            if self.verbosity:
                print(f"Loading CSV file: {data_path}")
            self.df = pd.read_csv(data_path)
            self.data_source = 'csv_file'
        elif path.is_dir():
            # New behavior: detect and load from folder
            self.data_source = 'folder'
            self.df = self._load_from_folder(path)
        else:
            raise FileNotFoundError(f"Path not found: {data_path}")

        self.neutron_id = neutron_id
        self.has_neutron_id = 'neutron_id' in self.df.columns
        self.has_photon_id = 'assoc_photon_id' in self.df.columns
        self.has_event_id = 'assoc_event_id' in self.df.columns

        if self.verbosity:
            print(f"Loaded {len(self.df)} rows")
            if self.has_photon_id:
                print(f"  - Contains photon associations")
            if self.has_event_id:
                print(f"  - Contains event associations")

        self._normalize_columns(sensor_size)
        self.h = {}
        self.keys = []

        # Convert time_step from nanoseconds to seconds
        time_step_sec = time_step * 1e-9
        bins = np.arange(start_time, end_time + time_step_sec, time_step_sec)
        self.prepare_histograms(bins)

    def _load_from_folder(self, folder_path):
        """
        Load data from a neutron data folder.
        Auto-detects AssociatedResults or runs neutron_event_analyzer if needed.

        Parameters:
            folder_path (Path): Path to the neutron data folder.

        Returns:
            pd.DataFrame: Loaded data.
        """
        folder_path = Path(folder_path)

        # Check for AssociatedResults folder
        assoc_results_dir = folder_path / "AssociatedResults"

        if assoc_results_dir.exists() and assoc_results_dir.is_dir():
            # Look for CSV files in AssociatedResults
            csv_files = list(assoc_results_dir.glob("*.csv"))

            if csv_files:
                if self.verbosity:
                    print(f"Found AssociatedResults folder with {len(csv_files)} CSV file(s)")
                # Use the first CSV file (or most recent)
                csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                if self.verbosity:
                    print(f"Loading: {csv_file.name}")
                return pd.read_csv(csv_file)

        # No AssociatedResults found, check for Exported folders
        exported_folders = [
            folder_path / "ExportedPixels",
            folder_path / "ExportedPhotons",
            folder_path / "ExportedEvents"
        ]

        has_exported = any(f.exists() and f.is_dir() for f in exported_folders)

        if has_exported:
            if self.auto_associate:
                if self.verbosity:
                    print("No AssociatedResults found. Running neutron_event_analyzer...")
                return self._run_association(folder_path)
            else:
                raise FileNotFoundError(
                    f"No AssociatedResults found in {folder_path}. "
                    f"Set auto_associate=True to run association automatically."
                )
        else:
            raise FileNotFoundError(
                f"No AssociatedResults or Exported folders found in {folder_path}"
            )

    def _run_association(self, data_folder):
        """
        Run neutron_event_analyzer to create associations.

        Parameters:
            data_folder (Path): Path to the neutron data folder.

        Returns:
            pd.DataFrame: Associated data.
        """
        try:
            import neutron_event_analyzer as nea
        except ImportError:
            raise ImportError(
                "neutron_event_analyzer package not found. "
                "Install it or manually run association first."
            )

        # Load settings if provided
        settings_to_use = self.settings
        if settings_to_use is None:
            # Check for settings.json in the data folder
            settings_file = data_folder / "settings.json"
            if settings_file.exists():
                if self.verbosity:
                    print(f"Loading settings from: {settings_file}")
                with open(settings_file, 'r') as f:
                    settings_to_use = json.load(f)

        # Initialize analyzer
        if self.verbosity:
            print(f"Initializing neutron_event_analyzer for: {data_folder}")

        analyser = nea.Analyse(
            data_folder=str(data_folder),
            settings=settings_to_use,
            verbosity=self.verbosity
        )

        # Determine what to load based on association_mode
        load_pixels = self.association_mode in ['pixel_photon', 'full']
        load_photons = True  # Always load photons
        load_events = self.association_mode in ['photon_event', 'full']

        if self.verbosity:
            print(f"Loading data (pixels={load_pixels}, photons={load_photons}, events={load_events})...")

        analyser.load(
            pixels=load_pixels,
            photons=load_photons,
            events=load_events,
            verbosity=self.verbosity
        )

        # Run association
        if self.verbosity:
            print(f"Running {self.association_mode} association...")

        if self.association_mode == 'full':
            analyser.associate_full(verbosity=self.verbosity)
        elif self.association_mode == 'pixel_photon':
            analyser.associate_pixels_to_photons(verbosity=self.verbosity)
        elif self.association_mode == 'photon_event':
            analyser.associate_photons_to_events(verbosity=self.verbosity)

        # Save associations
        output_path = analyser.save_associations(
            output_dir=str(data_folder / "AssociatedResults"),
            filename="associated_data.csv",
            format='csv',
            verbosity=self.verbosity
        )

        if self.verbosity:
            print(f"Associations saved to: {output_path}")

        # Return the combined dataframe
        return analyser.get_combined_dataframe()
    
    def _normalize_columns(self, sensor_size=8):
        """
        Detect the format and normalize column names to x, y, toa, tof.
        Supports three formats:
        1. Original format: x, y, toa, tof
        2. Alternative format: x2, y2, z2, id, neutron_id, toa2, photon_count, time_diff
        3. AssociatedResults format: x, y, t, tot, tof, assoc_photon_id, assoc_event_id, etc.

        Parameters:
            sensor_size (float): Size of the sensor in mm (used for normalization).
        """
        columns = self.df.columns.tolist()

        # Check if this is the AssociatedResults format (has t, tot, and association columns)
        if 't' in columns and 'tot' in columns and any('assoc_' in col for col in columns):
            # AssociatedResults format from neutron_event_analyzer
            # Rename 't' to 'toa' for consistency
            self.df = self.df.rename(columns={'t': 'toa'})

            # Ensure tof exists (duplicate toa if not)
            if 'tof' not in self.df.columns:
                self.df['tof'] = self.df['toa']

            # Keep all columns (including association columns)
            # Filter out rows with invalid time
            self.df = self.df.loc[(self.df["toa"] >= 0)]
            self.df = self.df.sort_values(by="toa")

        # Check if this is the alternative format (has x2, y2, toa2)
        elif 'x2' in columns and 'y2' in columns and 'toa2' in columns:
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

            # Keep tot if it exists
            if 'tot' in columns:
                final_cols.append('tot')

            self.df = self.df[final_cols]

        else:
            raise ValueError(
                f"Unrecognized CSV format. Expected either "
                f"(x, y, toa, tof), (x2, y2, toa2, ...), or AssociatedResults format. "
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
                        self.h[key], _, _ = np.histogram2d(
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
                        self.h[tbin], _, _ = np.histogram2d(
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

    def plot(self, key=None, neutron_id_filter=None,
             photon_range=None, photon_ids=None,
             pixel_range=None, pixel_ids=None,
             event_range=None, event_ids=None,
             toa_range=None,
             color_by='toa',
             zoom_region=None, zoom_size=None,
             custom_color=None, show_background=False, despine=True,
             time_bins=None, show_scale=False, cmap=None,
             show_labels=True, show_legend=False, auto_zoom_margin=5):
        """
        Plot the pixel hit time development using the stored histograms.

        Parameters:
            key (int or tuple): Starting index for the time bins. If neutron_id is enabled,
                               can be tuple (time_index, neutron_id) or just time_index.
            neutron_id_filter (int): If provided, only plot data for this specific neutron.
            photon_range (tuple): (start_idx, end_idx) to filter by photon index range.
            photon_ids (list): List of specific photon IDs to include.
            pixel_range (tuple): (start_idx, end_idx) to filter by pixel index range.
            pixel_ids (list): List of specific pixel indices to include.
            event_range (tuple): (start_idx, end_idx) to filter by event index range.
            event_ids (list): List of specific event IDs to include.
            toa_range (tuple): (min_toa, max_toa) in seconds to filter by time of arrival.
            color_by (str): 'toa' or 'tot' - which parameter to use for colormap.
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

        # Apply filters to create a filtered dataframe
        filtered_df = self._apply_filters(
            photon_range=photon_range,
            photon_ids=photon_ids,
            pixel_range=pixel_range,
            pixel_ids=pixel_ids,
            event_range=event_range,
            event_ids=event_ids,
            toa_range=toa_range,
            neutron_id_filter=neutron_id_filter
        )

        # If filters were applied, we need to recompute histograms with filtered data
        if filtered_df is not None:
            if self.verbosity:
                print(f"Filtered data: {len(filtered_df)} rows (from {len(self.df)} original)")

            # Temporarily replace df and recompute histograms
            original_df = self.df
            self.df = filtered_df

            # Recompute histograms with filtered data
            bins = self.keys  # Use existing time bins
            if len(bins) > 0:
                # Extract time range from existing bins
                if isinstance(bins[0], tuple):
                    time_bins_array = sorted(set(k[0] for k in bins))
                else:
                    time_bins_array = sorted(bins)

                # Create proper bin edges
                if len(time_bins_array) > 1:
                    step = time_bins_array[1] - time_bins_array[0]
                    bins_edges = np.append(time_bins_array, time_bins_array[-1] + step)
                else:
                    bins_edges = np.array([time_bins_array[0], time_bins_array[0] + 1e-9])

                self.prepare_histograms(bins_edges)

            # Plot with filtered histograms
            self._do_plot(key, neutron_id_filter, zoom_region, zoom_size, custom_color,
                         show_background, despine, time_bins, show_scale, cmap,
                         show_labels, show_legend, auto_zoom_margin, color_by)

            # Restore original data
            self.df = original_df
            # Recompute histograms with original data
            self.prepare_histograms(bins_edges)
        else:
            # No filters, use existing histograms
            self._do_plot(key, neutron_id_filter, zoom_region, zoom_size, custom_color,
                         show_background, despine, time_bins, show_scale, cmap,
                         show_labels, show_legend, auto_zoom_margin, color_by)

    def _apply_filters(self, photon_range=None, photon_ids=None,
                      pixel_range=None, pixel_ids=None,
                      event_range=None, event_ids=None,
                      toa_range=None, neutron_id_filter=None):
        """
        Apply filters to the dataframe.

        Returns:
            pd.DataFrame or None: Filtered dataframe, or None if no filters applied.
        """
        df = self.df.copy()
        any_filter_applied = False

        # Filter by photon range or IDs
        if photon_range is not None and self.has_photon_id:
            # Get unique photon IDs and filter by index range
            unique_photons = df['assoc_photon_id'].dropna().unique()
            sorted_photons = sorted(unique_photons)
            start_idx, end_idx = photon_range
            selected_photons = sorted_photons[start_idx:end_idx]
            df = df[df['assoc_photon_id'].isin(selected_photons)]
            any_filter_applied = True

        if photon_ids is not None and self.has_photon_id:
            df = df[df['assoc_photon_id'].isin(photon_ids)]
            any_filter_applied = True

        # Filter by event range or IDs
        if event_range is not None and self.has_event_id:
            unique_events = df['assoc_event_id'].dropna().unique()
            sorted_events = sorted(unique_events)
            start_idx, end_idx = event_range
            selected_events = sorted_events[start_idx:end_idx]
            df = df[df['assoc_event_id'].isin(selected_events)]
            any_filter_applied = True

        if event_ids is not None and self.has_event_id:
            df = df[df['assoc_event_id'].isin(event_ids)]
            any_filter_applied = True

        # Filter by pixel range (index in dataframe)
        if pixel_range is not None:
            start_idx, end_idx = pixel_range
            df = df.iloc[start_idx:end_idx]
            any_filter_applied = True

        if pixel_ids is not None:
            df = df.iloc[pixel_ids]
            any_filter_applied = True

        # Filter by TOA range
        if toa_range is not None:
            min_toa, max_toa = toa_range
            df = df[(df['toa'] >= min_toa) & (df['toa'] <= max_toa)]
            any_filter_applied = True

        # Filter by neutron ID (handled separately in _do_plot)
        # We don't filter here to maintain compatibility with existing code

        return df if any_filter_applied else None

    def _do_plot(self, key, neutron_id_filter, zoom_region, zoom_size, custom_color,
                show_background, despine, time_bins, show_scale, cmap,
                show_labels, show_legend, auto_zoom_margin, color_by):
        """
        Internal method to perform the actual plotting.
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
                                    show_labels, show_legend, auto_zoom_margin, color_by, self.df)
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
                                        show_labels, show_legend, auto_zoom_margin, color_by, self.df)
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
                                            show_labels, show_legend, auto_zoom_margin, color_by, self.df)
                    else:
                        # Shouldn't happen but handle gracefully
                        plot_time_development(self.h, self.keys, key,
                                            zoom_region, zoom_size, custom_color, show_background,
                                            despine, time_bins, show_scale, cmap,
                                            show_labels, show_legend, auto_zoom_margin, color_by, self.df)
        else:
            # Standard plotting without neutron grouping
            if isinstance(key, tuple):
                raise ValueError("Composite keys only valid when neutron_id=True")
            
            start_key_idx = key
            
            plot_time_development(self.h, self.keys, start_key_idx,
                                zoom_region, zoom_size, custom_color, show_background,
                                despine, time_bins, show_scale, cmap,
                                show_labels, show_legend, auto_zoom_margin, color_by, self.df)