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
            verbosity (int): 0=QUIET (no output), 1=BASIC (progress bars), 2=ADVANCED (detailed info).
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

        # Store default binning parameters
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step

        # Detect if input is file, folder, or wildcard pattern
        path = Path(data_path)

        # Check for wildcard patterns first (before file/dir checks)
        if '*' in str(data_path) or '[' in str(data_path):
            # Wildcard pattern - handle in _load_from_folder
            self.data_source = 'wildcard'
            self.df = self._load_from_folder(path)
        elif path.is_file():
            # Legacy behavior: load CSV file directly
            if self.verbosity >= 1:
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
        # Support both old and new column naming conventions
        # New format uses backslash: ph\id, ev\id, px\tot
        # Or underscore: ph_id, ev_id, px_tot
        self.has_photon_id = ('ph\\id' in self.df.columns or 'ph_id' in self.df.columns or 'assoc_photon_id' in self.df.columns)
        self.has_event_id = ('ev\\id' in self.df.columns or 'ev_id' in self.df.columns or 'assoc_event_id' in self.df.columns)
        self.has_tot = ('px\\tot' in self.df.columns or 'px_tot' in self.df.columns or 'tot' in self.df.columns)

        if self.verbosity >= 1:
            print(f"Loaded {len(self.df)} rows")
            if self.verbosity >= 2:
                if self.has_photon_id:
                    print(f"  - Contains photon associations")
                if self.has_event_id:
                    print(f"  - Contains event associations")

        self._normalize_columns(sensor_size)

    def _load_from_folder(self, folder_path):
        """
        Load data from a neutron data folder.
        Auto-detects AssociatedResults or runs neutron_event_analyzer if needed.
        Supports wildcards for selecting specific files.

        Parameters:
            folder_path (Path): Path to the neutron data folder or specific CSV file(s) with wildcards.

        Returns:
            pd.DataFrame: Loaded data (concatenated if multiple files).
        """
        folder_path = Path(folder_path)

        # Check if path contains wildcards
        if '*' in str(folder_path) or '[' in str(folder_path):
            # User specified wildcard pattern
            # Get the parent directory and the pattern
            if folder_path.is_absolute():
                # For absolute paths, glob from the parent
                parent = folder_path.parent
                pattern = folder_path.name
                csv_files = list(parent.glob(pattern))
            else:
                # For relative paths, glob from current directory
                csv_files = list(Path('.').glob(str(folder_path)))

            if not csv_files:
                raise FileNotFoundError(f"No files match pattern: {folder_path}")

            if self.verbosity >= 1:
                print(f"Found {len(csv_files)} file(s) matching pattern")

            # Load and concatenate all matching files
            dfs = []
            for csv_file in sorted(csv_files):
                if self.verbosity >= 2:
                    print(f"  Loading: {csv_file}")
                dfs.append(pd.read_csv(csv_file))

            return pd.concat(dfs, ignore_index=True)

        # Check for AssociatedResults folder
        assoc_results_dir = folder_path / "AssociatedResults"

        if assoc_results_dir.exists() and assoc_results_dir.is_dir():
            # Look for CSV files in AssociatedResults
            csv_files = list(assoc_results_dir.glob("*.csv"))

            if csv_files:
                if self.verbosity >= 1:
                    print(f"Found AssociatedResults folder with {len(csv_files)} CSV file(s)")

                # Load and concatenate ALL CSV files
                dfs = []
                for csv_file in sorted(csv_files):
                    if self.verbosity >= 2:
                        print(f"  Loading: {csv_file.name}")
                    dfs.append(pd.read_csv(csv_file))

                if self.verbosity >= 1:
                    print(f"Concatenating {len(dfs)} CSV file(s)")

                return pd.concat(dfs, ignore_index=True)

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
        Supports five formats:
        1. Original format: x, y, toa, tof
        2. Alternative format: x2, y2, z2, id, neutron_id, toa2, photon_count, time_diff
        3. AssociatedResults format (old): x, y, t, tot, tof, assoc_photon_id, assoc_event_id, etc.
        4. AssociatedResults format (new underscore): px_x, px_y, px_toa, px_tot, ph_id, ev_id, etc.
        5. AssociatedResults format (new backslash): px\x, px\y, px\toa, px\tot, ph\id, ev\id, etc.

        Parameters:
            sensor_size (float): Size of the sensor in mm (used for normalization).
        """
        columns = self.df.columns.tolist()

        # Check for NEW AssociatedResults format with backslash separator (px\*, ph\*, ev\*)
        if any(col.startswith('px\\') for col in columns):
            # New format with backslash - rename px\* columns to expected names
            rename_map = {}

            # Pixel columns
            if 'px\\x' in columns:
                rename_map['px\\x'] = 'x'
            if 'px\\y' in columns:
                rename_map['px\\y'] = 'y'
            if 'px\\toa' in columns:
                rename_map['px\\toa'] = 'toa'
            if 'px\\tot' in columns:
                rename_map['px\\tot'] = 'tot'
            if 'px\\tof' in columns:
                rename_map['px\\tof'] = 'tof'

            # Photon and event columns - keep ph\id and ev\id as is
            # (they'll be detected separately)

            self.df = self.df.rename(columns=rename_map)

            # Ensure tof exists (duplicate toa if not)
            if 'tof' not in self.df.columns and 'toa' in self.df.columns:
                self.df['tof'] = self.df['toa']

            # Filter out rows with invalid time
            self.df = self.df.loc[(self.df["toa"] >= 0)]
            self.df = self.df.sort_values(by="toa")

        # Check for NEW AssociatedResults format with underscore separator (px_*, ph_*, ev_*)
        elif any(col.startswith('px_') for col in columns):
            # New format with underscore - rename px_* columns to expected names
            rename_map = {}

            # Pixel columns
            if 'px_x' in columns:
                rename_map['px_x'] = 'x'
            if 'px_y' in columns:
                rename_map['px_y'] = 'y'
            if 'px_toa' in columns:
                rename_map['px_toa'] = 'toa'
            if 'px_tot' in columns:
                rename_map['px_tot'] = 'tot'
            if 'px_tof' in columns:
                rename_map['px_tof'] = 'tof'

            # Photon and event columns - keep ph_id and ev_id as is
            # (they'll be detected separately)

            self.df = self.df.rename(columns=rename_map)

            # Ensure tof exists (duplicate toa if not)
            if 'tof' not in self.df.columns and 'toa' in self.df.columns:
                self.df['tof'] = self.df['toa']

            # Filter out rows with invalid time
            self.df = self.df.loc[(self.df["toa"] >= 0)]
            self.df = self.df.sort_values(by="toa")

        # Check if this is the OLD AssociatedResults format (has t, tot, and association columns)
        elif 't' in columns and 'tot' in columns and any('assoc_' in col for col in columns):
            # AssociatedResults format from neutron_event_analyzer (old naming)
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
    def _get_column_name(self, column_type):
        """
        Get the actual column name for a given type, handling both old and new naming.

        Parameters:
            column_type (str): One of 'photon_id', 'event_id', 'tot'

        Returns:
            str or None: The actual column name in the dataframe, or None if not found.
        """
        mapping = {
            'photon_id': ['ph\\id', 'ph_id', 'assoc_photon_id'],
            'event_id': ['ev\\id', 'ev_id', 'assoc_event_id'],
            'tot': ['px\\tot', 'px_tot', 'tot']
        }

        for col in mapping.get(column_type, []):
            if col in self.df.columns:
                return col
        return None

    # Properties for easy data inspection
    @property
    def associated_df(self):
        """Access the underlying dataframe with all data including associations."""
        return self.df

    @property
    def photons(self):
        """Get information about available photons."""
        if not self.has_photon_id:
            return None

        photon_col = self._get_column_name('photon_id')
        unique_photons = self.df[photon_col].dropna().unique()
        return {
            'count': len(unique_photons),
            'ids': sorted(unique_photons.tolist()),
            'range': (unique_photons.min(), unique_photons.max()) if len(unique_photons) > 0 else (None, None)
        }

    @property
    def events(self):
        """Get information about available events."""
        if not self.has_event_id:
            return None

        event_col = self._get_column_name('event_id')
        unique_events = self.df[event_col].dropna().unique()
        return {
            'count': len(unique_events),
            'ids': sorted(unique_events.tolist()),
            'range': (unique_events.min(), unique_events.max()) if len(unique_events) > 0 else (None, None)
        }

    @property
    def pixels(self):
        """Get information about available pixel hits."""
        return {
            'count': len(self.df),
            'range': (0, len(self.df) - 1)
        }

    @property
    def times(self):
        """Get information about time range in the data."""
        if 'toa' not in self.df.columns or len(self.df) == 0:
            return None

        # Calculate expected number of bins
        time_step_sec = self.time_step * 1e-9
        expected_bins = int((self.end_time - self.start_time) / time_step_sec)

        return {
            'range': (self.df['toa'].min(), self.df['toa'].max()),
            'default_bins': expected_bins,
            'bin_width_ns': self.time_step
        }

    @property
    def neutrons(self):
        """Get information about available neutrons."""
        if not self.has_neutron_id:
            return None
        unique_neutrons = self.df['neutron_id'].dropna().unique()
        return {
            'count': len(unique_neutrons),
            'ids': sorted(unique_neutrons.tolist())
        }

    def info(self):
        """Print summary information about the data."""
        print("=" * 60)
        print("Data Summary")
        print("=" * 60)
        print(f"Total pixel hits: {len(self.df)}")
        print(f"Data source: {self.data_source}")

        if self.times:
            print(f"\nTime range: {self.times['range'][0]:.6f} - {self.times['range'][1]:.6f} s")
            print(f"Time bins: {self.times['bins']}")

        if self.photons:
            print(f"\nPhotons: {self.photons['count']}")
            print(f"  ID range: {self.photons['range'][0]} - {self.photons['range'][1]}")

        if self.events:
            print(f"\nEvents: {self.events['count']}")
            print(f"  ID range: {self.events['range'][0]} - {self.events['range'][1]}")

        if self.neutrons:
            print(f"\nNeutrons: {self.neutrons['count']}")
            print(f"  IDs: {self.neutrons['ids'][:10]}" + ("..." if self.neutrons['count'] > 10 else ""))

        print(f"\nAvailable columns: {', '.join(self.df.columns.tolist())}")
        print("=" * 60)

    def plot(self, key=None, neutron_id_filter=None,
             photons=None, pixels=None, events=None,
             toa_range=None, query=None,
             color_by='toa',
             zoom_region=None, zoom_size=None,
             custom_color=None, show_background=False, despine=True,
             time_bins=None, show_scale=False, cmap=None,
             show_labels=True, show_legend=False, auto_zoom_margin=5,
             verbosity=None):
        """
        Plot the pixel hit time development by filtering data, then creating histograms.

        Parameters:
            key (int): Starting index for the time bins (default: 0).
            neutron_id_filter (int): If provided, only plot data for this specific neutron.
            photons: Filter by photons. Can be:
                    - tuple (start, end): index range
                    - list [5, 10, 15]: specific IDs
                    - int 5: single ID
                    - slice(0, 10): slice object
            pixels: Filter by pixel hits. Can be:
                   - tuple (start, end): index range
                   - list [100, 200]: specific indices
                   - int 100: single index
                   - slice(0, 1000): slice object
            events: Filter by events. Can be:
                   - tuple (start, end): index range
                   - list [1, 2, 3]: specific IDs
                   - int 1: single ID
                   - slice(0, 5): slice object
            toa_range (tuple): (min_toa, max_toa) in seconds to filter by time of arrival.
            query (str): Pandas query string to filter data (e.g., "tot > 50 & x < 128").
            color_by (str): 'toa' or 'tot' - which parameter to use for colormap.
            zoom_region (tuple): (x_min, x_max, y_min, y_max) for zoom region.
            zoom_size (int): If provided, creates a square zoom region of this size around CoG.
            custom_color (str): Custom color map for plotting (deprecated, use cmap).
            show_background (bool): Whether to show background pixels.
            despine (bool): Whether to remove plot spines.
            time_bins: maximum time in nanoseconds (e.g., 40 for [10, 20, 30, 40]).
            show_scale (bool): Whether to show scale on the plot.
            cmap (str): Colormap to use for the plot.
            show_labels (bool): Whether to display timestamp text labels on pixels.
            show_legend (bool): Whether to display a legend with timestamp colors.
            auto_zoom_margin (int): Margin in pixels around data when auto-zooming.
            verbosity (int): Override instance verbosity. 0=QUIET, 1=BASIC, 2=ADVANCED.
        """
        from visualize_pixel_map.visualize import plot_time_development

        # Use provided verbosity or instance verbosity
        verb = verbosity if verbosity is not None else self.verbosity

        # Step 1: Apply filters to dataframe
        filtered_df = self._apply_filters(
            photons=photons,
            pixels=pixels,
            events=events,
            toa_range=toa_range,
            neutron_id_filter=neutron_id_filter,
            query=query,
            verbosity=verb
        )

        # Use filtered data or original if no filters applied
        plot_df = filtered_df if filtered_df is not None else self.df

        if len(plot_df) == 0:
            raise ValueError(
                "No pixel hits to plot after filtering. "
                "Check data.info() for available ranges or adjust your filters."
            )

        # Step 2: Create histograms from filtered data
        time_step_sec = self.time_step * 1e-9

        # Determine time range for histograms
        if toa_range is not None:
            hist_start, hist_end = toa_range
        else:
            hist_start = max(self.start_time, plot_df['toa'].min())
            hist_end = min(self.end_time, plot_df['toa'].max())

        bins = np.arange(hist_start, hist_end + time_step_sec, time_step_sec)

        if len(bins) < 2:
            raise ValueError(
                f"Not enough time bins for the data. "
                f"Data TOA range: {plot_df['toa'].min():.6f} - {plot_df['toa'].max():.6f}s"
            )

        h, keys = self._create_histograms(plot_df, bins, verbosity=verb)

        # Step 3: Plot
        if key is None:
            key = 0

        if key >= len(keys):
            raise ValueError(
                f"key={key} is out of range. Valid range: 0-{len(keys)-1}. "
                f"Total time bins: {len(keys)}"
            )

        return plot_time_development(
            h, keys, key,
            zoom_region, zoom_size, custom_color, show_background,
            despine, time_bins, show_scale, cmap,
            show_labels, show_legend, auto_zoom_margin, color_by, plot_df
        )

    def _apply_filters(self, photons=None, pixels=None, events=None,
                      toa_range=None, neutron_id_filter=None, query=None, verbosity=1):
        """
        Apply filters to the dataframe.

        Returns:
            pd.DataFrame or None: Filtered dataframe, or None if no filters applied.
        """
        df = self.df.copy()
        any_filter_applied = False
        original_count = len(df)

        # Helper function to parse flexible filter parameter
        def parse_filter(param):
            """Parse filter parameter into (use_range, use_ids) tuple."""
            if param is None:
                return None, None
            elif isinstance(param, slice):
                # Convert slice to tuple range
                return (param.start or 0, param.stop), None
            elif isinstance(param, tuple) and len(param) == 2:
                # Tuple is a range
                return param, None
            elif isinstance(param, (list, np.ndarray)):
                # List/array is specific IDs
                return None, param
            elif isinstance(param, (int, np.integer)):
                # Single integer is a single ID
                return None, [param]
            else:
                raise ValueError(f"Invalid filter parameter: {param}. "
                               "Expected tuple, list, int, or slice.")

        # Filter by photons
        if photons is not None and self.has_photon_id:
            photon_col = self._get_column_name('photon_id')
            photon_range, photon_ids = parse_filter(photons)

            if photon_range is not None:
                # Get unique photon IDs and filter by index range
                unique_photons = df[photon_col].dropna().unique()
                sorted_photons = sorted(unique_photons)
                start_idx, end_idx = photon_range
                selected_photons = sorted_photons[start_idx:end_idx]
                df = df[df[photon_col].isin(selected_photons)]
                any_filter_applied = True

            if photon_ids is not None:
                df = df[df[photon_col].isin(photon_ids)]
                any_filter_applied = True

        # Filter by events
        if events is not None and self.has_event_id:
            event_col = self._get_column_name('event_id')
            event_range, event_ids = parse_filter(events)

            if event_range is not None:
                unique_events = df[event_col].dropna().unique()
                sorted_events = sorted(unique_events)
                start_idx, end_idx = event_range
                selected_events = sorted_events[start_idx:end_idx]
                df = df[df[event_col].isin(selected_events)]
                any_filter_applied = True

            if event_ids is not None:
                df = df[df[event_col].isin(event_ids)]
                any_filter_applied = True

        # Filter by pixels (index in dataframe)
        if pixels is not None:
            pixel_range, pixel_ids = parse_filter(pixels)

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
            if verbosity >= 2:
                print(f"  - TOA range filter: {len(df)} rows remain")

        # Filter by neutron ID
        if neutron_id_filter is not None and self.has_neutron_id:
            df = df[df['neutron_id'] == neutron_id_filter]
            any_filter_applied = True
            if verbosity >= 2:
                print(f"  - Neutron ID filter: {len(df)} rows remain")

        # Apply pandas query string
        if query is not None:
            try:
                df = df.query(query)
                any_filter_applied = True
                if verbosity >= 2:
                    print(f"  - Query '{query}': {len(df)} rows remain")
            except Exception as e:
                raise ValueError(f"Invalid query string '{query}': {e}")

        # Print summary if filters were applied
        if any_filter_applied and verbosity >= 1:
            print(f"Filtered: {len(df):,} rows (from {original_count:,}, {len(df)/original_count*100:.1f}%)")

        return df if any_filter_applied else None

    def _create_histograms(self, df, bins, verbosity=1):
        """
        Create 2D histograms from filtered dataframe.

        Parameters:
            df (pd.DataFrame): Filtered dataframe to create histograms from.
            bins (np.ndarray): Time bin edges.
            verbosity (int): Verbosity level.

        Returns:
            tuple: (h, keys) where h is dict of histograms and keys is list of time bin values.
        """
        from tqdm.notebook import tqdm

        h = {}

        # Create time bins
        df_copy = df.copy()
        df_copy["tbin"] = pd.cut(df_copy["toa"], bins, labels=bins[:-1], include_lowest=True)
        df_binned = df_copy.query("toa < @bins[-1]")

        if len(df_binned) == 0:
            if verbosity >= 1:
                print("Warning: No data in specified time range")
            return {}, []

        unique_bins = df_binned["tbin"].dropna().unique()

        # Show progress bar only for verbosity >= 1
        if verbosity >= 1:
            iterator = tqdm(unique_bins, desc="Creating histograms")
        else:
            iterator = unique_bins

        for tbin in iterator:
            subset = df_binned[df_binned["tbin"] == tbin]
            if not subset.empty:
                h[tbin], _, _ = np.histogram2d(
                    subset["x"], subset["y"],
                    bins=[np.arange(257), np.arange(257)]  # 0-256 inclusive
                )

        keys = sorted(h.keys())

        if verbosity >= 2:
            print(f"Created {len(keys)} histograms covering {len(df_binned)} pixels")

        return h, keys
