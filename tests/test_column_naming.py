"""
Tests for column naming conventions and data loading.

Tests the following features:
1. Old column naming (assoc_photon_id, assoc_event_id, tot)
2. New column naming (ph_id, ev_id, px_tot)
3. Multiple CSV file loading and concatenation
4. Wildcard file loading
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from visualize_pixel_map import Data


class TestColumnNaming:
    """Test both old and new column naming conventions."""

    def create_old_format_csv(self, filepath):
        """Create a CSV file with old column naming."""
        data = {
            'x': np.random.randint(0, 256, 100),
            'y': np.random.randint(0, 256, 100),
            't': np.random.uniform(0, 0.01, 100),
            'tot': np.random.randint(10, 100, 100),
            'tof': np.random.uniform(0, 0.01, 100),
            'assoc_photon_id': np.random.randint(0, 10, 100),
            'assoc_event_id': np.random.randint(0, 5, 100)
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return df

    def create_new_format_csv(self, filepath, separator='_'):
        """Create a CSV file with new column naming (prefix-based).

        Args:
            filepath: Path to save CSV file
            separator: Either '_' for underscore, '\\' for backslash, or '/' for forward slash
        """
        if separator == '\\':
            data = {
                'px\\x': np.random.randint(0, 256, 100),
                'px\\y': np.random.randint(0, 256, 100),
                'px\\toa': np.random.uniform(0, 0.01, 100),
                'px\\tot': np.random.randint(10, 100, 100),
                'px\\tof': np.random.uniform(0, 0.01, 100),
                'ph\\id': np.random.randint(0, 10, 100),
                'ev\\id': np.random.randint(0, 5, 100)
            }
        elif separator == '/':
            data = {
                'px/x': np.random.randint(0, 256, 100),
                'px/y': np.random.randint(0, 256, 100),
                'px/toa': np.random.uniform(0, 0.01, 100),
                'px/tot': np.random.randint(10, 100, 100),
                'px/tof': np.random.uniform(0, 0.01, 100),
                'ph/id': np.random.randint(0, 10, 100),
                'ev/id': np.random.randint(0, 5, 100)
            }
        else:
            data = {
                'px_x': np.random.randint(0, 256, 100),
                'px_y': np.random.randint(0, 256, 100),
                'px_toa': np.random.uniform(0, 0.01, 100),
                'px_tot': np.random.randint(10, 100, 100),
                'px_tof': np.random.uniform(0, 0.01, 100),
                'ph_id': np.random.randint(0, 10, 100),
                'ev_id': np.random.randint(0, 5, 100)
            }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return df

    def test_old_column_naming(self):
        """Test that old column naming is correctly detected and normalized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data_old.csv"
            self.create_old_format_csv(csv_path)

            # Load data
            data = Data(str(csv_path), verbosity=0)

            # Check that columns were normalized
            assert 'x' in data.df.columns
            assert 'y' in data.df.columns
            assert 'toa' in data.df.columns  # t should be renamed to toa
            assert 'tot' in data.df.columns

            # Check that association columns are detected
            assert data.has_photon_id
            assert data.has_event_id
            assert data.has_tot

            # Check that properties work
            assert data.photons is not None
            assert data.events is not None
            assert data.photons['count'] > 0
            assert data.events['count'] > 0

    def test_new_column_naming(self):
        """Test that new column naming (prefix-based with underscore) is correctly detected and preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data_new.csv"
            self.create_new_format_csv(csv_path, separator='_')

            # Load data
            data = Data(str(csv_path), verbosity=0)

            # Check that px_* columns are PRESERVED (not renamed)
            assert 'px_x' in data.df.columns
            assert 'px_y' in data.df.columns
            assert 'px_toa' in data.df.columns
            assert 'px_tot' in data.df.columns

            # Check that ph_id and ev_id are present
            assert 'ph_id' in data.df.columns or 'assoc_photon_id' in data.df.columns
            assert 'ev_id' in data.df.columns or 'assoc_event_id' in data.df.columns

            # Check that association columns are detected
            assert data.has_photon_id
            assert data.has_event_id
            assert data.has_tot

            # Check that properties work with new naming
            assert data.photons is not None
            assert data.events is not None
            assert data.photons['count'] > 0
            assert data.events['count'] > 0

    def test_new_column_naming_backslash(self):
        """Test that new column naming (prefix-based with backslash) is correctly detected and preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data_new_backslash.csv"
            self.create_new_format_csv(csv_path, separator='\\')

            # Load data
            data = Data(str(csv_path), verbosity=0)

            # Check that px\* columns are PRESERVED (not renamed)
            assert 'px\\x' in data.df.columns
            assert 'px\\y' in data.df.columns
            assert 'px\\toa' in data.df.columns
            assert 'px\\tot' in data.df.columns

            # Check that ph\id and ev\id are present
            assert 'ph\\id' in data.df.columns or 'assoc_photon_id' in data.df.columns
            assert 'ev\\id' in data.df.columns or 'assoc_event_id' in data.df.columns

            # Check that association columns are detected
            assert data.has_photon_id
            assert data.has_event_id
            assert data.has_tot

            # Check that properties work with new naming
            assert data.photons is not None
            assert data.events is not None
            assert data.photons['count'] > 0
            assert data.events['count'] > 0

    def test_new_column_naming_forward_slash(self):
        """Test that new column naming (prefix-based with forward slash) is correctly detected and preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data_new_forward_slash.csv"
            self.create_new_format_csv(csv_path, separator='/')

            # Load data
            data = Data(str(csv_path), verbosity=0)

            # Check that px/* columns are PRESERVED (not renamed)
            assert 'px/x' in data.df.columns
            assert 'px/y' in data.df.columns
            assert 'px/toa' in data.df.columns
            assert 'px/tot' in data.df.columns

            # Check that ph/id and ev/id are present
            assert 'ph/id' in data.df.columns or 'assoc_photon_id' in data.df.columns
            assert 'ev/id' in data.df.columns or 'assoc_event_id' in data.df.columns

            # Check that association columns are detected
            assert data.has_photon_id
            assert data.has_event_id
            assert data.has_tot

            # Check that properties work with new naming
            assert data.photons is not None
            assert data.events is not None
            assert data.photons['count'] > 0
            assert data.events['count'] > 0

    def test_get_column_name_helper(self):
        """Test the _get_column_name helper method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with old naming
            csv_old = Path(tmpdir) / "old.csv"
            self.create_old_format_csv(csv_old)
            data_old = Data(str(csv_old), verbosity=0)

            # Should find old column names
            assert data_old._get_column_name('photon_id') in ['assoc_photon_id', 'ph_id', 'ph\\id']
            assert data_old._get_column_name('event_id') in ['assoc_event_id', 'ev_id', 'ev\\id']
            assert data_old._get_column_name('tot') in ['tot', 'px_tot', 'px\\tot']

            # Test with new naming (underscore)
            csv_new = Path(tmpdir) / "new.csv"
            self.create_new_format_csv(csv_new, separator='_')
            data_new = Data(str(csv_new), verbosity=0)

            # Should find new column names
            assert data_new._get_column_name('photon_id') in ['ph_id', 'assoc_photon_id', 'ph\\id']
            assert data_new._get_column_name('event_id') in ['ev_id', 'assoc_event_id', 'ev\\id']
            assert data_new._get_column_name('tot') in ['tot', 'px_tot', 'px\\tot']

            # Test with new naming (backslash)
            csv_backslash = Path(tmpdir) / "backslash.csv"
            self.create_new_format_csv(csv_backslash, separator='\\')
            data_backslash = Data(str(csv_backslash), verbosity=0)

            # Should find backslash column names
            assert data_backslash._get_column_name('photon_id') in ['ph\\id', 'ph_id', 'assoc_photon_id']
            assert data_backslash._get_column_name('event_id') in ['ev\\id', 'ev_id', 'assoc_event_id']
            assert data_backslash._get_column_name('tot') in ['tot', 'px\\tot', 'px_tot']

            # Test with new naming (forward slash)
            csv_forward = Path(tmpdir) / "forward_slash.csv"
            self.create_new_format_csv(csv_forward, separator='/')
            data_forward = Data(str(csv_forward), verbosity=0)

            # Should find forward slash column names
            assert data_forward._get_column_name('photon_id') in ['ph/id', 'ph\\id', 'ph_id', 'assoc_photon_id']
            assert data_forward._get_column_name('event_id') in ['ev/id', 'ev\\id', 'ev_id', 'assoc_event_id']
            assert data_forward._get_column_name('tot') in ['px/tot', 'tot', 'px\\tot', 'px_tot']


class TestMultipleCSVLoading:
    """Test loading and concatenating multiple CSV files."""

    def create_multiple_csvs(self, folder_path, num_files=3, format_type='old'):
        """Create multiple CSV files in AssociatedResults folder."""
        assoc_dir = folder_path / "AssociatedResults"
        assoc_dir.mkdir(exist_ok=True)

        for i in range(num_files):
            csv_path = assoc_dir / f"data_{i}.csv"
            if format_type == 'old':
                data = {
                    'x': np.random.randint(0, 256, 50),
                    'y': np.random.randint(0, 256, 50),
                    't': np.random.uniform(i * 0.01, (i + 1) * 0.01, 50),
                    'tot': np.random.randint(10, 100, 50),
                    'tof': np.random.uniform(i * 0.01, (i + 1) * 0.01, 50),
                    'assoc_photon_id': np.random.randint(i * 10, (i + 1) * 10, 50),
                    'assoc_event_id': np.random.randint(i * 5, (i + 1) * 5, 50)
                }
            else:  # new format
                data = {
                    'px_x': np.random.randint(0, 256, 50),
                    'px_y': np.random.randint(0, 256, 50),
                    'px_toa': np.random.uniform(i * 0.01, (i + 1) * 0.01, 50),
                    'px_tot': np.random.randint(10, 100, 50),
                    'px_tof': np.random.uniform(i * 0.01, (i + 1) * 0.01, 50),
                    'ph_id': np.random.randint(i * 10, (i + 1) * 10, 50),
                    'ev_id': np.random.randint(i * 5, (i + 1) * 5, 50)
                }

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

    def test_load_all_csvs(self):
        """Test loading all CSV files from AssociatedResults folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "neutrons"
            folder.mkdir()
            self.create_multiple_csvs(folder, num_files=3)

            # Load data - should concatenate all 3 files
            data = Data(str(folder), verbosity=0)

            # Should have 3 * 50 = 150 rows
            assert len(data.df) == 150

            # Check columns are normalized
            assert 'x' in data.df.columns
            assert 'y' in data.df.columns
            assert 'toa' in data.df.columns

    def test_wildcard_loading(self):
        """Test loading specific files with wildcards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "neutrons"
            folder.mkdir()
            self.create_multiple_csvs(folder, num_files=5)

            # Load only data_0.csv and data_1.csv using wildcard
            wildcard_path = str(folder / "AssociatedResults" / "data_[01].csv")
            data = Data(wildcard_path, verbosity=0)

            # Should have 2 * 50 = 100 rows
            assert len(data.df) == 100

    def test_wildcard_with_asterisk(self):
        """Test loading files with asterisk wildcard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "neutrons"
            folder.mkdir()
            self.create_multiple_csvs(folder, num_files=3)

            # Load all CSV files using * wildcard
            wildcard_path = str(folder / "AssociatedResults" / "data_*.csv")
            data = Data(wildcard_path, verbosity=0)

            # Should have 3 * 50 = 150 rows
            assert len(data.df) == 150


class TestFilteringWithNewColumns:
    """Test that filtering works with both old and new column naming."""

    def test_filter_photons_old_naming(self):
        """Test filtering by photons with old column naming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            data = {
                'x': np.random.randint(0, 256, 100),
                'y': np.random.randint(0, 256, 100),
                't': np.random.uniform(0, 0.01, 100),
                'tot': np.random.randint(10, 100, 100),
                'tof': np.random.uniform(0, 0.01, 100),
                'assoc_photon_id': [i // 10 for i in range(100)],  # 10 photons
                'assoc_event_id': [i // 20 for i in range(100)]    # 5 events
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            data_obj = Data(str(csv_path), verbosity=0)

            # Filter by photons
            filtered_df = data_obj._apply_filters(photons=(0, 3), verbosity=0)

            # Should only have data from first 3 photons
            photon_col = data_obj._get_column_name('photon_id')
            unique_photons = filtered_df[photon_col].unique()
            assert len(unique_photons) == 3

    def test_filter_photons_new_naming(self):
        """Test filtering by photons with new column naming (underscore)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            data = {
                'px_x': np.random.randint(0, 256, 100),
                'px_y': np.random.randint(0, 256, 100),
                'px_toa': np.random.uniform(0, 0.01, 100),
                'px_tot': np.random.randint(10, 100, 100),
                'px_tof': np.random.uniform(0, 0.01, 100),
                'ph_id': [i // 10 for i in range(100)],  # 10 photons
                'ev_id': [i // 20 for i in range(100)]   # 5 events
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            data_obj = Data(str(csv_path), verbosity=0)

            # Filter by photons
            filtered_df = data_obj._apply_filters(photons=(0, 3), verbosity=0)

            # Should only have data from first 3 photons
            photon_col = data_obj._get_column_name('photon_id')
            unique_photons = filtered_df[photon_col].unique()
            assert len(unique_photons) == 3

    def test_filter_photons_new_naming_backslash(self):
        """Test filtering by photons with new column naming (backslash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            data = {
                'px\\x': np.random.randint(0, 256, 100),
                'px\\y': np.random.randint(0, 256, 100),
                'px\\toa': np.random.uniform(0, 0.01, 100),
                'px\\tot': np.random.randint(10, 100, 100),
                'px\\tof': np.random.uniform(0, 0.01, 100),
                'ph\\id': [i // 10 for i in range(100)],  # 10 photons
                'ev\\id': [i // 20 for i in range(100)]   # 5 events
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            data_obj = Data(str(csv_path), verbosity=0)

            # Filter by photons
            filtered_df = data_obj._apply_filters(photons=(0, 3), verbosity=0)

            # Should only have data from first 3 photons
            photon_col = data_obj._get_column_name('photon_id')
            unique_photons = filtered_df[photon_col].unique()
            assert len(unique_photons) == 3

    def test_filter_photons_new_naming_forward_slash(self):
        """Test filtering by photons with new column naming (forward slash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            data = {
                'px/x': np.random.randint(0, 256, 100),
                'px/y': np.random.randint(0, 256, 100),
                'px/toa': np.random.uniform(0, 0.01, 100),
                'px/tot': np.random.randint(10, 100, 100),
                'px/tof': np.random.uniform(0, 0.01, 100),
                'ph/id': [i // 10 for i in range(100)],  # 10 photons
                'ev/id': [i // 20 for i in range(100)]   # 5 events
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            data_obj = Data(str(csv_path), verbosity=0)

            # Filter by photons
            filtered_df = data_obj._apply_filters(photons=(0, 3), verbosity=0)

            # Should only have data from first 3 photons
            photon_col = data_obj._get_column_name('photon_id')
            unique_photons = filtered_df[photon_col].unique()
            assert len(unique_photons) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
