# Column Naming Update - v2.0 Complete

## Summary

Successfully updated visualize_pixel_map to support the new column naming convention from neutron_event_analyzer while maintaining backward compatibility with the old naming convention.

## Changes Made

### 1. Column Naming Support (`analyse.py`)

#### Added Support for Both Naming Conventions

**Old Format (v1.x):**
- `assoc_photon_id`, `assoc_event_id`
- `tot` (time over threshold)
- `t` or `toa` (time of arrival)

**New Format (v2.0+):**
- `ph_id`, `ev_id` (photon and event IDs)
- `px_tot`, `px_toa`, `px_tof`, `px_x`, `px_y` (pixel columns with `px_*` prefix)
- Additional prefixes: `ph_*` (photon), `ev_*` (event)

#### Implementation Details

1. **Updated `__init__` method (lines 59-62)**:
   ```python
   # Support both old and new column naming conventions
   self.has_photon_id = ('ph_id' in self.df.columns or 'assoc_photon_id' in self.df.columns)
   self.has_event_id = ('ev_id' in self.df.columns or 'assoc_event_id' in self.df.columns)
   self.has_tot = ('px_tot' in self.df.columns or 'tot' in self.df.columns)
   ```

2. **Added `_get_column_name()` helper method (lines 344-363)**:
   ```python
   def _get_column_name(self, column_type):
       """Get the actual column name for a given type, handling both old and new naming."""
       mapping = {
           'photon_id': ['ph_id', 'assoc_photon_id'],
           'event_id': ['ev_id', 'assoc_event_id'],
           'tot': ['px_tot', 'tot']
       }
       for col in mapping.get(column_type, []):
           if col in self.df.columns:
               return col
       return None
   ```

3. **Updated `_normalize_columns()` method (lines 233-290)**:
   - Added detection for new format with `px_*` prefix
   - Renamed `px_x` → `x`, `px_y` → `y`, `px_toa` → `toa`, `px_tot` → `tot`, `px_tof` → `tof`
   - Kept `ph_id` and `ev_id` as-is (no renaming needed)

4. **Updated properties to use helper method**:
   - `photons` property (lines 372-383)
   - `events` property (lines 385-397)

   Both properties now use `self._get_column_name()` to get the correct column name.

5. **Updated `_apply_filters()` method (lines 602-635)**:
   - Uses `self._get_column_name('photon_id')` to get correct column
   - Uses `self._get_column_name('event_id')` to get correct column
   - Works with both old and new naming conventions

### 2. Wildcard File Loading

#### Updated `__init__` method (lines 41-60)
Added wildcard detection before file/directory checks:
```python
# Check for wildcard patterns first (before file/dir checks)
if '*' in str(data_path) or '[' in str(data_path):
    # Wildcard pattern - handle in _load_from_folder
    self.data_source = 'wildcard'
    self.df = self._load_from_folder(path)
```

#### Updated `_load_from_folder()` method (lines 79-119)
Improved wildcard handling:
```python
# Check if path contains wildcards
if '*' in str(folder_path) or '[' in str(folder_path):
    # Get the parent directory and the pattern
    if folder_path.is_absolute():
        parent = folder_path.parent
        pattern = folder_path.name
        csv_files = list(parent.glob(pattern))
    else:
        csv_files = list(Path('.').glob(str(folder_path)))
```

### 3. Multiple CSV File Concatenation

When loading from a folder with AssociatedResults:
```python
# Load and concatenate ALL CSV files
csv_files = list(assoc_results_dir.glob("*.csv"))
dfs = []
for csv_file in sorted(csv_files):
    dfs.append(pd.read_csv(csv_file))
return pd.concat(dfs, ignore_index=True)
```

### 4. Documentation

#### Updated README.md
- Comprehensive documentation of all new features
- Clear examples of both old and new column naming
- Wildcard usage examples
- Multiple CSV concatenation examples
- Complete API reference
- Migration guide from v1.x to v2.0

#### Created COLUMN_NAMING_UPDATE.md (this document)
- Technical summary of changes
- Implementation details
- Test coverage information

### 5. Tests

#### Created `tests/test_column_naming.py`
Comprehensive test suite with 8 tests:

1. **TestColumnNaming**:
   - `test_old_column_naming`: Tests old format detection and normalization
   - `test_new_column_naming`: Tests new format detection and normalization
   - `test_get_column_name_helper`: Tests the helper method

2. **TestMultipleCSVLoading**:
   - `test_load_all_csvs`: Tests loading all CSV files from AssociatedResults
   - `test_wildcard_loading`: Tests wildcard pattern `data_[01].csv`
   - `test_wildcard_with_asterisk`: Tests wildcard pattern `data_*.csv`

3. **TestFilteringWithNewColumns**:
   - `test_filter_photons_old_naming`: Tests filtering with old column naming
   - `test_filter_photons_new_naming`: Tests filtering with new column naming

**All 8 tests pass successfully ✓**

## Usage Examples

### Loading with New Column Naming

```python
import visualize_pixel_map as vpm

# Load data with new column format (px_*, ph_*, ev_*)
data = vpm.Data("data/neutrons")

# Properties work with both old and new naming
print(f"Photons: {data.photons['count']}")
print(f"Events: {data.events['count']}")

# Filtering works with both old and new naming
data.plot(
    photons=(0, 10),
    events=slice(0, 5),
    query="tot > 50"  # Works with both 'tot' and 'px_tot'
)
```

### Loading with Wildcards

```python
# Load specific files
data = vpm.Data("data/AssociatedResults/data_[01].csv")

# Load all files matching pattern
data = vpm.Data("data/AssociatedResults/data_*.csv")
```

### Multiple CSV Files

```python
# Automatically loads and concatenates all CSV files
data = vpm.Data("data/neutrons")
# If data/neutrons/AssociatedResults/ contains:
#   - data_0.csv (100 rows)
#   - data_1.csv (100 rows)
#   - data_2.csv (100 rows)
# Result: 300 rows total
```

## Backward Compatibility

✅ **100% backward compatible**

All existing code using the old column naming will continue to work:
- Old CSV files with `assoc_photon_id`, `assoc_event_id`, `tot` are fully supported
- Properties and filtering automatically detect which naming convention is in use
- No code changes required for existing projects

## Files Modified

1. **src/visualize_pixel_map/analyse.py**:
   - Updated `__init__` method
   - Updated `_load_from_folder()` method
   - Updated `_normalize_columns()` method
   - Added `_get_column_name()` helper method
   - Updated `photons` and `events` properties
   - Updated `_apply_filters()` method

2. **README.md**:
   - Complete rewrite with v2.0 features
   - New column naming documentation
   - Wildcard usage examples
   - Multiple CSV concatenation examples
   - Complete API reference

3. **tests/test_column_naming.py** (new):
   - Comprehensive test suite
   - Tests for both naming conventions
   - Tests for wildcard loading
   - Tests for multiple CSV concatenation
   - Tests for filtering with both naming conventions

## Testing

All tests pass:
```
============================= test session starts ==============================
tests/test_column_naming.py::TestColumnNaming::test_old_column_naming PASSED
tests/test_column_naming.py::TestColumnNaming::test_new_column_naming PASSED
tests/test_column_naming.py::TestColumnNaming::test_get_column_name_helper PASSED
tests/test_column_naming.py::TestMultipleCSVLoading::test_load_all_csvs PASSED
tests/test_column_naming.py::TestMultipleCSVLoading::test_wildcard_loading PASSED
tests/test_column_naming.py::TestMultipleCSVLoading::test_wildcard_with_asterisk PASSED
tests/test_column_naming.py::TestFilteringWithNewColumns::test_filter_photons_old_naming PASSED
tests/test_column_naming.py::TestFilteringWithNewColumns::test_filter_photons_new_naming PASSED

============================== 8 passed in 0.50s
```

## Next Steps

1. Run tests with real data to verify everything works as expected
2. Update tutorial notebooks to show new features
3. Consider adding more tests for edge cases
4. Update CHANGELOG.md for v2.0 release

## Summary Statistics

- **Lines of code modified**: ~150
- **New test file**: 1 (262 lines)
- **Tests added**: 8
- **Test pass rate**: 100%
- **Backward compatibility**: 100%
- **Documentation updated**: README.md, COLUMN_NAMING_UPDATE.md
