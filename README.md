Code and notebooks for testing new inversions setups.

Useful code from here should be merged into OpenGHG or OpenGHG Inversions eventually.

## Scripts
### `inversions install_kernel`
Command that installs an IPython kernel pointing at the project's virtualenv:
- Defaults to using the current Python interpreter (perfect for uv-managed environments)
- Supports custom venv paths, kernel names, and display names
- Can optionally patch notebooks to use the new kernel
- Automatically ensures ipykernel and jupyter_client are installed

**Usage:**
```bash
# Install kernel using current Python interpreter
inversions install_kernel

# Install with custom name and display
inversions install_kernel --name my-kernel --display "My Kernel"

# Install and patch all notebooks to use this kernel
inversions install_kernel --venv /path/to/venv --patch-notebooks

# Force reinstall if kernel already exists
inversions install_kernel --force
```

### Import Collection Script (`scripts/collect_imports.py`)
Script that parses jupytext .py files and hoists imports to a single cell:
- Collects all top-level import statements from code cells
- Deduplicates imports while preserving order
- Creates an "Imports" cell after the front matter
- Removes original scattered imports from code cells
- Supports dry-run and backup modes

**Usage:**
```bash
# Process one or more files
python scripts/collect_imports.py file.py

# Preview changes without modifying files
python scripts/collect_imports.py --dry-run file.py

# Create backups before modifying
python scripts/collect_imports.py --backup file1.py file2.py
```
