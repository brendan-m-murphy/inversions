"""#!/usr/bin/env python3
# content of tests/test_collect_imports.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def test_collect_imports_hoists_top_level_imports(tmp_path):
    """Run the collect_imports script on a sample jupytext-generated .py file
    and verify that top-level imports are hoisted into an Imports cell near
    the top of the file.
    """
    sample = '''# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data for SF6 tests
#
# I already saved some data using 4h averaging and 250 basis functions, along with the filters we typically use for PARIS.
#
# It would be helpful to save the data in a state where I could change the number of basis functions or filtering.

# %%
from pathlib import Path

sf6_path = Path("/group/chem/acrg/PARIS_inversions/sf6/")
sf6_base_nid2025_path = sf6_path / "RHIME_NAME_EUROPE_FLAT_ConfigNID2025_sf6_yearly"
# ini_files = !ls {sf6_base_nid2025_path / "*.ini"}
# get 2015-2024
ini_files = ini_files[2:-1]

# %%
ini_files

# %%
# %run inversions_experimental_code/data_functions.py

# %% [markdown]
# ## Test for MultiObs and MultiFootprint with ModelScenario
#
# Alignment failed when trying to pass MultiObs and MultiFootprint directly to ModelScenario, so I resorted to using a loop over sites.

# %%
from openghg_inversions.hbmcmc.run_hbmcmc import hbmcmc_extract_param
from openghg.util import split_function_inputs

params = read_ini(ini_files[0])
data_params, _ =  split_function_inputs(params, data_processing)
'''    
    fpath = tmp_path / "sample.py"
    fpath.write_text(sample)

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "collect_imports.py"
    assert script.exists(), f"collect_imports script not found at {script}"

    # Run the script in no-inplace mode so it prints the transformed file to stdout
    cp = subprocess.run([sys.executable, str(script), "--no-inplace", str(fpath)],
                        check=True, capture_output=True, text=True)
    out = cp.stdout

    # Check that an Imports heading was inserted
    assert "# # Imports" in out

    # Check that expected imports were hoisted
    assert "from pathlib import Path" in out
    assert "from openghg_inversions.hbmcmc.run_hbmcmc import hbmcmc_extract_param" in out
    assert "from openghg.util import split_function_inputs" in out

    # Ensure imports appear before the Data heading
    assert out.index("from pathlib import Path") < out.index("# # Data for SF6 tests")

    # Ensure the original import lines are not duplicated later in the file
    assert out.count("from openghg_inversions.hbmcmc.run_hbmcmc import hbmcmc_extract_param") == 1
"""