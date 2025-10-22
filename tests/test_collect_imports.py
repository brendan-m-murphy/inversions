# tests/test_collect_imports.py

import sys
import subprocess
from pathlib import Path
from textwrap import dedent

SAMPLE = dedent('''
# ---
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


def test_collect_imports_creates_imports_block(tmp_path: Path) -> None:
    # write sample to temp file
    p = tmp_path / "sample.py"
    p.write_text(SAMPLE, encoding="utf8")

    # run the collect_imports script using the repo python
    script = Path("scripts/collect_imports.py")
    assert script.exists(), "collect_imports script not found"

    cp = subprocess.run([sys.executable, str(script), "--no-inplace", str(p)],
                        check=True, text=True, capture_output=True)
    out = cp.stdout

    # basic sanity checks
    assert "# # Imports" in out

    # imports should appear exactly once each
    assert out.count("from pathlib import Path") == 1
    assert out.count("from openghg_inversions.hbmcmc.run_hbmcmc import hbmcmc_extract_param") == 1
    assert out.count("from openghg.util import split_function_inputs") == 1

    # imports block should appear before the Data for SF6 tests heading
    imports_idx = out.index("# # Imports")
    data_idx = out.index("# # Data for SF6 tests") if "# # Data for SF6 tests" in out else out.index("# # Data for SF6 tests".replace("# #", "#"))
    assert imports_idx < data_idx
