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
# ## Imports

# %%
from functools import partial
from pathlib import Path
from openghg.util import split_function_inputs
import xarray as xr
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import zipfile
import zarr
import re

from inversions.dask_helpers import zip_on_done

# replacement for %run
from inversions.data_functions import (
    read_ini,
    data_processing,
    fp_all_to_datatree,
    filter_data_vars,
    store_data_var,
    create_merged_data,
    create_and_save_merged_data,
    search_merged_data,
    load_merged_data,
)

# %% [markdown]
# # Data for SF6 tests
#
# I already saved some data using 4h averaging and 250 basis functions, along with the filters we typically use for PARIS.
#
# It would be helpful to save the data in a state where I could change the number of basis functions or filtering.

# %%

sf6_path = Path("/group/chem/acrg/PARIS_inversions/sf6/")
sf6_base_nid2025_path = sf6_path / "RHIME_NAME_EUROPE_FLAT_ConfigNID2025_sf6_yearly"
ini_files = [p.name for p in sf6_base_nid2025_path.glob("*.ini")]
# get 2015-2024
ini_files = ini_files[2:-1]

# %%
ini_files


# %% [markdown]
# ## Test for MultiObs and MultiFootprint with ModelScenario
#
# Alignment failed when trying to pass MultiObs and MultiFootprint directly to ModelScenario, so I resorted to using a loop over sites.

# %%

params = read_ini(ini_files[0])
data_params, _ = split_function_inputs(params, data_processing)

fp_all = data_processing(**data_params)

dt = fp_all_to_datatree(fp_all, rechunk=False)
dt

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Storing this data for later processing
#
# If I want to store the data without applying basis functions or filtering, what do I need to store?
# - need mean fp x mean flux, but we can get this from fp x flux...
# - need some footprint info for filters (mostly met. data, only need `fp` for `local_influence` filter)
#
# What if we want to resample again?
# - we don't have "number of obs" (but maybe we just don't have it for this data... F gases use Medusa)
# - can use resampling methods from openghg... this might just be an approximation though
#
# What is needed for post-processing?
# - fluxes
# - obs and obs uncertainties
# - release lat/lon


# %%
# test new helper function
dt = create_merged_data(params, chunks={"time": 400})

# %%
dt = dt.map_over_datasets(filter_data_vars, store_data_var)

# %% [markdown]
# # Storing data
#
# We'll store this data in `/user/work/bm13805/sf6_model_testing_data`.

# %%
work_path = Path("/user/work/bm13805/")
data_path = work_path / "sf6_model_testing_data"
output_name = "4h-no-basis-no-filt"


# %%

cluster = SLURMCluster(
    processes=4,
    cores=8,
    memory="40GB",
    walltime="01:00:00",
    account="chem007981",
)
client = Client(cluster)

# %%
client

# %%
client.close()

# %%
print(cluster.job_script())

# %%
njobs = len(ini_files)

# %%
func = partial(
    create_and_save_merged_data,
    merged_data_dir=data_path,
    output_name=output_name,
    chunks={"time": 400},
)

# %%
cluster.scale(jobs=njobs)
# cluster.scale(memory="40GB")
# cluster.adapt(minimum=0, maximum=10)


# %%
futures = client.map(func, ini_files)

# %%
for f in futures:
    f.add_done_callback(zip_on_done)

# %%
for ini_file, future in zip(ini_files, futures):
    #    future.cancel()
    print(ini_file.split("/")[-1], future)

# %%
# !ls -lsh {data_path} | grep -E "sf6"

# %%
# #to_delete = !ls {data_path} | grep -E "sf6.*zip"
# for f in to_delete:
# #    !rm {data_path / f}

# %%
results = [p.name for p in data_path.glob("sf6*zip")]

# %%

with zipfile.ZipFile(data_path / results[0]) as zf:
    for x in zf.infolist():
        print(x)
# dt_2015 = xr.open_datatree(data_path / "sf6_2015-01-01_4h-no-basis-no-filt_merged-data.zarr.zip", engine="zarr")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Test for saving and loading to/from zarr ZipStore
#
# ...I needed to pass `engine="zarr"` to `xr.open_datatree`


# %%
# !rm -rf {data_path/"test.zarr.zip"}
def test_zip_store(ini_file):
    params0 = read_ini(ini_file)
    dt = create_merged_data(params0)
    with zarr.ZipStore(data_path / "test.zarr.zip", mode="w") as store:
        dt.drop_nodes("scenario").to_zarr(store, mode="w")


future = client.submit(test_zip_store, ini_files[0])

# %%
future

# %%
test_dt = xr.open_datatree(data_path / "test.zarr.zip", engine="zarr", chunks={})

# %%
test_dt

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Checking results

# %%
# results = !ls {data_path} | grep -E "sf6.*zip"
results

# %%
dt = xr.open_datatree(data_path / results[0], engine="zarr", chunks={})

# %%
dt.scenario.MHD.fp_x_flux.encoding

# %%
dt.scenario.TAC.fp_x_flux.chunksizes

# %% [markdown]
# # Loading merged data
#
# We need to specify engine="zarr" and chunks={}, plus look up the name we created, so it will be useful to have a helper function to do this.

# %%

merged_data_name_pat = re.compile(
    r"(?P<species>[a-zA-Z0-9]+)_(?P<start_date>[\d-]+)_(?P<output_name>.+)_merged-data"
)

# %%
print(results[0])
print(merged_data_name_pat)
print(merged_data_name_pat.search(results[0]).groupdict())

# %%
merged_data_name_pat.search("ch4_2020-01-01_output_name_merged-data").groupdict()

# %%
md_info = search_merged_data(data_path)
md_info["ext"] = md_info["path"].apply(lambda x: x.suffix)
md_info

# %%
# !ls -lhst {data_path/"sf6_2016-01-01_4h-no-basis-no-filt_merged-data.zarr"/"scenario"/"CMN"/"fp_x_flux"}

# %% [markdown]
# So some of the zipped data has been duplicated... this seems bad.
#
# Does this happen when I open the data? 2017 doesn't have an unzipped copy, so I can try opening that...

# %%
md_info.loc[4, "path"]

# %%
dt_2017 = xr.open_datatree(data_path / md_info.loc[4, "path"], engine="zarr", chunks={})

# %%
# !ls -lst {data_path}

# %% [markdown]
# What if I open without chunks={}?

# %%
dt_2017 = xr.open_datatree(data_path / md_info.loc[4, "path"], engine="zarr")

# %%
dt_2017.scenario.TAC.fp_x_flux.data

# %%
# !ls -lsht {data_path}

# %% [markdown]
# From an earlier cell, I can see that right after the futures finished, eventually I had just the .zarr.zip results. Maybe the processes that were removing the .zarr files failed somehow?
#
# Or maybe some older jobs completed?
#
# Is the zipped data okay?

# %%
dt_2015_1 = xr.open_datatree(data_path / md_info.iloc[0, -2], engine="zarr")
dt_2015_2 = xr.open_datatree(data_path / md_info.iloc[1, -2], engine="zarr")
xr.testing.assert_isomorphic(dt_2015_1, dt_2015_2)

# %% [markdown]
# Okay so we can delete the unzipped data...

# %%
files_raw = [p.name for p in data_path.iterdir()]
files = [fr.strip().split()[-1] for fr in files_raw]
files

# for f in files[1:4]:
# #    !rm -rf {data_path/f}

# %%
dt_2021 = load_merged_data(data_path, start_date="2021")

# %%
dt_2021

# %%
