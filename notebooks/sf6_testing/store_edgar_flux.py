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
# # Storing EDGAR fluxes
#
# EDGAR fluxes for SF6 weren't available, so I added them to an object store.
# Since I didn't want to deal with xesmf, I added some custom code to do this.

# %%
from pathlib import Path
import openghg

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from openghg.retrieve import get_flux, search_flux

from inversions.regridding import regrid_2d, transform_edgar_flux_file
from inversions.utils import ls

# %%
flux_res = search_flux(species="sf6", domain="europe")

# %%
flux_res

# %%
flux_path = Path("/group/chem/acrg/Gridded_fluxes/SF6/EDGAR_v8.0/")
flux_path_yearly = flux_path / "yearly"
flux_path_yearly_sectoral = flux_path / "yearly_sectoral"

# !ls {flux_path_yearly} | head

# %%
# !ls {flux_path_yearly_sectoral}

# %% [markdown]
# It seems these fluxes aren't stored anywhere, so we'll have create a store...

# %%
stores_path = Path("/group/chem/acrg/object_stores/")
# !ls {stores_path}/paris

# %%
from pprint import pprint
from openghg.objectstore import get_readable_buckets

pprint(get_readable_buckets())

# %% [markdown]
# Maybe I can just look at the raw files first...

# %% [markdown]
# ## EDGAR fluxes for SF6

# %%
yearly_files = ls(flux_path_yearly)
yearly_files[-10:]

# %%
ed2022 = xr.open_dataset(flux_path_yearly / yearly_files[-1])

# %%
ed2022

# %%
from openghg.util import find_domain

lat, lon = find_domain("europe")

# %%
lat_slice = slice(lat.min(), lat.max())
lon_slice = slice(lon.min(), lon.max())
np.log(ed2022.fluxes.sel(lat=lat_slice, lon=lon_slice)).plot()

# %% [markdown]
# Looks okay... let's zoom in on Europe

# %%
np.log(ed2022.fluxes.sel(lat=slice(35, 60), lon=slice(-15, 20))).plot()


# %%
ed2022_regridded = regrid_2d(ed2022, in_data_var="fluxes").rename(fluxes="flux")

# %%
ed2022_regridded

# %%
np.log(ed2022_regridded.flux).plot()


# %%
flux2022 = transform_edgar_flux_file(flux_path_yearly / yearly_files[-1], "europe")

# %%
flux2022.flux.pint.quantify().max().compute()

# %% [markdown]
# # Making synthetic data from EDGAR fluxes
#
# ## Storing EDGAR total fluxes for SF6
# We probably want to store the fluxes in an object store, since we'll need to compare with them later.

# %%
# !ls /group/chem/acrg/object_stores/paris

# %%
# !cat /user/home/bm13805/.openghg/openghg.conf

# %%
from openghg.util._user import _add_path_to_config

_add_path_to_config(
    "group/chem/acrg/object_stores/paris/sf6_testing_store", name="sf6_testing_store"
)

# %%
from openghg.standardise import standardise_flux

# %%
# standardise_flux?

# %% [markdown]
# Let's get the flux data regridded and keep it in memory, since we'll need it to make the synthetic obs.

# %%
fluxes = [
    transform_edgar_flux_file(flux_path_yearly / yf, domain="europe", species="sf6")
    for yf in yearly_files[-10:]
]

# %%
flux_ds = xr.concat(fluxes, dim="time")

# %%
flux_ds.flux.max().compute()

# %%
import tempfile

with tempfile.NamedTemporaryFile() as f:
    flux_ds.to_netcdf(f, engine="h5netcdf")
    standardise_flux(
        f.name,
        domain="europe",
        source="edgar-annual-total",
        species="sf6",
        database="edgar",
        database_version="v8.0",
        period="yearly",
        if_exists="new",
        store="sf6_testing_store",
    )

# %%
flux_res = search_flux(species="sf6", domain="europe")

# %%
flux_res
