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

from openghg.retrieve import *

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
ed_cube = ed2022.fluxes.to_iris()

# %%
ed_cube

# %%
# direct construction
from iris.coords import DimCoord
from iris.cube import Cube

ed_lat = ed2022.coords["lat"].values
ed_lon = ed2022.coords["lon"].values
ed_cube_lat = DimCoord(ed_lat, standard_name="latitude", units="degrees")
ed_cube_lon = DimCoord(ed_lon, standard_name="longitude", units="degrees")
ed_cube2 = Cube(ed2022.fluxes.values, dim_coords_and_dims=[(ed_cube_lat, 0), (ed_cube_lon, 1)])
ed_cube2

# %%
#ed_cube2.coord("latitude").guess_bounds()
ed_cube2.coord("longitude").guess_bounds()
ed_cube2

# %%
lat_out, lon_out = find_domain("europe")
nlat = len(lat_out)
nlon = len(lon_out)
out_da = xr.DataArray(np.zeros((nlat, nlon), dtype="float32"), coords=[lat_out, lon_out], dims=["latitude", "longitude"])
out_da.latitude.attrs["units"] = "degrees"
out_da.longitude.attrs["units"] = "degrees"
out_cube = out_da.to_iris()

# %%
out_cube

# %%
# direct method for making output cube... this mattered for some reason
cube_lat_out = DimCoord(lat_out, standard_name="latitude", units="degrees")
cube_lon_out = DimCoord(lon_out, standard_name="longitude", units="degrees")
out_cube2 = Cube(
    np.zeros((len(lat_out), len(lon_out)), np.float32),
    dim_coords_and_dims=[(cube_lat_out, 0), (cube_lon_out, 1)],
)

# %%
out_cube2

# %%
out_cube == out_cube2

# %%
out_cube.coords()

# %%
out_cube2.coords()


# %%
def zeros_from_domain(domain: str) -> xr.DataArray:
    """Make a DataArray of zeros with lat/lon coords from given domain."""
    lat_out, lon_out = find_domain(domain)
    nlat = len(lat_out)
    nlon = len(lon_out)
    da_out = xr.DataArray(np.zeros((nlat, nlon)),
                          coords=[lat_out, lon_out], 
                          dims=["latitude", "longitude"])
    da_out.latitude.attrs = {"units": "degrees", "standard_name": "latitude"}
    da_out.longitude.attrs = {"units": "degrees", "standard_name": "longitude"}
    
    return da_out


def _regrid_2d(
    input_data: xr.DataArray,
    in_lat_coord: str = "lat",
    in_lon_coord: str = "lon",
    domain: str = "EUROPE",
) -> xr.DataArray:
    from iris.analysis import AreaWeighted

    cube_out = zeros_from_domain(domain).to_iris()
    cube_out.coord("latitude").guess_bounds()
    cube_out.coord("longitude").guess_bounds()

    cube_in = input_data.rename({in_lat_coord: "lat", in_lon_coord: "lon"}).to_iris()
    cube_in.coord("latitude").guess_bounds()
    cube_in.coord("longitude").guess_bounds()

    cube_regridded = cube_in.regrid(cube_out, AreaWeighted(mdtol=1.0))

    result = xr.DataArray.from_iris(cube_regridded).rename({"latitude": "lat", "longitude": "lon"})
    result.attrs = input_data.attrs
    result.attrs.pop("ChunkSizes", None)
    
    return result    

def regrid_2d(
    input_data: xr.Dataset | xr.DataArray,
    in_lat_coord: str = "lat",
    in_lon_coord: str = "lon",
    in_data_var: str = "flux",
    domain: str = "EUROPE",
) -> xr.Dataset:
    """Regrid 2d data to given domain.

    NOTE: only one data variable is regridded.

    Args:
        input_data: data to regrid
        in_lat_coord: name of latitude coordinate for input_data
        in_lon_coord: name of longitude coordinate for input_data
        in_data_var: name of the data variabile to regrid
        domain: name of domain to regrid to.
    Returns:
        xr.Dataset containing the regridded data.
    """
    if isinstance(input_data, xr.Dataset):
        data_in = input_data[in_data_var]
    else:
        data_in = input_data

    result = _regrid_2d(data_in, in_lat_coord, in_lon_coord, domain).to_dataset()
    result.attrs = input_data.attrs
    return result


# %%
ed2022_regridded = regrid_2d(ed2022, in_data_var="fluxes").rename(fluxes="flux")

# %%
ed2022_regridded

# %%
np.log(ed2022_regridded.flux).plot()

# %%
from openghg.standardise.meta import define_species_label
from openghg.util import molar_mass

sf6_molar_mass = molar_mass(define_species_label("sf6")[0])  # why use "define_species_label"?

# %%
sf6_molar_mass

# %%
molar_mass("sF6")

# %%
ed2022_regridded.flux.max() / molar_mass("sf6")

# %%
from openghg.util import cf_ureg


# %%
sf6_molar_mass = molar_mass("sf6") * cf_ureg.g / cf_ureg.mol
sf6_molar_mass

# %%
cf_ureg.define("@alias kilogram = kg")

# %%
cf_ureg((ed2022_regridded.pint.quantify() / sf6_molar_mass).pint.dequantify().flux.attrs["units"]).to("mol/m2/s")

# %%
print(cf_ureg.parse_unit_name("Mg"))
print(cf_ureg.parse_unit_name("Mg", case_sensitive=True))

# %%
from openghg.util._units import inverse_unit_mapping
pprint(inverse_unit_mapping)


# %%
def transform_edgar_flux_file(filepath: str | Path, domain: str, species: str | None = None, year: int | None = None, month: int = 1) -> xr.Dataset:
    with xr.open_dataset(filepath, chunks={}) as ds:
        species = species or ds.fluxes.attrs.get("substance")
        ds_regridded = regrid_2d(ds, domain=domain, in_data_var="fluxes")

    ds_regridded = ds_regridded.rename(fluxes="flux")

    if year is None:
        try:
            year = int(ds_regridded.flux.attrs["year"])
        except (KeyError, ValueError):
            pass

    if year is not None:
        ds_regridded = ds_regridded.expand_dims({"time": [pd.Timestamp(year=year, month=month, day=1)]})

    if species is None:
        return ds_regridded

    molmass = molar_mass(species) * cf_ureg.g / cf_ureg.mol

    with xr.set_options(keep_attrs=True):
        ds_regridded = (ds_regridded.pint.quantify() / molmass).pint.to("mol/m2/s")

    return ds_regridded.pint.dequantify()


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

_add_path_to_config("group/chem/acrg/object_stores/paris/sf6_testing_store", name="sf6_testing_store")

# %%
from openghg.standardise import standardise_flux

# %%
# standardise_flux?

# %% [markdown]
# Let's get the flux data regridded and keep it in memory, since we'll need it to make the synthetic obs.

# %%
fluxes = [transform_edgar_flux_file(flux_path_yearly / yf, domain="europe", species="sf6") for yf in yearly_files[-10:]]

# %%
flux_ds = xr.concat(fluxes, dim="time")

# %%
flux_ds.flux.max().compute()

# %%
import tempfile

with tempfile.NamedTemporaryFile() as f:
    flux_ds.to_netcdf(f, engine="h5netcdf")
    standardise_flux(f.name,
                     domain="europe", 
                     source="edgar-annual-total", 
                     species="sf6", 
                     database="edgar", 
                     database_version="v8.0", 
                     period="yearly", 
                     if_exists="new",
                     store="sf6_testing_store")

# %%
flux_res = search_flux(species="sf6", domain="europe")

# %%
flux_res
