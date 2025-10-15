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
# # Synthetic data for SF6
#
# The only "objective" way we have to assess the (potential) quality of our posterior fluxes is by constructing observations based on known emissions.
#
# First, let's see what prior data we have available.

# %%
from pathlib import Path
import openghg

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from openghg.retrieve import *

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
# yearly_files = !ls {flux_path_yearly}
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

# %% [markdown]
# # Making synthetic obs

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

# %%
params = read_ini(ini_files[-3])

# %%
pprint(dict(params))

# %%
obs_params, _ = split_function_inputs(params, MultiObs.__init__)
obs_params["inlets"] = params["inlet"]
obs_params["instruments"] = params["instrument"]
obs_params["averaging_periods"] = ["1h"] * len(obs_params["sites"])
obs_params["obs_data_levels"] = [None] * len(obs_params["sites"])
pprint(obs_params)

# %%
obs_2022 = MultiObs(**obs_params)

# %%
fp_params, _ = split_function_inputs(params, MultiFootprint.__init__)
pprint(fp_params)

# %%
fp_params["fp_heights"] = params["fp_height"]
fp_params["met_model"] = [None] * len(fp_params["sites"])
fp_params["model"] = "name"
pprint(fp_params)

# %%
fp_2022 = MultiFootprint(**fp_params, obs_data=obs_2022.obs, obs_sites=obs_2022.sites)

# %%
dt_2022 = xr.DataTree.from_dict({k: v.data for k, v in fp_2022.footprints.items()})


# %%
def flux_mult(footprint: xr.Dataset, flux: xr.Dataset, ffill: bool = False) -> xr.Dataset:
    if "fp" not in footprint.data_vars:
        return footprint
        
    if ffill:
        flux = flux.reindex_like(footprint, method="ffill")
    result = footprint.copy()

    fp_x_flux = (footprint.fp.pint.quantify() * flux.flux.pint.quantify()).pint.dequantify()
    result["mod_obs"] = fp_x_flux.sum(["lat", "lon"]).astype("float32")
    result["mod_obs"].attrs["units"] = fp_x_flux.attrs["units"]
    return result


# %%
flux_2022_obj = get_flux(species="sf6", domain="europe", source="edgar-annual-total")
flux_2022 = flux_2022_obj.data
flux_2022_obj.metadata

# %%
flux_2022.flux.max().compute()

# %%
dt_2022 = dt_2022.map_over_datasets(flux_mult, flux_2022, True)

# %%
dt_2022.MHD.mod_obs

# %%
dt_2022.MHD.mod_obs.values[:20]

# %%
mhd_2022_synth = dt_2022.MHD.mod_obs.pint.quantify().pint.to("ppt").pint.dequantify().compute()

# %%
mhd_2022_synth.to_series().describe()

# %%
data_path = Path("sf6_model_testing_data/")
merged_data = load_merged_data(data_path, start_date="2022", species="sf6")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

mhd_2022_synth.plot(ax=ax, label="synth", alpha=0.5)
obs_2022.obs["MHD"].data.mf.plot(ax=ax, label="real", alpha=0.5)
fig.legend()
#ax.set_ylim(10.6, 11.6)

# %%
merged_data.scenario.MHD


# %%
def nesw_bc_basis(ds: xr.Dataset) -> xr.DataArray:
    bc_ds = ds[[f"bc_{d}" for d in "nesw"]].rename({f"bc_{d}": d for d in "nesw"})
    return bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region")


# %%
baseline = nesw_bc_basis(merged_data.scenario.MHD.dataset).sum("bc_region")

# %%
baseline = baseline.reindex_like(mhd_2022_synth, method="ffill")

# %%
mod_obs = merged_data.scenario.MHD.fp_x_flux.sum(["lat", "lon"])
mod_obs = mod_obs.reindex_like(mhd_2022_synth, method="ffill")

# %%
fig, ax = plt.subplots()

(baseline + mhd_2022_synth - 0.02).resample(time="4h").mean().plot(ax=ax, label="synth", alpha=0.5)
obs_2022.obs["MHD"].data.mf.resample(time="4h").mean().plot(ax=ax, label="real", alpha=0.5)
(mod_obs + baseline - 0.02).resample(time="4h").mean().plot(ax=ax, label="prior", alpha=0.5)
fig.legend()
ax.set_ylim(10.9, 11.6)

# %%
fix, ax = plt.subplots()

mhd_2022_synth.plot(ax=ax, label="synth", alpha=0.5)
(obs_2022.obs["MHD"].data.mf - baseline).plot(ax=ax, label="real - baseline", alpha=0.5)
mod_obs.plot(ax=ax, label="mod obs", alpha=0.5)
ax.legend()

# %%
baseline2 = obs_2022.obs["MHD"].data.mf.resample(time="14D").median()
baseline2 = baseline2.reindex_like(mhd_2022_synth, method="ffill")

# %%
fig, ax = plt.subplots()

baseline.plot(ax=ax, label="prior baseline", alpha=0.5)
baseline2.plot(ax=ax, label="14D median baseline", alpha=0.5)
fig.legend()
ax.set_ylim(10.9, 11.6)

# %%
fig, ax = plt.subplots()
bias = 0.0 # -0.02
(baseline2 + mhd_2022_synth + bias).plot(ax=ax, label="synth", alpha=0.5)
obs_2022.obs["MHD"].data.mf.plot(ax=ax, label="real", alpha=0.5)
(mod_obs + baseline2 + bias).plot(ax=ax, label="prior", alpha=0.5)
fig.legend()
ax.set_ylim(10.9, 11.6)

# %% [markdown]
# ## Adding uncertainty
#
# We could use the magnitude of the real obs minus the baseline to get an idea of the size of the errors we might want to add.

# %%
stats = (obs_2022.obs["MHD"].data.mf - baseline).to_series().describe()
stats

# %%
baseline.shape


# %%
def make_noise_like(da: xr.DataArray, sigma: float, seed: int | None = None) -> xr.DataArray:
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(scale=sigma, size=da.shape).astype("float32")
    result = xr.DataArray(noise, coords=da.coords, dims=da.dims)
    return result


# %%
noise = make_noise_like(mhd_2022_synth, sigma=0.03, seed=123456)

# %%
fix, ax = plt.subplots()

(noise + mhd_2022_synth - 0.02).resample(time="4h").mean().plot(ax=ax, label="synth+noise", alpha=0.5)
(obs_2022.obs["MHD"].data.mf - baseline2).resample(time="4h").mean().plot(ax=ax, label="real - baseline", alpha=0.5)
#mod_obs.plot(ax=ax, label="mod obs", alpha=0.5)
#noise.plot(ax=ax, label="noise", alpha=0.3)
#obs_2022.obs["MHD"].data.mf_repeatability.plot(ax=ax, label="mf repeatability", alpha=0.3)
ax.legend()

# %% [markdown]
# ## Making synthetic obs
#
# - baseline: load merged data and get prior baseline
# - pollution events: load edgar flux, load multi_fp and map over data tree
# - uncertainties: take 1/2 IQR of obs for sigma (could also do no uncertainty, half this, and twice this)
# - mf_repeatability: make this constant, equal to value of sigma? or just copy from obs?

# %%
flux = get_flux(species="sf6", domain="europe", source="edgar-annual-total").data

year = "2015"
ini_file = next(ini_file for ini_file in ini_files if year in ini_file)
params = read_ini(ini_file)

obs_params, _ = split_function_inputs(params, MultiObs.__init__)
obs_params["inlets"] = params["inlet"]
obs_params["instruments"] = params["instrument"]
obs_params["averaging_periods"] = ["1h"] * len(obs_params["sites"])
obs_params["obs_data_levels"] = [None] * len(obs_params["sites"])
multi_obs = MultiObs(**obs_params)

fp_params, _ = split_function_inputs(params, MultiFootprint.__init__)
fp_params["fp_heights"] = params["fp_height"]
fp_params["met_model"] = [None] * len(fp_params["sites"])
fp_params["model"] = "name"
multi_fp = MultiFootprint(**fp_params, obs_data=multi_obs.obs, obs_sites=multi_obs.sites)

flux_sel = flux.sel(time=f"{year}-01-01")
fp_dt = xr.DataTree.from_dict({k: v.data for k, v in multi_fp.footprints.items()}).map_over_datasets(flux_mult, flux_sel, True)

def get_mod_obs(ds: xr.Dataset) -> xr.Dataset:
    if "mod_obs" not in ds.data_vars:
        return ds
    return ds[["mod_obs"]].pint.quantify().pint.to("ppt").pint.dequantify().compute()

all_mod_obs = fp_dt.map_over_datasets(get_mod_obs)

# %%
all_mod_obs

# %%
data_path = Path("sf6_model_testing_data/")
merged_data = load_merged_data(data_path, start_date=year, species="sf6")

def make_prior_baseline(ds: xr.Dataset) -> xr.Dataset:
    if "bc_n" not in ds.data_vars:
        return ds
    return nesw_bc_basis(ds).sum("bc_region").rename("baseline").to_dataset()

baselines = merged_data.scenario.map_over_datasets(make_prior_baseline)

# %%
baselines

# %%
baseline_dict = {k: v.to_dataset() for k, v in baselines.items()}

# %%
baseline_dict

# %%
mod_obs_dict = {k: v.to_dataset() for k, v in all_mod_obs.items()}
mod_obs_dict

# %%
synth_obs = {}
std_cutoff = 0.7
std_resample = "8D"
std_scaling = 0.9
for site, mod_obs in mod_obs_dict.items():
    baseline = baseline_dict[site]
    baseline = baseline.reindex_like(mod_obs, method="nearest")
    ds = xr.merge([mod_obs, baseline])

    obs = multi_obs.obs[site.upper()].data
    std = std_scaling * ((obs.mf
        .where(obs.mf < obs.mf.quantile(std_cutoff).values, drop=True)
        .resample(time=std_resample).std())
#        .reindex_like(mod_obs.mod_obs, method="nearest")
        .median()
        )
    ds["noise"] = make_noise_like(mod_obs.mod_obs, sigma=std.values, seed=123456789)
    ds["noise"].attrs["sigma"] = std
    
    ds["sf6"] = mod_obs.mod_obs + baseline.baseline
    ds["sf6"].attrs = obs.mf.attrs
    ds.attrs = obs.attrs
    ds["sf6_repeatability"] = std * xr.ones_like(ds["sf6"])
    ds["sf6_repeatability"].attrs["units"] = df["sf6"].attrs["units"]
    synth_obs[site] = ds.compute()
    

# %%
multi_obs.obs["MHD"].data.mf.to_series().describe()

# %%
import scipy.stats as sst

# %%
synth_obs["MHD"].noise.attrs

# %% [markdown]
# ## Plotting synthetic obs

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
to_subtract = [0, (baseline_dict["MHD"].baseline -0.03).reindex_like(synth_obs["MHD"], method="ffill")]


for to_sub, ax in zip(to_subtract, axs.flat):
    (synth_obs["MHD"].sf6 + synth_obs["MHD"].noise - to_sub).plot(ax=ax, label="synth+noise", alpha=0.3)
    (multi_obs.obs["MHD"].data.mf - to_sub).plot(ax=ax, label="obs", alpha=0.3)
    if not isinstance(to_sub, xr.DataArray):
        to_subtract[1].plot(ax=ax, label="baseline", alpha=0.3)
    (synth_obs["MHD"].sf6 - to_sub).plot(ax=ax, label="synth", alpha=0.5)
    ax.legend()

# %%
pol_events = (multi_obs.obs["MHD"].data.mf - to_subtract[1]).to_series().sort_values(ascending=False)[:20].sort_index()
pol_events

# %%
multi_fp.footprints["MHD"].data

# %%
multi_fp.footprints["MHD"].data[["wind_from_direction", "wind_speed", "atmosphere_boundary_layer_thickness"]].sel(time=pol_events.index).compute().to_dataframe()

# %%
{k: v for k, v in multi_obs.obs["MHD"].metadata.items() if "station" in k}

# %%
!# in a notebook cell
# !wget -q -O natural_earth_50.zip "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

# %%
import geopandas as gpd

world = gpd.read_file("natural_earth_50.zip")  

# %%
nrows = len(pol_events) // 2
fig, axs = plt.subplots(nrows, 2, figsize=(15, 5 * nrows))
times = ["2015-04-07 15:00:00", "2015-11-01 06:00:00", "2015-02-11 11:00:00", "2015-08-17 09:00:00", "2015-03-19 09:00:00", "2015-10-04 04:00:00"]
times.sort()
times = pol_events.index
lat_min, lat_max = 40, 63
lon_min, lon_max = -20, 15
for time, ax, pe in zip(times, axs.flat, pol_events.values):
    np.pow(multi_fp.footprints["MHD"].data.fp.sel(time=time, lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)), 0.1).plot(ax=ax, vmin=0, vmax=1.1)
    world.boundary.plot(ax=ax, linewidth=0.6, edgecolor='white')  # or .plot(facecolor='none')
    ax.set_xlim(float(lon_min), float(lon_max))
    ax.set_ylim(float(lat_min), float(lat_max))
    ax.set_title(f"time = {time}, pe={pe:.4f}ppt")

# %%
info_2010 = pd.read_csv("sf6_model_testing_data/sf6_info_2010.csv")
info_2015 = pd.read_csv("sf6_model_testing_data/sf6_info_2015.csv")

# %%
# info_2010.loc[30, "longitude"] *= -1  # fix glasgow longitude
info_2010

# %%
possible_point_sources = gpd.GeoDataFrame({"label": info_2010.company_name.values}, geometry=gpd.points_from_xy(info_2010.longitude, info_2010.latitude), crs="EPSG:4326")

# %%
fig, ax = plt.subplots(figsize=(15, 8))
lon_min, lon_max = -15, 25
lat_min, lat_max = 35, 65
np.pow(flux_sel.flux, 0.1).plot(ax=ax)
world.boundary.plot(ax=ax, linewidth=0.6, edgecolor='white')
# possible_point_sources.plot(ax=ax, color="red", markersize=50, zorder=6)

# Plot points (geometry.x = lon, geometry.y = lat)
xs = possible_point_sources.geometry.x
ys = possible_point_sources.geometry.y
ax.scatter(xs, ys, s=40, c='red', edgecolor='k', zorder=6)

# Add labels next to the points
#labels = possible_point_sources['label']
labels = info_2010.index
for x, y, lab in zip(xs, ys, labels):
    # ax.text(x + 0.05, y + 0.05, lab, fontsize=8, zorder=7, color="white")  # tweak offsets as needed
    ax.annotate(
        lab,
        xy=(x, y),                    # data coords for the point
        xytext=(3, 3),                # offset in points (x, y)
        textcoords='offset points',   # interpret xytext in display points
        fontsize=8,
        zorder=7,
        bbox=dict(facecolor='white', alpha=0.6, pad=1),
        clip_on=True,
    )

ax.set_xlim(float(lon_min), float(lon_max))
ax.set_ylim(float(lat_min), float(lat_max))
ax.legend()

# %% [markdown]
# ## Pipeline for making synth obs
#
# 1. pollution events: load edgar flux, load multi_fp, map over data tree
# 2. get baseline: load merged data, compute NESW and sum over regions; fill this to match obs
# 3. get dictionaries of data sets for PE and baseline (not sure how to combine DataTrees)
# 4. make synth obs
# 5. make noise: "winsorised" stdev of real obs minus baseline (to de-trend); take the median and use the same value for all times as sigma; scale this median (0.9 seems "about right", could do less for "easier" scenarios)
# 6. make `sf6` and `sf6_repeatability`; repeatability will just be the sigma used to make the noise (?)
# 7. standardise
#    - get metadata from obs: species, site, inlet
#    - use source_format = "openghg"
#    - sampling period can be 1h? ...no this is e.g. 20 minutes for Medusa
#    - add calibration scale?
#    - use e.g. instrument: "edgar-v8_mod-baseline_sigma-0.9" and "dataset_source": "synthetic"
#    - info_metadata = {"flux": "edgar-annual-total", "baseline": "modelled baseline", "noise": "normal", "noise_scaling": 0.9}
#
#
#

# %% [markdown]
# ### How should we standardise?
#
# Use instrument to encode some features, put details on the features into "additional metadata"?

# %%
pprint(multi_obs.obs["MHD"].metadata)

# %%
from openghg.standardise import standardise_surface
# standardise_surface?

# %% [markdown]
# And the schema we need to use:

# %%
from openghg.store import ObsSurface
ObsSurface.schema("sf6")

# %%
from openghg.types import MetadataAndData

bucket = get_readable_buckets()["sf6_testing_store"]
ObsSurface(bucket=bucket).get_lookup_keys([MetadataAndData(metadata={}, data=synth_obs["MHD"])])

# %%
ObsSurface(bucket=bucket).add_metakeys()


# %% [markdown]
# ### Collecting data creation code

# %%
def get_multi_obs(params: dict, **kwargs) -> MultiObs:
    params = params | kwargs
    obs_params, _ = split_function_inputs(params, MultiObs.__init__)
    obs_params["inlets"] = params["inlet"]
    obs_params["instruments"] = params["instrument"]
    obs_params["averaging_periods"] = ["1h"] * len(obs_params["sites"]) # TODO: don't hard code this
    obs_params["obs_data_levels"] = obs_params.get("obs_data_level") or [None] * len(obs_params["sites"])
    return MultiObs(**obs_params)


def get_multi_fp(params: dict, multi_obs: MultiObs | None = None, **kwargs) -> MultiFootprint:
    params = params | kwargs
    fp_params, _ = split_function_inputs(params, MultiFootprint.__init__)
    fp_params["fp_heights"] = params["fp_height"]
    fp_params["met_model"] = [None] * len(fp_params["sites"])
    fp_params["model"] = "name"

    if multi_obs is None:
        return MultiFootprint(**fp_params)

    return MultiFootprint(**fp_params, obs_data=multi_obs.obs, obs_sites=multi_obs.sites)


def flux_mult(footprint: xr.Dataset, flux: xr.Dataset, ffill: bool = False) -> xr.Dataset:
    """Map over a DataTree of footprints to make modelled pollution events."""
    if "fp" not in footprint.data_vars:
        return footprint

    if flux.sizes.get("time") == 1:
        flux = flux.squeeze("time")
        ffill = False
    
    if ffill:
        flux = flux.reindex_like(footprint, method="ffill")
    result = footprint.copy()

    fp_x_flux = (footprint.fp.pint.quantify() * flux.flux.pint.quantify()).pint.dequantify()
    result["mod_obs"] = fp_x_flux.sum(["lat", "lon"]).astype("float32")
    result["mod_obs"].attrs["units"] = fp_x_flux.attrs["units"]
    return result


def make_mod_obs_dict_and_multi_obs_fp(ini_file: str | Path, flux: xr.Dataset, **kwargs):
    """Make dict of modelled obs datasets."""
    params = read_ini(ini_file)
    multi_obs = get_multi_obs(params, **kwargs)
    multi_fp = get_multi_fp(params, multi_obs, **kwargs)

    fp_dt = xr.DataTree.from_dict({k: v.data for k, v in multi_fp.footprints.items()}).map_over_datasets(flux_mult, flux, True)

    def get_mod_obs(ds: xr.Dataset) -> xr.Dataset:
        if "mod_obs" not in ds.data_vars:
            return ds
        return ds[["mod_obs"]].pint.quantify().pint.to("ppt").pint.dequantify().compute()

    all_mod_obs = fp_dt.map_over_datasets(get_mod_obs)
    mod_obs_dict = {k: v.to_dataset() for k, v in all_mod_obs.items()}
    return mod_obs_dict, multi_obs, multi_fp


def make_baseline_dict(merged_data: xr.DataTree) -> dict[str, xr.Dataset]:
    def make_prior_baseline(ds: xr.Dataset) -> xr.Dataset:
        if "bc_n" not in ds.data_vars:
            return ds
        return nesw_bc_basis(ds).sum("bc_region").rename("baseline").to_dataset()

    baselines = merged_data.scenario.map_over_datasets(make_prior_baseline)    
    return {k: v.to_dataset() for k, v in baselines.items()}


# %%
def choose_key(k: str) -> bool:
    if k in ("site", "species", "units", "inlet"):
        return True
    if any(x in k for x in ("inlet", "station", "sampling", "data_owner")):
        return True
    return False
                
def make_synth_obs(
    mod_obs_dict: dict,
    baseline_dict: dict,
    multi_obs: MultiObs, 
    std_cutoff: float = 0.7, 
    std_resample: str = "8D", 
    std_scaling: float = 0.9,
    seed: int | None = 123456789,
    bias: float | dict[str, float] | None = None
) -> dict[str, xr.Dataset]:
    synth_obs = {}

    if bias is None:
        bias = 0.0

    if isinstance(bias, float):
        bias = {site: bias for site in mod_obs_dict}
    
    for site, mod_obs in mod_obs_dict.items():
        baseline = baseline_dict[site]
        baseline = baseline.reindex_like(mod_obs, method="nearest")
        ds = xr.merge([mod_obs, baseline])

        obs = multi_obs.obs[site.upper()].data
        species = obs.attrs.get("species") or multi_obs[site.upper()].metadata.get("species")
        species = species.lower()

        
        std = std_scaling * ((obs.mf
            .where(obs.mf < obs.mf.quantile(std_cutoff).values, drop=True)
            .resample(time=std_resample).std())
    #        .reindex_like(mod_obs.mod_obs, method="nearest")
            .median()
            )
        ds["noise"] = make_noise_like(mod_obs.mod_obs, sigma=std.values, seed=seed)
        try:
            ds["noise"].attrs["sigma"] = float(std.values)
        except (AttributeError, ValueError):
            continue

        if std_scaling != 0.0:
            ds[species] = mod_obs.mod_obs + baseline.baseline + bias.get(site, 0.0) + ds["noise"]
        else:
            ds[species] = mod_obs.mod_obs + baseline.baseline + bias.get(site, 0.0)
            
        ds[species].attrs = obs.mf.attrs

                
        global_attrs = {k: v for k, v in obs.attrs.items() if choose_key(k)}

        # we used 1h averaging, so set sampling period...
        global_attrs["sampling_period"] = "3600.0"
        global_attrs["sampling_period_unit"] = "s"
        
        ds.attrs = global_attrs
        
        ds[f"{species}_repeatability"] = std * xr.ones_like(ds[species])
        ds[f"{species}_repeatability"].attrs["units"] = ds[species].attrs["units"]
        synth_obs[site] = ds.compute()

    return synth_obs

# %% [markdown]
# standardise
# get metadata from obs: species, site, inlet
# use source_format = "openghg"
# sampling period can be 1h? ...no this is e.g. 20 minutes for Medusa
# add calibration scale?
# use e.g. instrument: "edgar-v8_mod-baseline_sigma-0.9" and "dataset_source": "synthetic"
# info_metadata = {"flux": "edgar-annual-total", "baseline": "modelled baseline", "noise": "normal", "noise_scaling": 0.9}

# %%
from collections import defaultdict


def make_standardise_args(sites: list[str], multi_obs: MultiObs, instrument: str, flux: FluxData | None = None, **kwargs) -> dict[str, dict]:
    """For each site, create a dict of args to pass to standardise_surface."""
    result = defaultdict(dict)

    for site in sites:
        result[site]["site"] = site
        result[site]["source_format"] = "openghg"
        result[site]["dataset_source"] = "synthetic"
        result[site]["instrument"] = instrument
        result[site]["info_metadata"] = kwargs
        result[site]["update_mismatch"] = "metadata"

        
        if site.upper() in multi_obs.obs:
            attrs = multi_obs.obs[site.upper()].data.attrs
            meta = multi_obs.obs[site.upper()].metadata
#            result[site]["species"] = meta.get("species")
            result[site]["inlet"] = meta.get("inlet")
            result[site]["network"] = "synthetic" # meta.get("network") or attrs.get("network", "synthetic")
            result[site]["calibration_scale"] = meta.get("scale") or meta.get("calibration_scale", "synthetic")

    return result

# %% [markdown]
# #### Step 1: get flux

# %%
flux_obj = get_flux(species="sf6", domain="europe", source="edgar-annual-total")
flux = flux_obj.data

# %% [markdown]
# #### Step 2: get mod obs dict and multi obs/fp
#
# - Get ini file
# - pass start and end date as kwargs

# %%
ini_files[0]

# %%
mod_obs_dict, multi_obs, multi_fp = make_mod_obs_dict_and_multi_obs_fp(ini_files[0], flux, start_date="2015-01-01", end_date="2025-01-01")

# %% [markdown]
# #### Step 3: make baselines
#
# We need to load all merged data, make baseline dicts for each year, then concat

# %%
data_path = Path("sf6_model_testing_data/")
search_merged_data(data_path) 

# %%
all_merged_data = [load_merged_data(data_path, start_date=start_date, species="sf6") for start_date in search_merged_data(data_path).start_date]

# %%
baseline_dicts = [make_baseline_dict(merged_data) for merged_data in all_merged_data]

# %%
from itertools import chain

all_sites = set(chain.from_iterable([tuple(bld.keys()) for bld in baseline_dicts]))
all_sites

# %%
combined_baseline_dict = {}

for site in all_sites:
    data = []
    for baseline_dict in baseline_dicts:
        if site in baseline_dict:
            data.append(baseline_dict[site])
    combined_baseline_dict[site] = xr.concat(data, dim="time")


# %%
combined_baseline_dict["MHD"]

# %% [markdown]
# #### Step 4: make synth obs
#
# Combine mod obs and baseline, plus add noise
#
# We can make multiple versions with different levels of noise, and versions with biases.

# %%
synth_obs_args1 = {   
    "std_cutoff": 0.7, 
    "std_resample": "8D", 
    "std_scaling": 0.0,
    "seed": 123456789,
    "bias": None,
}

synth_obs1 = make_synth_obs(mod_obs_dict=mod_obs_dict, baseline_dict=combined_baseline_dict, multi_obs=multi_obs, **synth_obs_args1)

# %%
synth_obs1["MHD"]

# %% [markdown]
# #### Step 5: standardise
#
# we need to make args for standardise_surface, save to a temporary netcdf, then standardise

# %%
std_args1 = make_standardise_args(sites=list(synth_obs1.keys()), multi_obs=multi_obs, instrument="edgar-annual-total_mod-baseline_no-noise_no-bias", **synth_obs_args1)

# %%
all_sites

# %%
for k, v in std_args1["CBW"].items():
    print(k, v, type(v))
    if isinstance(v, dict):
        for k1, v2 in v.items():
            print("\t", k1, type(v2))

# %%
for k, v in synth_obs1["JFJ"].attrs.items():
    if not choose_key(k):
        continue
    print(k, v, type(v))

# %%
with tempfile.NamedTemporaryFile() as f:
    ds = synth_obs1["JFJ"].copy()
    ds.noise.attrs.pop("sigma", None)
    ds.to_netcdf(f, engine="h5netcdf")
    # %debug standardise_surface(filepath=f.name, store="sf6_testing_store", if_exists="new", **std_args1["JFJ"])

# %% jupyter={"outputs_hidden": true}
import tempfile

for site, ds in synth_obs1.items():
    print("\n", site)
    ds = ds.copy()
    with tempfile.NamedTemporaryFile() as f:
        ds.noise.attrs.pop("sigma", None)
        def maybe_str(x):
            try:
                return str(x)
            except Exception:
                return x
        ds.attrs = {k: maybe_str(v) for k, v in ds.attrs.items() if choose_key(k)}
        if "sampling_period" not in ds.attrs:
            ds.attrs["sampling_period"] = "3600.0"
#        for key in std_args1[site]:
#            ds.attrs.pop(key, None)
        ds.to_netcdf(f, engine="h5netcdf")
        try:
            standardise_surface(filepath=f.name, store="sf6_testing_store", if_exists="new", **std_args1[site])
        except Exception as e:
            print("Error:", e)
            print("Attrs:\n", ds.attrs)
                     

# %% [markdown]
# Check progress

# %%
obs_res = search_surface(species="sf6", store="sf6_testing_store")

# %%
obs_res.results[["site", "network", "latest_version", "object_store", "uuid"]]

# %%
from openghg.dataobjects import data_manager

# %%
# data_manager?

# %%
dm = data_manager(data_type="surface", store="sf6_testing_store", species="sf6")

# %%
dm.metadata

# %%
# dm.delete_datasource?

# %%
#to_delete = [k for k, v in dm.metadata.items() if v["network"] != "synthetic"]
for k in to_delete:
    dm.delete_datasource(k)


# %% [markdown]
# ### Making more variations

# %%
def maybe_str(x):
    try:
        return str(x)
    except Exception:
        return x


def standardise_synth_obs(synth_obs: dict, std_args: dict, **kwargs):
    for site, ds in synth_obs.items():
        print("\nStandardising:", site)
        ds = ds.copy()
        with tempfile.NamedTemporaryFile() as f:
            ds.noise.attrs.pop("sigma", None)

            ds.attrs = {k: maybe_str(v) for k, v in ds.attrs.items() if choose_key(k)}
            if "sampling_period" not in ds.attrs:
                ds.attrs["sampling_period"] = "3600.0"
    
            ds.to_netcdf(f, engine="h5netcdf")
            try:
                standardise_surface(filepath=f.name, store="sf6_testing_store", **std_args[site], **kwargs)
            except Exception as e:
                print("Error:", e)



# %%
synth_obs_args2 = {   
    "std_cutoff": 0.7, 
    "std_resample": "8D", 
    "std_scaling": 0.3,
    "seed": 123456789,
    "bias": None,
}

synth_obs2 = make_synth_obs(mod_obs_dict=mod_obs_dict, baseline_dict=combined_baseline_dict, multi_obs=multi_obs, **synth_obs_args2)
std_args2 = make_standardise_args(sites=list(synth_obs2.keys()), multi_obs=multi_obs, instrument="edgar-annual-total_mod-baseline_0.3-noise_no-bias", **synth_obs_args2)

standardise_synth_obs(synth_obs2, std_args2)


                     

# %%
obs_res = search_surface(species="sf6", store="sf6_testing_store")
obs_res.results.instrument.unique()

# %%
for scale in [0.6]: #[0.0, 0.3, 0.9, 1.2, 2.0]:
    for bias in [None, -0.03, -0.1, 0.03, 0.1]:
        synth_obs_args = {   
            "std_cutoff": 0.7, 
            "std_resample": "8D", 
            "std_scaling": scale,
            "seed": 123456789,
            "bias": bias,
        }
        if bias is not None and scale != 0.0:
            instrument = f"edgar-annual-total_mod-baseline_{scale:.1f}-noise_{bias:.2f}-bias"
        elif bias is not None:
            instrument = f"edgar-annual-total_mod-baseline_no-noise_{bias:.2f}-bias"
        elif scale != 0.0:
            instrument = f"edgar-annual-total_mod-baseline_{scale:.1f}-noise_no-bias"
        else:
            instrument = f"edgar-annual-total_mod-baseline_no-noise_no-bias"

        synth_obs = make_synth_obs(mod_obs_dict=mod_obs_dict, baseline_dict=combined_baseline_dict, multi_obs=multi_obs, **synth_obs_args)
        std_args = make_standardise_args(sites=list(synth_obs.keys()), multi_obs=multi_obs, instrument=instrument, **synth_obs_args)
        standardise_synth_obs(synth_obs, std_args, if_exists="new")

# %%
to_delete = obs_res.results.loc[(obs_res.results.timestamp < "2025-10-10 19:33:49")].uuid

# %%
mhd_obs = get_obs_surface(species="sf6", store="sf6_testing_store", site="mhd", instrument="edgar-annual-total_mod-baseline_06-noise_no-bias")
mhd_obs_no_noise = get_obs_surface(species="sf6", store="sf6_testing_store", site="mhd", instrument="edgar-annual-total_mod-baseline_no-noise_no-bias")

# %%
fig, ax = plt.subplots()

mhd_obs.data.mf.sel(time=slice("2018-01-01", "2018-02-01")).plot(ax=ax, label="synth",alpha=0.4)
#mhd_obs_no_noise.data.mf.sel(time=slice("2018-01-01", "2018-02-01")).plot(ax=ax, label="synth, no noise",alpha=0.4)
multi_obs.obs["MHD"].data.mf.sel(time=slice("2018-01-01", "2018-02-01")).plot(ax=ax, label="true", alpha=0.4)
ax.legend()

# %% [markdown]
# # Storing more fluxes, baselines

# %%
sf6_results_path = Path("/group/chem/acrg/PARIS_results_sharing/sf6_for_brendan/")
# !ls -ls {sf6_results_path}

# %%
# sf6_res_files = !ls {sf6_results_path} | grep -v "concentrations"
sf6_res_files

# %%
file_no = -2
ds = xr.open_dataset(sf6_results_path / sf6_res_files[file_no], engine="h5netcdf")

# %%
ds.country.values

# %%
print(ds.country_flux_total_posterior.sel(country=["CHE", "DEU"]).to_series().unstack().iloc[-10:,:])

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
year = 2018
fig.suptitle(sf6_res_files[file_no][:-3] + f" {year}")

lat_slice = slice(37, None)
lon_slice = slice(-14, 25)

lat_min, lat_max = lat_slice.start, lat_slice.stop
lon_min, lon_max = lon_slice.start, lon_slice.stop

#vmin, vmax = -39, -26
vmin, vmax = 0, 2.5e-12

(ds.flux_total_prior).sel(time=f"{year}-07-01", method="nearest").sel(latitude=lat_slice, longitude=lon_slice).plot(ax=axs[0], vmin=vmin, vmax=vmax)
axs[0].set_title("prior")

(ds.flux_total_posterior).sel(time=f"{year}-07-01", method="nearest").sel(latitude=lat_slice, longitude=lon_slice).plot(ax=axs[1], vmin=vmin, vmax=vmax)
axs[1].set_title("posterior")

for ax in axs.flat:
    world.boundary.plot(ax=ax, linewidth=0.6, edgecolor='white')  # or .plot(facecolor='none')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# %%
sf6_rhime_path = Path("/group/chem/acrg/PARIS_inversions/sf6/")
# !ls {sf6_rhime_path}

# %%
# !ls {sf6_rhime_path / "RHIME_NAME_EUROPE_FLAT_PARISNID2026_sf6_yearly"}

# %%
year = 2020
ds2 = xr.open_dataset(sf6_rhime_path / "RHIME_NAME_EUROPE_FLAT_PARISNID2026_sf6_yearly" / f"SF6_EUROPE_PARIS_conc_{year}-01-01.nc")

# %%
ds2

# %%
ds2.sitenames.values

# %%
from openghg.retrieve import *

obs_res = search_surface(species="sf6", site="KIT")

# %%

# %%
obs = obs_res.retrieve_all()
obs[1].data.sf6.time[:20]

# %%
ds3 = xr.open_mfdataset(str(sf6_rhime_path / "RHIME_NAME_EUROPE_FLAT_PARISNID2026_sf6_yearly" / "SF6_EUROPE_PARIS_flux_*-01-01.nc"))
print(ds3.country_flux_total_posterior.sel(country=["CHE", "DEU"]).to_series().unstack())

# %%
ds3.country_flux_total_posterior.sel(country=[c for c in ds3.country.values if len(c) == 3]).to_series().describe()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

fig.suptitle(f"RHIME_NAME_EUROPE_FLAT_PARISNID2026_sf6_yearly {year}")

lat_slice = slice(37, None)
lon_slice = slice(-14, 25)

lat_min, lat_max = lat_slice.start, lat_slice.stop
lon_min, lon_max = lon_slice.start, lon_slice.stop

#vmin, vmax = -39, -26
vmin, vmax = 0, 10e-13

(ds2.flux_total_prior).sel(latitude=lat_slice, longitude=lon_slice).plot(ax=axs[0], vmin=vmin, vmax=vmax)
axs[0].set_title("prior")

(ds2.flux_total_posterior).sel(latitude=lat_slice, longitude=lon_slice).plot(ax=axs[1], vmin=vmin, vmax=vmax)
axs[1].set_title("posterior")

for ax in axs.flat:
    world.boundary.plot(ax=ax, linewidth=0.6, edgecolor='white')  # or .plot(facecolor='none')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# %% [markdown]
# ## Baselines from InTEM and ELRIS

# %%
baseline_dicts

# %%
# sf6_conc_files = !ls {sf6_results_path} | grep "concentrations"
sf6_conc_files

# %%
conc1 = xr.open_dataset(sf6_results_path / sf6_conc_files[-2])


# %%
def fix_flux_index_coord(ds):
    mindex = pd.MultiIndex.from_arrays([ds.platform.values[ds.number_of_identifier.values.astype(int)], ds.time.values], names=["platform", "time"])
    ds = ds.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, "index"))
    return ds

conc1 = fix_flux_index_coord(conc1)

# %%
conc1.stdev_mf_total.sel(platform="MHD").plot()

# %%
