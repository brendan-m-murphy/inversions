from pathlib import Path

import numpy as np
import pandas as pd
import pint
import xarray as xr
from iris.analysis import AreaWeighted

from openghg.util import cf_ureg, find_domain
from openghg.util import molar_mass as _molar_mass
from openghg.standardise.meta import define_species_label


cf_ureg.define(
    "@alias kilogram = kg"
)  # TODO: case-insensitive mode causes problems with all prefix-gram units


def molar_mass(species: str) -> pint.Quantity:
    """Get molar mass of species with pint units of g/mol."""
    spec_label = define_species_label(species)[
        0
    ]  # this is done in EDGAR parsing code... why do we need it?
    mm = _molar_mass(spec_label)
    return mm * cf_ureg.g / cf_ureg.mol


def zeros_from_domain(domain: str, short_dim_names: bool = False) -> xr.DataArray:
    """Make a DataArray of zeros with lat/lon coords from given domain."""
    lat_out, lon_out = find_domain(domain)
    nlat = len(lat_out)
    nlon = len(lon_out)
    dims = ["lat", "lon"] if short_dim_names else ["latitude", "longitude"]
    da_out = xr.DataArray(np.zeros((nlat, nlon)), coords=[lat_out, lon_out], dims=dims)
    da_out[dims[0]].attrs = {"units": "degrees", "standard_name": "latitude"}
    da_out[dims[1]].attrs = {"units": "degrees", "standard_name": "longitude"}

    return da_out


def _regrid_2d(
    input_data: xr.DataArray,
    in_lat_coord: str = "lat",
    in_lon_coord: str = "lon",
    domain: str = "EUROPE",
) -> xr.DataArray:

    cube_out = zeros_from_domain(domain).to_iris()
    cube_out.coord("latitude").guess_bounds()
    cube_out.coord("longitude").guess_bounds()

    cube_in = input_data.rename({in_lat_coord: "lat", in_lon_coord: "lon"}).to_iris()
    cube_in.coord("latitude").guess_bounds()
    cube_in.coord("longitude").guess_bounds()

    cube_regridded = cube_in.regrid(cube_out, AreaWeighted(mdtol=1.0))

    result = xr.DataArray.from_iris(cube_regridded).rename(
        {"latitude": "lat", "longitude": "lon"}
    )
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


def transform_edgar_flux_file(
    filepath: str | Path,
    domain: str,
    species: str | None = None,
    year: int | None = None,
    month: int = 1,
) -> xr.Dataset:
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
        ds_regridded = ds_regridded.expand_dims(
            {"time": [pd.Timestamp(year=year, month=month, day=1)]}
        )

    if species is None:
        return ds_regridded

    molmass = molar_mass(species)

    with xr.set_options(keep_attrs=True):
        ds_regridded = (ds_regridded.pint.quantify() / molmass).pint.to("mol/m2/s")

    return ds_regridded.pint.dequantify()
