"""Functions for plotting inversion outputs.

In particular, helpers for working with fluxy.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from fluxy.io import edit_vars_and_attributes, read_config_files


def _handle_multi_path_and_glob_netcdf(
    path: str | Path | list[str | Path], glob: str | None = None
) -> list[Path]:
    glob = glob or "*.nc"
    if not isinstance(path, list):
        path = Path(path)
        path = [path] if "nc" in path.suffix.lower() else list(path.glob(glob))

    path = [Path(p) for p in path]

    return sorted(path, key=lambda x: x.name)  # type: ignore


def open_mf_paris_flux(flux_path: str | Path, glob: str = "*flux*.nc") -> xr.Dataset:
    """Open multiple flux output files."""
    flux_path = Path(flux_path)
    return xr.open_mfdataset(str(flux_path / glob))


def open_mf_paris_conc(
    conc_path: str | Path | list[str | Path], glob: str = "*conc*.nc"
) -> xr.Dataset:
    """Open multiple concentration output files."""
    paths = _handle_multi_path_and_glob_netcdf(conc_path, glob)
    # open netCDFs and convert nsite dim + sitenames coord to a dimension coordinate
    # so we can concatenate
    datasets = [
        xr.open_dataset(p).rename(sitenames="site").swap_dims(nsite="site")
        for p in paths
    ]
    ds = xr.concat(datasets, dim="time", join="outer")

    # convert back to template format before returning
    return ds.swap_dims(site="nsite").rename(site="sitenames")


def restore_mindex(
    data: xr.Dataset,
    index_var: str = "index",
    platform_var: str = "platform",
    indicator_var: str = "number_of_identifier",
) -> xr.Dataset:
    platform_arr = data[platform_var].values[data[indicator_var].values.astype("int")]
    mindex = pd.MultiIndex.from_arrays([platform_arr, data.time.values])
    return data.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, index_var))
