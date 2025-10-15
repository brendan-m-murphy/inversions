from functools import partial
import logging
from pathlib import Path
import re
from typing import Any, Callable, ChainMap, Iterable, cast, Literal, Mapping, overload
from typing_extensions import Self
import warnings

import numcodecs
from numcodecs import Blosc, Quantize
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from openghg.analyse import ModelScenario
from openghg.dataobjects import BoundaryConditionsData, FluxData, FootprintData, ObsData
from openghg.retrieve import get_obs_surface, get_footprint, get_flux, get_bc
from openghg.types import SearchError
from openghg.util import extract_float, split_function_inputs

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.basis.algorithms import quadtree_algorithm, weighted_algorithm
from openghg_inversions.hbmcmc.run_hbmcmc import hbmcmc_extract_param
from openghg_inversions.inversion_data.get_data import add_obs_error, convert_to_list
from openghg_inversions.inversion_data.getters import (
    get_flux_data,
    get_footprint_data,
    get_obs_data,
)
from openghg_inversions.inversion_data.scenario import merged_scenario_data
from openghg_inversions.inversion_data.serialise import _save_merged_data, _make_merged_data_name


read_ini = partial(hbmcmc_extract_param, mcmc_type="fixedbasisMCMC", print_param=False)

# TODO: add functions for parsing data params from ini file into data objects
# then make constructors for MultiObs and MultiFootprint use these
# with a class method for parsing these data objects
#
# As it is, the arguments for MultiObs.__init__ are unwieldy...

class MultiObs(ObsData):
    def __init__(
        self,
        species: str,
        start_date: str,
        end_date: str,
        sites: list[str],
        inlets: list[str | None],
        obs_data_levels: list[str | None],
        instruments: list[str | None],
        averaging_periods: list[str | None],
        calibration_scale: str | None = None,
        store=None,
        obs_store=None,
        **kwargs,
    ) -> None:
        self.species = species
        self.start_date = start_date
        self.end_date = end_date
        self._sites = sites
        self._inlets = inlets

        self.store = obs_store or store

        self.average = averaging_periods

        valid_kwargs, _ = split_function_inputs(kwargs, get_obs_surface)
        self.kwargs = valid_kwargs

        # TODO: handle case of site with multiple inlets

        self.obs = {}
        self.sites = []
        self.inlets = []

        keep_variables = [
            f"{species}",
            f"{species}_variability",
            f"{species}_repeatability",
            f"{species}_number_of_observations",
            "inlet",  # needed if multiple inlets combined
            "inlet_height",  # sometimes needed if inlet='multiple' (may be outdated soon)
        ]
        warnings.warn(f"Dropping all variables besides {keep_variables}")

        # will set these after we fetch the first dataset
        target_units = None

        for site, inlet, avg, data_level, instrument in zip(self._sites, self._inlets, self.average, obs_data_levels, instruments):
            try:
                obs = get_obs_surface(
                    species=self.species,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    site=site,
                    inlet=inlet,
                    data_level=data_level,
                    instrument=instrument,
                    average=avg,
                    store=self.store,
                    target_units=target_units,
                    calibration_scale=calibration_scale,
                    keep_variables=keep_variables,
                    **self.kwargs,
                )
            except Exception as e:
                print(f"Couldn't get obs for site {site} and inlet {inlet} from store {self.store}: {e}")
            else:
                self.obs[site] = obs
                self.sites.append(site)
                self.inlets.append(inlet)

                if target_units is None:
                    target_units = {dv: obs.data.mf.attrs["units"] for dv in ("mf", "mf_repeatability", "mf_variability")}

                if calibration_scale is None:
                    calibration_scale = obs.metadata.get("calibration_scale")

        if not self.obs:
            raise SearchError(
                f"No obs. found for {self.species} at sites {self._sites} in store {self.store}"
            )

        self.calibration_scale = calibration_scale
        self.units = target_units

        self._combined_ds = xr.concat(
            [x.data.expand_dims(site=[site]) for site, x in self.obs.items()], dim="site"
        )

    def combined_ds(self) -> xr.Dataset:
        return self._combined_ds

    @property
    def data(self) -> xr.Dataset:
        return self._combined_ds

    @property
    def metadata(self) -> dict:
        return {"species": self.species, "site": None, "inlet": None}


class MultiFootprint:
    def __init__(
        self,
        domain,
        start_date,
        end_date,
        fp_heights,
        sites,
        met_model: list[str | None],
        fp_species="inert",
        store=None,
        footprint_store=None,
        model=None,
        obs_data: dict | None = None,
        obs_sites: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.domain = domain
        self.species = fp_species
        self.start_date = start_date
        self.end_date = end_date
        self._sites = sites
        self._inlets = fp_heights
        self.store = footprint_store or store
        self.model = model
        self._met_model = met_model

        valid_kwargs, _ = split_function_inputs(kwargs, get_footprint)
        self.kwargs = valid_kwargs

        # need more robust way to do this...
        if "species" in self.kwargs:
            del self.kwargs["species"]

        obs_data = obs_data or {}

        # TODO: cover case of a site with multiple inlets

        self.footprints = {}
        self.sites = []
        self.inlets = []
        self.met_model = []
        for site, inlet, met_mod in zip(self._sites, self._inlets, self._met_model):
            if obs_sites is not None and site not in obs_sites:
                continue
            try:
                fp = get_footprint_data(
                    domain=self.domain,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fp_species=self.species,
                    site=site,
                    fp_height=inlet,
                    model=self.model,
                    met_model=met_mod,
                    obs_data=obs_data.get(site),
                    stores=self.store,
                )
                # fp = get_footprint(
                #     domain=self.domain,
                #     start_date=self.start_date,
                #     end_date=self.end_date,
                #     species=self.species,
                #     site=site,
                #     inlet=inlet,
                #     model=self.model,
                #     met_model=self.met_model,
                #     store=self.store,
                #     **self.kwargs,
                # )
            except Exception as e:
                print(
                    f"Couldn't get footprint for site {site} and inlet {inlet} from store {self.store}: {e}"
                )
            else:
                self.footprints[site] = fp
                self.sites.append(site)
                self.inlets.append(inlet)
                self.met_model.append(met_mod)

        self._combined_ds = xr.concat(
            [x.data.expand_dims(site=[site]) for site, x in self.footprints.items()], dim="site"
        )

    def combined_ds(self) -> xr.Dataset:
        return self._combined_ds

    @property
    def data(self) -> xr.Dataset:
        return self._combined_ds

    @property
    def metadata(self) -> dict:
        return {"species": self.species, "domain": self.domain, "inlet": None}


def data_processing(
    species: str,
    sites: list | str,
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    obs_data_level: list[str | None] | str | None = None,
    platform: list[str | None] | str | None = None,
    inlet: list[str | None] | str | None = None,
    instrument: list[str | None] | str | None = None,
    calibration_scale: str | None = None,
    met_model: list[str | None] | str | None = None,
    fp_model: str | None = None,
    fp_height: list[str | None | Literal["auto"]] | Literal["auto"] | str | None = None,
    fp_species: str | None = None,
    emissions_name: list | None = None,
    use_bc: bool = True,
    bc_input: str | None = None,
    bc_store: str | None = None,
    obs_store: str | list[str] | None = None,
    footprint_store: str | list[str] | None = None,
    emissions_store: str | None = None,
    add_averaging_error: bool = True,
) -> xr.Dataset:
    sites = [site.upper() for site in sites]

    # Convert 'None' args to list
    nsites = len(sites)
    inlet = convert_to_list(inlet, nsites, "inlet")
    instrument = convert_to_list(instrument, nsites, "instrument")
    fp_height = convert_to_list(fp_height, nsites, "fp_height")
    obs_data_level = convert_to_list(obs_data_level, nsites, "obs_data_level")
    met_model = convert_to_list(met_model, nsites, "met_model")
    averaging_period = convert_to_list(averaging_period, nsites, "averaging_period")
    platform = convert_to_list(platform, nsites, "platform")

    # Get flux data
    if emissions_name is None:
        raise ValueError("`emissions_name` must be specified")

    flux_dict = get_flux_data(
        sources=emissions_name,
        species=species,
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        store=emissions_store,
    )

    # Get BC data
    if use_bc is True:
        try:
            bc_data = get_bc(
                species=species,
                domain=domain,
                bc_input=bc_input,
                start_date=start_date,
                end_date=end_date,
                store=bc_store,
            )
        except SearchError as e:
            raise SearchError("Could not find matching boundary conditions.") from e
    else:
        bc_data = None

    multi_obs = MultiObs(
        species=species,
        start_date=start_date,
        end_date=end_date,
        sites=sites,
        inlets=inlet,
        instruments=instrument,
        obs_data_levels=obs_data_level,
        calibration_scale=calibration_scale,
        obs_store=obs_store,
        averaging_periods=averaging_period,
    )

    multi_fp = MultiFootprint(
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        fp_heights=fp_height,
        sites=sites,
        footprint_store=footprint_store,
        model=fp_model,
        met_model=met_model,
        obs_data=multi_obs.obs,
        obs_sites=multi_obs.sites,
    )


    split_by_sectors = len(flux_dict) > 1

    # create fp_all dict
    fp_all = {}
    fp_all[".species"] = species.upper()

    if isinstance(multi_obs.units.get("mf"), str):
        fp_all[".units"] = extract_float(multi_obs.units.get("mf"))
    else:
        fp_all[".units"] = multi_obs.units.get("mf")

    fp_all[".scales"] = {site: multi_obs.calibration_scale for site in multi_fp.sites}

    fp_all[".flux"] = flux_dict
    if use_bc:
        fp_all[".bc"] = bc_data

    for site in multi_obs.sites:
        model_scenario = ModelScenario(
            obs=multi_obs.obs[site],
            footprint=multi_fp.footprints[site],
            flux=flux_dict,
            bc=bc_data,
        )
        fp_all[site.upper()] = model_scenario.footprints_data_merge(
            calc_fp_x_flux=True,
            split_by_sectors=split_by_sectors,
            calc_bc_sensitivity=True,
            cache=False,
        )

    return fp_all


def rechunk_ds(ds: xr.Dataset, time: int = 240, **kwargs) -> xr.Dataset:
    default_chunks = {"lat": 293, "lon": 391, "height": 20, "bc_region": 4}
    default_chunks.update(kwargs)

    chunks = {dim: default_chunks.get(dim) for dim in ds.dims if dim != "time"}
    if ds.sizes.get("time", 0) > time:
        chunks["time"] = time
    elif "time" in ds.dims:
        chunks["time"] = ds.sizes["time"]
    if "region" in ds.dims:
        chunks["region"] = ds.sizes["region"]
    return ds.chunk(chunks)


def fp_all_to_datatree(fp_all: dict, name: str | None = None, rechunk: bool = True, chunks: dict | None = None) -> xr.DataTree:
    scenario = {k: v for k, v in fp_all.items() if not k.startswith(".")}
    attrs = {k.removeprefix("."): v for k, v in fp_all.items() if k in [".species", ".scales", ".units"]}
    aux_data = {k.removeprefix("."): v for k, v in fp_all.items() if (k not in scenario) and (k.removeprefix(".") not in attrs)}

    # nest flux (this can be done automatically from nested dict according to xarray docs, but
    # it doesn't work for me... maybe I need to update xarray
    #aux_data["/flux"] = xr.DataTree.from_dict({k: v.data for k, v in aux_data["flux"].items()})
    #del aux_data["flux"]

    # "flux" as a dataset... this might not work if we're mixing high/low frequency fluxes
    # but it works for multiple sectors
    aux_data["flux"] = xr.Dataset({k: v.data.flux for k, v in aux_data["flux"].items()})

    # get data from BoundaryConditionsData object... maybe we
    # should put metadata in global attrs for this group?
    aux_data["bc"] = aux_data["bc"].data

    # fix issue with units for time coord
    if "units" in aux_data["bc"].coords["time"].attrs:
        del aux_data["bc"].coords["time"].attrs["units"]

    # add basis as data variable at root?
    basis = aux_data.pop("basis", None)

    # add basis within group?
    # aux_data["basis"] = aux_data["basis"].rename("basis").to_dataset()

    dt_dict = aux_data.copy()
    dt_dict["/scenario"] = xr.DataTree.from_dict({k: v for k, v in scenario.items()})

    dt = xr.DataTree.from_dict(dt_dict)
    dt.attrs = attrs

    if basis is not None:
        dt["basis"] = basis

    if name is not None:
        dt.name = name

    if rechunk or chunks:
        chunks = chunks or {}
        dt = dt.map_over_datasets(rechunk_ds, kwargs=chunks)

    return dt


def datatree_to_fp_all(dt: xr.DataTree) -> dict:
    d = dt.to_dict()
    result = {}
    result[".flux"] = {dv: FluxData(data=d["/flux"][[dv]], metadata={}) for dv in d["/flux"].data_vars}
    result[".bc"] = BoundaryConditionsData(data=d["/bc"], metadata={})
    if "basis" in d["/"]:
        result[".basis"] = d["/"].basis
    result[".species"] = dt.attrs.get("species")
    result[".units"] = dt.attrs.get("units")
    result[".scales"] = dt.attrs.get("scales")
    for k, v in d.items():
        if k.startswith("/scenario/"):
            site = k.split("/")[-1]
            result[site] = v
    return result


def filter_data_vars(ds: xr.Dataset, cond: Callable[[str], bool]) -> xr.Dataset:
    """Function to filter data variables by a condition."""
    keep_dvs = [dv for dv in ds.data_vars if cond(str(dv))]
    return ds[keep_dvs]


def store_data_var(dv: str) -> bool:
    """Condition for filtering out data variables we don't want to store."""
    return dv not in ("fp", "mf_mod", "bc_mod") and "particle" not in dv


def set_encoding(ds: xr.Dataset, compressor: Blosc | None = None, overwrite: bool = False) -> xr.Dataset:
    compressor = compressor or Blosc("zstd", 5, Blosc.SHUFFLE)
    for dv in ds.data_vars:
        if not ds[dv].encoding or overwrite:
            ds[dv].encoding["compressors"] = [compressor]
    return ds


def save_merged_data(
    dt: xr.DataTree,
    merged_data_dir: str | Path,
    species: str | None = None,
    start_date: str | None = None,
    output_name: str | None = None,
    merged_data_name: str | None = None,
    zip_zarr: bool = False,
) -> Path:
    """Save DataTree with merged data to `merged_data_dir`.

    The name of the output file can be specified using `merged_data_name`, or
    a standard name will be created given `species`, `start_date`, and `output_name`.

    If `merged_data_name` is not given, then `species`, `start_date`, and `output_name` must be provided.

    The data is stored in a zarr ZipStore. The path to this store is returned, and the data can be
    loaded by calling `xr.open_zarr` on this path.

    Args:
        dt: DataTree of merged data to save
        merged_data_dir: path to directory where merged data will be saved
        species: species of inversion
        start_date: start date of inversion period
        output_name: output name parameter used for inversion run
        merged_data_name: name to use for saved data.

    Returns:wd
        Path: path to stored data.
    """
    if merged_data_name is None:
        if any(arg is None for arg in [species, start_date, output_name]):
            raise ValueError(
                "If `merged_date_name` isn't given, then "
                "`species`, `start_date`, and `output_name` must be provided."
            )
        merged_data_name = _make_merged_data_name(species, start_date, output_name)  # type: ignore

    if isinstance(merged_data_dir, str):
        merged_data_dir = Path(merged_data_dir)

    ext = ".zarr.zip" if zip_zarr else ".zarr"
    output_path = merged_data_dir / (merged_data_name + ext)

    # make sure we have encoding
    dt = dt.map_over_datasets(set_encoding, kwargs={"overwrite": True})

    if zip_zarr:
        with zarr.ZipStore(output_path, mode="w") as store:
            dt.to_zarr(store, mode="w", compute=True)
    else:
        dt.to_zarr(output_path, mode="w", compute=True)

    return output_path


def create_merged_data(params: dict, chunks: dict | None = None) -> xr.DataTree:
    data_params, _ = split_function_inputs(params, data_processing)
    fp_all = data_processing(**data_params)
    return fp_all_to_datatree(fp_all, chunks=chunks)


def create_and_save_merged_data(ini_file: str | Path, merged_data_dir: str | Path, output_name: str, chunks: dict | None = None) -> Path:
    params = read_ini(ini_file)
    dt = create_merged_data(params, chunks)

    # filter out variables we don't need for basis functions and filtering
    dt = dt.map_over_datasets(filter_data_vars, store_data_var)

    return save_merged_data(
        dt,
        merged_data_dir,
        species=params.get("species"),
        start_date=params.get("start_date"),
        output_name=output_name
    )


def _parse_merged_data_name(name: str) -> dict[str, str]:
    """Extract species, start date, and output name from merged data file name."""
    merged_data_name_pat = re.compile(r"(?P<species>[a-zA-Z0-9]+)_(?P<start_date>[\d-]+)_(?P<output_name>.+)_merged-data")
    m = merged_data_name_pat.search(name)
    if m is None:
        raise ValueError(f"Merged data name {name} could not be parsed.")
    return m.groupdict()


def search_merged_data(merged_data_dir: str | Path) -> pd.DataFrame:
    merged_data_dir = Path(merged_data_dir)

    result = []
    for path in merged_data_dir.iterdir():
        try:
            info = _parse_merged_data_name(path.name)
        except ValueError:
            continue
        else:
            info["path"] = path
            result.append(info)
    return pd.DataFrame(result).sort_values(["species", "output_name", "start_date"]).reset_index(drop=True)


def load_merged_data(merged_data_dir: str | Path, start_date: str, species: str | None = None, output_name: str | None = None) -> xr.DataTree:
    df = search_merged_data(merged_data_dir)
    filt = df.start_date.str.contains(start_date)

    if species:
        filt = filt & (df.species == species)

    if output_name:
        filt = filt & (df.output_name == output_name)

    df_filt = df.loc[filt]

    if df_filt.empty:
        raise ValueError(f"Merged data with output_name={output_name}, start_date={start_date}, and species={species} not found in {merged_data_dir}.")

    row = df_filt.iloc[0]
    return xr.open_datatree(row.path, engine="zarr", chunks={})
