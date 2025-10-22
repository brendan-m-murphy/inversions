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
# # Tests on synthetic data

# %% [markdown]
# ## Existing models, EDGAR synthetic data
#
# ### Plan
#
# - create merged data with all variations of obs (with a "instrument" or "experiment" dimension)
#   - this should do basis functions before filtering, like current setup
#   - use same filtering options
#   - start with merged data with fp_x_flux
#   - resample synth obs to 4h
# - make code to load data, create inputs, build model, and save outputs
# - make model config settings
# - configure output locations for each config

# %% [markdown]
# ### Making merged data with all synth obs
#
# The merged data w/o basis and filtering is stored by year, it is probably easiest to match this and use a different "output_name".

# %%
# %run inversions_experimental_code/data_functions.py

# %% [markdown]
# #### Getting all synth obs

# %%
from openghg.retrieve import search_surface, get_obs_surface

# %%
res = search_surface(species="sf6", store="sf6_testing_store")

# %%
res.results.groupby("site").instrument.count()

# %%
sorted(res.results.instrument.unique())

# %% [markdown]
# We'll parse these for noise and bias values as floats, then add these as dimensions.

# %%
sites = sorted(res.results.site.unique())
instruments = sorted(res.results.instrument.unique())

instr_to_dims = {}
for instr in instruments:
    noise, bias = instr.split("_")[-2:]
    noise_val = (
        0.0 if noise.startswith("no") else float(noise.removesuffix("-noise")) / 10
    )
    bias_val = 0.0 if bias.startswith("no") else float(bias.removesuffix("-bias")) / 100
    instr_to_dims[instr] = {"noise": [noise_val], "bias": [bias_val]}

# %%
all_obs = []
other_args = {
    "start_date": "2015-01-01",
    "end_date": "2025-01-01",
    "inlets": [None] * len(sites),
    "obs_data_levels": [None] * len(sites),
    "averaging_periods": ["4h"] * len(sites),
}
for instr in instruments:
    multi_obs = MultiObs(
        species="sf6", sites=sites, instruments=[instr] * len(sites), **other_args
    )
    all_obs.append(multi_obs.data.expand_dims(instr_to_dims[instr]))

# %%
combined_obs = xr.combine_by_coords(all_obs, combine_attrs="drop_conflicts")


# %%
combined_obs

# %% [markdown]
# #### Loading merged data
#
# Once data is in, we need to compute basis functions.
#
# We maybe also need this data for computing outputs?

# %%
from pathlib import Path

data_path = Path("/user/work/bm13805/sf6_model_testing_data/")
md_res = search_merged_data(data_path)
md_res

# %%
all_merged_data = [
    load_merged_data(merged_data_dir=data_path, species="sf6", start_date=start_date)
    for start_date in md_res.start_date
]

# %%
all_merged_data[0]

# %% [markdown]
# #### Making and applying basis functions

# %%
from openghg_inversions.basis.algorithms import weighted_algorithm

# %%
# weighted_algorithm?

# %%
intem_regions = xr.open_dataset(
    "/user/work/bm13805/openghg_inversions/openghg_inversions/basis/outer_region_definition_EUROPE.nc"
).region
_, intem_regions = xr.align(
    all_merged_data[0].flux.to_dataset(), intem_regions, join="override"
)


# %%
def mean_fp_x_flux(dt: xr.DataTree, mask: xr.DataArray | None = None) -> xr.DataArray:
    if mask is not None:
        ds_list = [
            v.fp_x_flux.where(mask, drop=True).expand_dims({"site": [k]})
            for k, v in dt.scenario.items()
        ]
    else:
        ds_list = [
            v.fp_x_flux.expand_dims({"site": [k]}) for k, v in dt.scenario.items()
        ]
    return xr.concat(ds_list, dim="site").mean(["site", "time"])


# %%
np.log(mean_fp_x_flux(all_merged_data[0]).compute()).plot()

# %%
np.log(all_merged_data[0].flux["flat-annual-total"]).plot()

# %%
from functools import partial


def weighted_fixed_outer_regions_basis(
    merged_data: xr.DataTree, nbasis: int = 250, domain: str = "EUROPE"
) -> xr.DataArray:
    intem_regions = xr.open_dataset(
        "/user/work/bm13805/openghg_inversions/openghg_inversions/basis/outer_region_definition_EUROPE.nc"
    ).region
    _, intem_regions = xr.align(
        merged_data.flux.to_dataset(), intem_regions, join="override"
    )

    inner_index = intem_regions.max().values
    mask = intem_regions == inner_index
    grid = mean_fp_x_flux(merged_data, mask=mask)
    grid = grid / grid.max()

    func = partial(weighted_algorithm, nregion=nbasis, bucket=1, domain=domain)
    inner_region = xr.apply_ufunc(func, grid.as_numpy()).rename("basis")

    basis = intem_regions.rename("basis")

    loc_dict = {
        "lat": slice(inner_region.lat.min(), inner_region.lat.max() + 0.1),
        "lon": slice(inner_region.lon.min(), inner_region.lon.max() + 0.1),
    }
    basis.loc[loc_dict] = (inner_region + inner_index - 1).squeeze().values

    basis += 1  # intem_region_definitions.nc regions start at 0, not 1

    return basis


# %%
basis_functions = [
    weighted_fixed_outer_regions_basis(merged_data) for merged_data in all_merged_data
]

# %%
# %run inversions_experimental_code/basis_functions.py

# %%
bf1 = BasisFunctions(basis_functions[0], all_merged_data[0].flux["flat-annual-total"])

# %%
import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file("natural_earth_50.zip")

# %%
fig, ax = plt.subplots(figsize=(15, 8))

lat_min, lat_max = 40, 70
lon_min, lon_max = -20, 20

# lat_min, lat_max = basis_functions[0].lat.min().values, basis_functions[0].lat.max().values
# lon_min, lon_max = basis_functions[0].lon.min().values, basis_functions[0].lon.max().values


bf1.plot(shuffle=True)
world.boundary.plot(
    ax=ax, linewidth=0.6, edgecolor="white"
)  # or .plot(facecolor='none')
ax.set_xlim(float(lon_min), float(lon_max))
ax.set_ylim(float(lat_min), float(lat_max))

# %%
all_merged_data[0].scenario.map_over_datasets(
    lambda ds: (
        bf1.sensitivities(ds.fp_x_flux).rename("H").to_dataset()
        if "fp_x_flux" in ds.data_vars
        else ds
    )
)

# %%
bf_objs = [
    BasisFunctions(bf, md.flux["flat-annual-total"])
    for bf, md in zip(basis_functions, all_merged_data)
]


# %%
def apply_basis_functions(ds: xr.Dataset, bf: BasisFunctions) -> xr.Dataset:
    if "fp_x_flux" not in ds:
        return ds
    return bf.sensitivities(ds.fp_x_flux).rename("H").to_dataset()


h_matrix_datatrees = [
    md.map_over_datasets(partial(apply_basis_functions, bf=bf))
    for md, bf in zip(all_merged_data, bf_objs)
]

# %%
h_matrix_datatrees[0]


# %%
def concat_dict(ds_dict: dict[str, xr.Dataset], dim: str) -> xr.Dataset:
    return xr.concat([v.expand_dims({dim: [k]}) for k, v in ds_dict.items()], dim=dim)


def concat_tree(dt: xr.DataTree, dim: str) -> xr.Dataset:
    ds_dict = {k: v.to_dataset() for k, v in dt.items()}
    return concat_dict(ds_dict, dim)


# %%
h_matrices = [concat_tree(dt.scenario, "site") for dt in h_matrix_datatrees]

# %%
h_matrices[0]


# %%
def nesw_bc_basis(ds: xr.Dataset) -> xr.DataArray:
    bc_ds = ds[[f"bc_{d}" for d in "nesw"]].rename({f"bc_{d}": d for d in "nesw"})
    return bc_ds.sum(["lat", "lon", "height"]).to_dataarray(dim="bc_region")


# %%
h_bc_matrices = [
    concat_tree(
        md.scenario.map_over_datasets(
            lambda ds: (
                nesw_bc_basis(ds).rename("H_bc").to_dataset() if "bc_n" in ds else ds
            )
        ),
        dim="site",
    )
    for md in all_merged_data
]

# %%
h_bc_matrices[0]

# %% [markdown]
# #### Adding mf_error, splitting obs by year, and merging

# %%
combined_obs["mf_repeatability"] = combined_obs["mf_repeatability"].astype("float32")
combined_obs["mf_error"] = np.sqrt(
    combined_obs.mf_repeatability**2 + combined_obs.mf_variability**2
)

# %%
from typing import Literal


def make_dates_df(
    year: int,
    n_periods: int,
    frequency: Literal["annual", "yearly", "monthly"] = "annual",
    initial_month: int = 1,
    array_job_id: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame containing `n_periods` start and end dates starting at the given
    `year` and initial month (`initial_month` defaults to 1).

    Args:
        year: year to start first period
        n_periods: number of periods (months or years) in result
        frequency: length of periods, "annual" or "monthly"
        initial_month: month to start first period (default 1)

    Returns:
        DataFrame containing columns for start and end dates.
    """
    if frequency in ["annual", "yearly"]:
        freq = "YS"
        n_years, n_months = n_periods, 0
        offset = pd.DateOffset(years=1)  # offset for start vs. end dates
    elif frequency == "monthly":
        freq = "MS"
        n_years, n_months = 0, n_periods
        offset = pd.DateOffset(months=1)  # offset for start vs. end dates
    else:
        raise ValueError(f"Frequency {frequency} not accepted.")

    start = pd.Timestamp(year, initial_month, 1)
    end = start + pd.DateOffset(years=n_years, months=n_months)  # type: ignore

    start_dates = pd.date_range(start, end, inclusive="left", freq=freq)
    end_dates = start_dates + offset

    dates_df = pd.DataFrame({"start_date": start_dates, "end_date": end_dates})

    if array_job_id:
        dates_df.index += 1
        dates_df = dates_df.rename_axis("array_job_id")

    return dates_df


# %%
slices = []
for _, s, e in make_dates_df(2015, 10, array_job_id=False).itertuples():
    slices.append(slice(s, e))

# %%
split_obs = [combined_obs.sel(time=s) for s in slices]

# %%
split_obs = [so.assign_coords(site=split_obs[0].site.str.upper()) for so in split_obs]

# %%
from collections import defaultdict

# add info for later filtering
inlet_infos = []
for md in all_merged_data:
    inlet_info = defaultdict(list)
    columns = ["inlet_height_magl", "inlet_latitude", "inlet_longitude"]
    for k, v in md.scenario.items():
        for col in columns:
            try:
                val = float(v.attrs.get(col, np.nan))
            except:
                val = np.nan
            inlet_info[k].append(val)
    df = pd.DataFrame.from_dict(inlet_info).T
    df.columns = columns
    df.index.name = "site"
    inlet_infos.append(df.to_xarray())


# %%
def func(ds):
    dvs = [
        dv
        for dv in ds.data_vars
        if (ds[dv].dims == ("time",)) & (not str(dv).startswith("mf"))
    ]
    return ds[dvs]


met_data = [
    concat_tree(md.scenario.map_over_datasets(func), dim="site")
    for md in all_merged_data
]

# %%
all_data = [
    xr.merge([h, hbc, obs, iinfo, met], join="left")
    for h, hbc, obs, iinfo, met in zip(
        h_matrices, h_bc_matrices, split_obs, inlet_infos, met_data
    )
]

# %%
all_data[0]

# %%
all_data[0].where(
    (all_data[0].atmosphere_boundary_layer_thickness.compute() > 200.0)
    | (
        all_data[0].site.isin(("CMN", "JFJ"))
        | (
            all_data[0].atmosphere_boundary_layer_thickness.compute()
            > 50.0 + all_data[0].inlet_height_magl.compute()
        )
    )
).stack(nmeasure=("site", "time")).dropna("nmeasure")


# %%
def local_time(data):
    return data.time + xr.apply_ufunc(
        lambda x: pd.to_timedelta(24 * 60 * x / 360.0, unit="h"), data.inlet_longitude
    )


def local_hour(data):
    return local_time(data).dt.hour


# %%
local_time(all_data[0])

# %%
from openghg_inversions.hbmcmc.hbmcmc_output import ncdf_encoding


for data in all_data:
    start_date = str(data.time.values[0]).split("T")[0]
    output_name = f"sf6_{start_date}_synth_merged.nc"
    encoding = ncdf_encoding(data)
    data.to_netcdf(data_path / output_name, encoding=encoding)

# %%
# !ls -lsth {data_path}
# #files = !ls {data_path} | grep synth_merged
# for file in files:
# #    !rm {data_path / file}

# %%
with xr.open_dataset(data_path / "sf6_2015-01-01_synth_merged.nc") as ds:
    print(ds)

# %% [markdown]
# #### I forgot to filter!

# %%
all_data[0]


# %% [markdown]
# We need PBLH aligned with this style of dataset.


# %%
def func(ds):
    dvs = [
        dv
        for dv in ds.data_vars
        if (ds[dv].dims == ("time",)) & (not str(dv).startswith("mf"))
    ]
    return ds[dvs]


met_data = [
    concat_tree(md.scenario.map_over_datasets(func), dim="site")
    for md in all_merged_data
]

# %%
met_data

# %%
all_data[0].where(
    (met_data[0].atmosphere_boundary_layer_thickness.compute() > 200.0)
    | all_data[0].site.isin(["CMN", "JFJ"])
)

# %%
all_merged_data[0]

# %%
inlet_height_magl = {}
for md in all_merged_data:
    for k, v in md.scenario.items():
        if "inlet_height_magl" in v.attrs:
            inlet_height_magl[k] = v.attrs["inlet_height_magl"]

# %%
ihms = pd.Series(inlet_height_magl)
ihms.index.name = "site"
inlet_height_da = ihms.to_xarray()

# %%
all_data[0].where(
    (met_data[0].atmosphere_boundary_layer_thickness.compute() > 50.0 + inlet_height_da)
    | all_data[0].site.isin(["CMN", "JFJ"]),
    drop=True,
)


# %%
def pblh_min_filter(data, met, threshold=200.0, no_filter: list[str] | None = None):
    no_filter = no_filter or []
    return data.where(
        (met.atmosphere_boundary_layer_thickness.compute() > threshold)
        | data.site.isin(no_filter)
    )


def pblh_diff_filter(
    data, met, diff_threshold=200.0, no_filter: list[str] | None = None
):
    no_filter = no_filter or []
    return data.where(
        (
            met.atmosphere_boundary_layer_thickness.compute()
            > diff_threshold + inlet_height_da
        )
        | data.site.isin(no_filter)
    )


# %%
filtered_data = []

for data, met in zip(all_data, met_data):
    fdata = pblh_min_filter(data, met, no_filter=["JFJ", "CMN"])
    fdata = pblh_diff_filter(fdata, met, no_filter=["JFJ", "CMN"])
    filtered_data.append(fdata)

# %%
filtered_data[0].stack(nmeasure=("site", "time")).dropna("nmeasure")

# %%
all_data[0]

# %% [markdown]
# ### Running inversions

# %% [markdown]
# #### Making inversion inputs

# %%
# %run likelihood_tests.py

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
def pblh_filters(
    ds: xr.Dataset,
    no_filter=("CMN", "JFJ"),
    pblh_min_thres: float = 200.0,
    pblh_diff_thres: float = 50.0,
) -> xr.Dataset:
    pblh_min_filt = ds.atmosphere_boundary_layer_thickness.compute() > pblh_min_thres
    pblh_diff_filt = (
        ds.atmosphere_boundary_layer_thickness.compute()
        > pblh_diff_thres + ds.inlet_height_magl.compute()
    )
    no_filt = ds.site.isin(no_filter)
    return ds.where(no_filt | (pblh_min_filt & pblh_diff_filt))


# %%
def percentile_error_method(ds: xr.Dataset) -> np.ndarray:
    mf = ds.mf.as_numpy().sortby("time")
    monthly_50pc = mf.resample(time="MS").quantile(0.5)
    monthly_5pc = mf.resample(time="MS").quantile(0.05)
    res_err = (monthly_50pc - monthly_5pc).groupby("site").mean(dim="time")

    return res_err.values


# %%
def select_inversion_data_vars(ds: xr.Dataset) -> xr.Dataset:
    inversion_data_vars = [
        "H",
        "H_bc",
        "mf",
        "mf_error",
        "mf_repeatability",
        "mf_variability",
    ]
    dvs = [dv for dv in ds.data_vars if dv in inversion_data_vars]
    return ds[dvs]


# %%
def make_inv_inputs2(
    ds: xr.Dataset,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    min_error: str | dict[str, float] | float = 0.0,
    min_error_per_site: bool = True,
):
    # compute min error values (do this before stacking)
    if isinstance(min_error, float):
        ds["min_error"] = min_error * xr.ones_like(ds.mf)
    elif isinstance(min_error, dict):
        sites = np.unique(ds.site)
        min_err_values = np.array([min_error[site] for site in sites])
    elif min_error == "percentile":
        min_err_values = percentile_error_method(ds)
    else:
        raise ValueError(f"Option '{min_error}' is not valid.")

    # stack
    ds = ds.stack(nmeasure=("site", "time")).dropna("nmeasure")

    # add BC freq
    if bc_freq is not None:
        temp = setup_bc(ds.H_bc, bc_freq)
        ds = ds.drop_dims("bc_region")
        ds["H_bc"] = temp

    ds["site_indicator"] = make_site_indicator(ds.site)
    ds["sigma_freq_index"] = setup_sigma_freq(ds.time, freq=sigma_freq)

    # set up min error in more complicated cases
    if "min_error" not in ds:

        def setup_min_error2(min_err_values, site_indicator):
            return min_err_values[..., site_indicator]

        ds["min_error"] = xr.apply_ufunc(
            lambda x: setup_min_error2(min_err_values, x),
            ds.site_indicator,
            input_core_dims=[["nmeasure"]],
            output_core_dims=[list(ds.mf.dims)],
        )

    # ds["basis_flat"] = fp_data[".basis"]

    return ds


# %%
params = read_ini(ini_files[0])
params.get("calculate_min_error")


# %%
def filtered_inv_input(data: xr.Dataset, bc_freq, sigma_freq, min_error) -> xr.Dataset:
    return (
        (
            data.pipe(pblh_filters)
            .pipe(select_inversion_data_vars)
            .pipe(
                make_inv_inputs2,
                bc_freq=bc_freq,
                sigma_freq=sigma_freq,
                min_error=min_error,
            )
        )
        .compute()
        .dropna("nmeasure")
    )


# %% [markdown]
# #### Making models


# %%
def rhime_model(
    inv_input: xr.Dataset,
    x_prior: dict,
    bc_prior: dict,
    sig_prior: dict,
    offset: bool = True,
    pefo: bool = True,
) -> pm.Model:
    with pm.Model() as model:
        mu = add_linear_component(
            inv_input.H,
            data_name="hx",
            prior_args=x_prior,
            var_name="x",
            output_name="mu",
        )
        mu_bc = add_linear_component(
            inv_input.H_bc,
            data_name="hbc",
            prior_args=bc_prior,
            var_name="bc",
            output_name="mu_bc",
            compute_deterministic=True,
        )

        if offset:
            mu_bc = mu_bc + make_offset(inv_input.site_indicator, {"pdf": "normal"})

        add_old_likelihood(
            inv_input,
            sig_prior,
            mu=mu,
            mu_bc=mu_bc,
            power=1.99,
            pollution_events_from_obs=pefo,
        )

    return model


# %%

# %% [markdown]
# #### Parameter setup

# %%
experiment_configs = pd.DataFrame(
    [{k: v[0] for k, v in val.items()} for val in instr_to_dims.values()]
)
experiment_configs["pefo"] = [[True, False]] * 30
experiment_configs["offset"] = [[True, False]] * 30
experiment_configs = experiment_configs.explode("pefo", ignore_index=True)
experiment_configs = experiment_configs.explode("offset", ignore_index=True)
experiment_configs

# %%
# experiment_configs.to_csv(data_path / "experiment_configs1.csv")
experiment_configs = pd.read_csv(data_path / "experiment_configs1.csv")

# %%
ec1_subset = experiment_configs.loc[
    (experiment_configs.bias.isin([0.0, -0.1]))
    & (experiment_configs.noise.isin([0.0, 0.6, 2.0]))
]

# %%
ec1_subset

# %%
# ec1_subset.iloc[0]
for num, row in ec1_subset.iterrows():
    print(num, row, type(row))
    break


# %%
def base_data_args(params: dict) -> dict:
    result = dict(
        bc_freq=params.get("bc_freq"),
        sigma_freq=params.get("sigma_freq"),
        min_error=params.get("calculate_min_error") or params.get("min_error", 0.0),
    )
    return result


# %%
def model_args(params: dict, exp_conf: dict) -> dict:
    result = {
        "x_prior": params.get("xprior"),
        "bc_prior": params.get("bcprior"),
        "sig_prior": params.get("sigprior"),
        "offset": exp_conf.get("offset", True),
        "pefo": exp_conf.get("pefo", True),
    }
    if result["x_prior"].get("pdf", "").lower() == "lognormal" and params.get(
        "reparameterise_log_normal"
    ):
        result["x_prior"]["reparameterise"] = True
    return result


# %% [markdown]
# Example use:

# %%
exp_conf = dict(ec1_subset.iloc[0])
params = read_ini(ini_files[0])

inv_input = filtered_inv_input(all_data[0], **base_data_args(params)).sel(
    noise=exp_conf["noise"], bias=exp_conf["bias"]
)
model = rhime_model(inv_input, **model_args(params, exp_conf))

# %%
inv_input

# %% [markdown]
# #### Set-up sampling params and output options
#
# The plan is to dump the raw traces, tagged by year and "experiment config number".

# %%
out_path = Path("/group/chem/acrg/PARIS_inversions/sf6/brendan_tests")
# !ls {out_path}

# %%
log_path = out_path / "logs"
log_path.mkdir()

# %%
sample_kwargs = default_sample_kwargs.copy()
sample_kwargs["blas_cores"] = 8
sample_kwargs["draws"] = 1000
sample_kwargs["tune"] = 1000
sample_kwargs


# %%
def make_out_name(year, exp_conf_number):
    return f"{year}_config_{exp_conf_number}"


# %%
import zarr


def zarr_trace(out_path: Path, out_name: str):
    store = zarr.DirectoryStore(out_path / (out_name + "_trace.zarr"))
    return pm.backends.zarr.ZarrTrace(
        store, compressor=pm.util.UNSET, draws_per_chunk=200, include_transformed=True
    )


# %%
def run_inversion(
    data_path: Path,
    out_path: Path,
    exp_num: int,
    exp_conf: dict,
    ini_file: str,
    sample_kwargs: dict,
    zarr_backend: bool = True,
    save_trace: bool = True,
    error_noise: float | None = None,
):
    params = dict(read_ini(ini_file))
    year = params["start_date"][:4]
    print(f"Experiment {exp_num}: year {year}, {exp_conf}")
    out_name = make_out_name(year, exp_num)

    if save_trace and zarr_backend:
        sample_kwargs["trace"] = zarr_trace(out_path, out_name)

    merged_data_path = data_path / f"sf6_{params['start_date']}_synth_merged.nc"
    with xr.open_dataset(merged_data_path, cache=False) as data:
        print(f"Experiment {exp_num} loading data {merged_data_path}")
        inv_input_all = filtered_inv_input(data, **base_data_args(params))
        inv_input = inv_input_all.sel(noise=exp_conf["noise"], bias=exp_conf["bias"])

        if error_noise is not None:
            inv_input["mf_error"] = inv_input_all.mf_error.sel(
                noise=error_noise, bias=0.0
            )

    inv_input = inv_input.dropna("nmeasure")
    print(
        f"Experiment {exp_num}, year {Path(merged_data_path).name[:-4]} building model."
    )
    model = rhime_model(inv_input, **model_args(params, exp_conf))
    print(f"Experiment {exp_num}, year {Path(merged_data_path).name[:-4]} sampling.")
    idata = pm.sample(model=model, **sample_kwargs)
    print(f"Experiment {exp_num}, year {Path(merged_data_path).name[:-4]} done.")

    if save_trace and not zarr_backend:
        idata.to_netcdf(out_path / (out_name + "_trace.nc"))

    if not save_trace:
        return idata


# %% [markdown]
# #### Launch jobs
#
# - I'm going to run 6 (of 30) data variations for 10 years each, with 4 different models.
# - I'll use 24 cores, 6 for each chain
# - Using one process will mean one worker per mcmc run (I hope)
# - I could use one job per variation and run the years in a loop...

# %%
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(
    processes=1,
    cores=8,
    memory="20GB",
    walltime="02:00:00",
    account="chem007981",
    log_directory=str(log_path),
)
client = Client(cluster)
client

# %%
client.restart()

# %%
cluster.scale(jobs=6)

# %%
dict(sorted(cluster.workers.items(), key=lambda x: x[0]))

# %%
available_workers = [
    v.get("id")
    for v in client.scheduler_info(n_workers=len(cluster.workers))["workers"].values()
]

# %%
len(available_workers)

# %%
# retire bad workers...
for k, v in cluster.workers.items():
    if k in available_workers:
        continue
    print("Closing", k)
    v.close()

# %%
# inspect workers
workers = client.scheduler_info(n_workers=len(cluster.workers))[
    "workers"
]  # dict: {worker_address: info}
for addr, info in workers.items():
    print(addr)
    pprint(info)
    break
    print("  host:", info.get("host"))  # hostname where worker runs
    print("  pid:", info.get("pid"))  # worker process id
    print("  nthreads:", info.get("nthreads"))
    print("  memory_limit:", info.get("memory_limit"))
    print("  last_seen:", info.get("last-seen"))  # timestamp (may be milliseconds)
    print()

# %%
done_keys = []

# %%
from collections import defaultdict
import time

futures = defaultdict(list)  # futures keyed by experiment config number

count = 0
workers = available_workers  # list(cluster.workers.keys())
for ini_file in ini_files[:5]:
    for exp_num, row in ec1_subset2.iterrows():
        exp_conf = dict(row)

        func = partial(
            run_inversion,
            data_path=data_path,
            out_path=out_path,
            exp_num=exp_num,
            exp_conf=exp_conf,
            ini_file=ini_file,
            sample_kwargs=sample_kwargs,
            zarr_backend=False,
        )
        worker_name = workers[count]
        key = f"exp-{exp_num}_{Path(ini_file).name[:-4]}"
        if key in done_keys or key in processing:
            continue
        future = client.submit(func, workers=worker_name, key=key)
        futures[exp_num].append(future)
        time.sleep(2)
        count += 1
        count = count % len(available_workers)


# %%
futures

# %%
for v in futures.values():
    for f in v:
        if f.key not in done_keys and f.key not in processing:
            f.cancel()

# %%
from pprint import pprint

# dir(client)
pprint(client.processing())
pprint(client.futures)
# #client.retry?

# %%
# close workers we're not using
to_close = [k for k, v in client.processing().items() if not v]
client.retire_workers(workers=to_close, close_workers=True, remove=True)

# %%
# cancel everything that isn't processing and start again
processing = []
for v in client.processing().values():
    if v:
        processing.extend(v)

processing

# %%
for k, v in client.futures.items():
    if k not in processing:
        print("Cancelling", k)
        client.cancel(v)

# %%
for k, v in futures.items():
    for f in v.copy():
        if f.done():
            done_keys.append(f.key)
            f.release()
            v.remove(f)
            print(f)

# %%
# dir(client)
to_dec = []
for k, v in client.refcount.items():
    if k in done_keys:
        print(k, v)
        to_dec.append(k)

# %%
for k in to_dec:
    client._dec_ref(k)
client.refcount

# %%
client.restart()

# %%
client.refcount

# %%
# !ls -lsh {out_path}

# %%
for key in done_keys:
    try:
        client.cancel(key)
    except Exception as e:
        print(e)

# %%
client.close()

# %%
cluster.scale(n=0)

# %%
cluster.get_logs()

# %% [markdown]
# #### Trying without dask...

# %%
funcs = []
for i, (exp_num, row) in enumerate(ec1_subset.iterrows()):
    exp_conf = dict(row)
    for ini_file in ini_files:
        func = partial(
            run_inversion,
            data_path=data_path,
            out_path=out_path,
            exp_num=exp_num,
            exp_conf=exp_conf,
            ini_file=ini_file,
            sample_kwargs=sample_kwargs,
        )
        funcs.append(func)

# %%
funcs[0]

# %%
func = partial(funcs[0], sample_kwargs=(sample_kwargs | {"progressbar": True}))

# %%
result = func()

# %%
out_path = funcs[0].keywords["out_path"]
# !ls -R {out_path}

# %% [markdown]
# This doesn't seem to be writing anything out... it looks like the ZarrTrace feature doesn't work with the numpyro sampler.
#
# Let's test on another config.

# %%
ec1_subset2 = ec1_subset.loc[
    (ec1_subset.offset == False) & (ec1_subset.bias == 0.0) & (ec1_subset.noise != 0.0)
].sort_values(["noise", "pefo"])
ec1_subset2

# %%
sample_kwargs

# %%
exp_num = 97
exp_conf = dict(ec1_subset.loc[exp_num])
ini_file = ini_files[0]
result = run_inversion(
    data_path=data_path,
    out_path=out_path,
    exp_num=exp_num,
    exp_conf=exp_conf,
    ini_file=ini_file,
    sample_kwargs=sample_kwargs | {"progressbar": True, "tune": 100, "draws": 100},
    error_noise=0.6,
)

# %%
result

# %%
result2 = run_inversion(
    data_path=data_path,
    out_path=out_path,
    exp_num=exp_num,
    exp_conf=exp_conf,
    ini_file=ini_file,
    sample_kwargs=sample_kwargs
    | {"progressbar": True, "tune": 100, "draws": 100, "blas_cores": 4},
)

# %%
from openghg.retrieve import *

flux_obj = get_flux(species="sf6", domain="europe", source="edgar-annual-total")

# %%
import matplotlib.pyplot as plt

fix, axs = plt.subplots(1, 3, figsize=(15, 7))
vmin, vmax = -39, -28.5
lat_slice = slice(37, None)
lon_slice = slice(-14, 25)

flux_smoothed = bf_objs[0].interpolate(
    bf_objs[0].project(flux_obj.data.flux.isel(time=0), normalise=True)
)
np.log(flux_smoothed * (bf_objs[0].flux > 0).astype(float)).sel(
    lat=lat_slice, lon=lon_slice
).plot(ax=axs[0], vmin=vmin, vmax=vmax)
# np.log(flux_obj.data.flux.isel(time=0)).sel(lat=lat_slice, lon=lon_slice).plot(ax=axs[0], vmin=vmin, vmax=vmax)

np.log(
    bf_objs[0].interpolate(result2.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(ax=axs[1], vmin=vmin, vmax=vmax)
np.log(bf_objs[0].flux).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[2], vmin=vmin, vmax=vmax
)

# %%
np.log(flux_obj.data.flux.isel(time=0)).plot(vmin=vmin, vmax=vmax)

# %% [markdown]
# Let's try the easiest scenario: no noise, no bias:
# - 117: pefo True, no offset
# - 118: pefo False, no offset

# %%
exp_num = 117
exp_conf = dict(ec1_subset.loc[exp_num])
ini_file = ini_files[0]

result3 = run_inversion(
    data_path=data_path,
    out_path=out_path,
    exp_num=exp_num,
    exp_conf=exp_conf,
    ini_file=ini_file,
    sample_kwargs=sample_kwargs
    | {"progressbar": True, "tune": 200, "draws": 400, "blas_cores": 8},
    error_noise=0.6,
)

# %%
# for this run, I've added the option to reparameterise log normals
exp_num = 118
exp_conf = dict(ec1_subset.loc[exp_num])
ini_file = ini_files[0]

result4 = run_inversion(
    data_path=data_path,
    out_path=out_path,
    exp_num=exp_num,
    exp_conf=exp_conf,
    ini_file=ini_file,
    sample_kwargs=sample_kwargs
    | {"progressbar": True, "tune": 200, "draws": 400, "blas_cores": 8},
)

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
vmin, vmax = -39, -28

lat_slice = slice(37, None)
lon_slice = slice(-14, 25)


lat_min, lat_max = lat_slice.start, lat_slice.stop
lon_min, lon_max = lon_slice.start, lon_slice.stop

# plot prior
np.log(bf_objs[0].flux).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[0, 0], vmin=vmin, vmax=vmax, label="prior"
)
axs[0, 0].set_title("prior")

# plot true
flux_smoothed = bf_objs[0].interpolate(
    bf_objs[0].project(flux_obj.data.flux.isel(time=0), normalise=True)
)
np.log(flux_smoothed * (bf_objs[0].flux > 0).astype(float)).sel(
    lat=lat_slice, lon=lon_slice
).plot(
    ax=axs[0, 1], vmin=vmin, vmax=vmax, label="true, smoothed to basis, masked by prior"
)
# np.log(flux_obj.data.flux.isel(time=0) * (bf_objs[0].flux > 0).astype(float)).sel(lat=lat_slice, lon=lon_slice).plot(ax=axs[0, 1], vmin=vmin, vmax=vmax)
axs[0, 1].set_title("true")

# plot pefo true
np.log(
    bf_objs[0].interpolate(result3.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[1, 0], vmin=vmin, vmax=vmax, label="no noise, no bias, pefo True, no offset"
)
axs[1, 0].set_title("exp. 117, no noise, pefo True, offset True (by mistake)")

# plot pefo false
np.log(
    bf_objs[0].interpolate(result4.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[1, 1], vmin=vmin, vmax=vmax, label="no noise, no bias, pefo False, no offset"
)
axs[1, 1].set_title("exp. 118")

# plot country borders
for ax in axs.flat:
    world.boundary.plot(
        ax=ax, linewidth=0.6, edgecolor="white"
    )  # or .plot(facecolor='none')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# %% [markdown]
# Let's run the same number of samples for these two model setups for the data with 2.0 sigma scaling (i.e. max noise)
# - 97: 2.0 noise, pefo=True
# - 99: 2.0 noise, pefo=False

# %%
exp_num = 97
exp_conf = dict(ec1_subset.loc[exp_num])
ini_file = ini_files[0]

result6 = run_inversion(
    data_path=data_path,
    out_path=out_path,
    exp_num=exp_num,
    exp_conf=exp_conf,
    ini_file=ini_file,
    sample_kwargs=sample_kwargs
    | {"progressbar": True, "tune": 200, "draws": 400, "blas_cores": 8},
)

# %%
exp_num = 99
exp_conf = dict(ec1_subset.loc[exp_num])
ini_file = ini_files[0]

result5 = run_inversion(
    data_path=data_path,
    out_path=out_path,
    exp_num=exp_num,
    exp_conf=exp_conf,
    ini_file=ini_file,
    sample_kwargs=sample_kwargs
    | {"progressbar": True, "tune": 200, "draws": 400, "blas_cores": 8},
)

# %%

fig, axs = plt.subplots(3, 2, figsize=(15, 22))
vmin, vmax = -39, -28

lat_slice = slice(37, None)
lon_slice = slice(-14, 25)


lat_min, lat_max = lat_slice.start, lat_slice.stop
lon_min, lon_max = lon_slice.start, lon_slice.stop

# plot prior
np.log(bf_objs[0].flux).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[0, 0], vmin=vmin, vmax=vmax, label="prior"
)
axs[0, 0].set_title("prior")

# plot true
flux_smoothed = bf_objs[0].interpolate(
    bf_objs[0].project(flux_obj.data.flux.isel(time=0), normalise=True)
)
# np.log(flux_smoothed * (bf_objs[0].flux > 0).astype(float)).sel(lat=lat_slice, lon=lon_slice).plot(ax=axs[0, 1], vmin=vmin, vmax=vmax, label="true, smoothed to basis, masked by prior")
np.log(flux_obj.data.flux.isel(time=0) * (bf_objs[0].flux > 0).astype(float)).sel(
    lat=lat_slice, lon=lon_slice
).plot(ax=axs[0, 1], vmin=vmin, vmax=vmax)
axs[0, 1].set_title("true")

# plot pefo true
np.log(
    bf_objs[0].interpolate(result3.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[1, 0], vmin=vmin, vmax=vmax, label="no noise, no bias, pefo True, no offset"
)
axs[1, 0].set_title("exp. 117, no noise, pefo True, offset True (by mistake)")

# plot pefo false
np.log(
    bf_objs[0].interpolate(result4.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[1, 1], vmin=vmin, vmax=vmax, label="no noise, no bias, pefo False, no offset"
)
axs[1, 1].set_title("exp. 118, no noise, pefo False")

# plot pefo true, with noise
np.log(
    bf_objs[0].interpolate(result6.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[2, 0], vmin=vmin, vmax=vmax, label="2.0 noise, no bias, pefo True, no offset"
)
axs[2, 0].set_title("exp. 97, 2.0 noise, pefo True")

# plot pefo false, with noise
np.log(
    bf_objs[0].interpolate(result5.posterior.x.mean(["chain", "draw"]), flux=True)
).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs[2, 1],
    vmin=vmin,
    vmax=vmax,
    label="2.0 noise, no bias, pefo False, no offset",
)
axs[2, 1].set_title("exp. 99, 2.0 noise, pefo False")

# plot country borders
for ax in axs.flat:
    world.boundary.plot(
        ax=ax, linewidth=0.6, edgecolor="white"
    )  # or .plot(facecolor='none')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# %% [markdown]
# ### Runs for more years
#
#

# %%
# !ls -lsh {out_path}

# %%
from collections import namedtuple

TraceInfo = namedtuple("TraceInfo", "exp_num, title, trace")


def get_trace_info(year: int) -> list[TraceInfo]:
    traces = [az.InferenceData.from_netcdf(f) for f in out_path.glob(f"{year}*.nc")]
    exp_nums = [p.name.split("_")[-2] for p in list(out_path.glob(f"{year}*.nc"))]
    titles = [
        f"Experiment {exp_num}: noise {row['noise']}, pefo {row['pefo']}"
        for exp_num, row in ec1_subset2.loc[map(int, exp_nums)].iterrows()
    ]
    return [
        TraceInfo(exp_num, title, trace)
        for exp_num, title, trace in zip(exp_nums, titles, traces)
    ]


# %%
import arviz as az

traces_2017 = [az.InferenceData.from_netcdf(f) for f in out_path.glob("2017*.nc")]

# %%
exps_2017 = [p.name.split("_")[-2] for p in list(out_path.glob("2017*.nc"))]
exps_2017

# %%
ec1_subset2

# %%
titles = [
    f"Experiment {exp_num}: noise {row['noise']}, pefo {row['pefo']}"
    for exp_num, row in ec1_subset2.loc[map(int, exps_2017)].iterrows()
]

# %%
titles

# %%
traces_2017[0]

# %%
bf_obj = bf_objs[2]
bf_obj.flux


# %%
# !ls {data_path}


# %%
def get_fp_flux(year):
    with xr.open_datatree(
        data_path / f"sf6_{year}-01-01_4h-no-basis-no-filt_merged-data.zarr.zip",
        engine="zarr",
    ) as dt:
        return mean_fp_x_flux(dt)


# %%
fpflux

# %%
np.random.shuffle(bf_obj.labels_shuffled)

# %%

fig, axs = plt.subplots(4, 2, figsize=(15, 29))
vmin, vmax = -39, -28

lat_slice = slice(37, None)
lon_slice = slice(-14, 25)


lat_min, lat_max = lat_slice.start, lat_slice.stop
lon_min, lon_max = lon_slice.start, lon_slice.stop

# plot prior
np.log(bf_obj.flux).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs.flat[0], vmin=vmin, vmax=vmax, label="prior"
)
axs.flat[0].set_title("prior")

# plot true
flux_smoothed = bf_obj.interpolate(
    bf_obj.project(flux_obj.data.flux.isel(time=0), normalise=True)
)
np.log(flux_smoothed * (bf_obj.flux > 0).astype(float)).sel(
    lat=lat_slice, lon=lon_slice
).plot(
    ax=axs[0, 1], vmin=vmin, vmax=vmax, label="true, smoothed to basis, masked by prior"
)
# np.log(flux_obj.data.flux.isel(time=0) * (bf_obj.flux > 0).astype(float)).sel(lat=lat_slice, lon=lon_slice).plot(ax=axs.flat[1], vmin=vmin, vmax=vmax)
axs.flat[1].set_title("true")

for trace, title, ax in zip(traces_2017, titles, axs.flat[2:]):
    np.log(
        bf_obj.interpolate(trace.posterior.x.mean(["chain", "draw"]), flux=True)
    ).sel(lat=lat_slice, lon=lon_slice).plot(
        ax=ax, vmin=vmin, vmax=vmax, label="no noise, no bias, pefo True, no offset"
    )
    ax.set_title(title)

np.log(fpflux).sel(lat=lat_slice, lon=lon_slice).plot(
    ax=axs.flat[-2], label="mean fp x flux"
)
axs.flat[-2].set_title("mean fp x flux")

bf_obj.plot(shuffle=True, ax=axs.flat[-1])
axs.flat[-1].set_title("basis functions")

# plot country borders
for ax in axs.flat:
    world.boundary.plot(
        ax=ax, linewidth=0.6, edgecolor="white"
    )  # or .plot(facecolor='none')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

# %% [markdown]
# #### 2015

# %%
year = 2015
trace_info_2015 = get_trace_info(2015)
bf_obj_2015 = bf_objs[0]
fpflux_2015 = get_fp_flux(2015)

# %%
year = 2015
flux_2015 = get_flux(
    species="sf6", domain="europe", source="edgar-annual-total"
).data.flux.sel(time=slice(f"{year}-01-01", f"{year + 1}-01-01"))

# %%
flux_2015


# %%
def plot_experiments(
    trace_info,
    bf_obj,
    fpflux,
    flux,
    year,
    smooth_true: bool = True,
    mask_true: bool = True,
):
    fig, axs = plt.subplots(4, 2, figsize=(15, 29))
    fig.suptitle(f"Experiments for {year}")

    vmin, vmax = -39, -28

    #    lat_slice = slice(37, None)
    #    lon_slice = slice(-14, 25)
    lat_slice = slice(None, None)
    lon_slice = slice(None, None)

    lat_min, lat_max = lat_slice.start, lat_slice.stop
    lon_min, lon_max = lon_slice.start, lon_slice.stop

    # plot prior
    np.log(bf_obj.flux).sel(lat=lat_slice, lon=lon_slice).plot(
        ax=axs.flat[0], vmin=vmin, vmax=vmax, label="prior"
    )
    axs.flat[0].set_title("prior")

    # plot true
    mask = (bf_obj.flux > 0).astype(float) if mask_true else 1.0

    if "time" in flux.dims:
        flux = flux.isel(time=0)

    if smooth_true:
        flux_smoothed = bf_obj.interpolate(bf_obj.project(flux, normalise=True))
        np.log(flux_smoothed * mask).sel(lat=lat_slice, lon=lon_slice).plot(
            ax=axs[0, 1],
            vmin=vmin,
            vmax=vmax,
            label="true, smoothed to basis, masked by prior",
        )
    else:
        np.log(flux * mask).sel(lat=lat_slice, lon=lon_slice).plot(
            ax=axs.flat[1], vmin=vmin, vmax=vmax
        )
    axs.flat[1].set_title("true")

    for (_, title, trace), ax in zip(trace_info, axs.flat[2:]):
        np.log(
            bf_obj.interpolate(trace.posterior.x.mean(["chain", "draw"]), flux=True)
        ).sel(lat=lat_slice, lon=lon_slice).plot(
            ax=ax, vmin=vmin, vmax=vmax, label="no noise, no bias, pefo True, no offset"
        )
        ax.set_title(title)

    np.log(fpflux).sel(lat=lat_slice, lon=lon_slice).plot(
        ax=axs.flat[-2], label="mean fp x flux"
    )
    axs.flat[-2].set_title("mean fp x flux")

    bf_obj.plot(shuffle=True, ax=axs.flat[-1])
    axs.flat[-1].set_title("basis functions")

    # plot country borders
    for ax in axs.flat:
        world.boundary.plot(
            ax=ax, linewidth=0.6, edgecolor="white"
        )  # or .plot(facecolor='none')
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)


# %%
plot_experiments(
    trace_info_2015,
    bf_obj_2015,
    fpflux_2015,
    flux_2015,
    2015,
    smooth_true=False,
    mask_true=False,
)

# %% [markdown]
# #### 2016

# %%
year = 2016
plot_args_2016 = dict(
    trace_info=get_trace_info(year),
    bf_obj=bf_objs[year - 2015],
    fpflux=get_fp_flux(year),
    flux=get_flux(
        species="sf6", domain="europe", source="edgar-annual-total"
    ).data.flux.sel(time=slice(f"{year}-01-01", f"{year + 1}-01-01")),
)
plot_experiments(**plot_args_2016)

# %% [markdown]
# #### 2018

# %%
year = 2018
plot_args_2018 = dict(
    trace_info=get_trace_info(year),
    bf_obj=bf_objs[year - 2015],
    fpflux=get_fp_flux(year),
    flux=get_flux(
        species="sf6", domain="europe", source="edgar-annual-total"
    ).data.flux.sel(time=slice(f"{year}-01-01", f"{year + 1}-01-01")),
    year=year,
)
plot_experiments(**plot_args_2018)

# %% [markdown]
# #### 2019

# %%
year = 2019
plot_args_2019 = dict(
    trace_info=get_trace_info(year),
    bf_obj=bf_objs[year - 2015],
    fpflux=get_fp_flux(year),
    flux=get_flux(
        species="sf6", domain="europe", source="edgar-annual-total"
    ).data.flux.sel(time=slice(f"{year}-01-01", f"{year + 1}-01-01")),
    year=year,
)
plot_experiments(**plot_args_2019)

# %% [markdown]
# ### Country totals for 2015-2019

# %%
from openghg_inversions.array_ops import sparse_xr_dot, align_sparse_lat_lon
from openghg_inversions.postprocessing.countries import Countries

# %%
# Countries.from_file?

# %%
countries = Countries.from_file(
    "/group/chem/acrg/LPDM/countries/country_EUROPE_EEZ_PARIS_gapfilled.nc",
    country_code="alpha3",
)

# %%
# countries.get_x_to_country_mat??

# %%
from openghg.util import molar_mass

sf6_mm = molar_mass("sf6")
print(sf6_mm)

# %%
flux = get_flux(species="sf6", domain="europe", source="edgar-annual-total").data.flux

_, flux_aligned = xr.align(countries.area_grid, flux, join="override")
true_country_totals = sparse_xr_dot(
    countries.matrix, countries.area_grid * flux_aligned
).compute()

# %%
flux * 1e-3 / sf6_mm * (365.25 * 24 * 3600)

# %%
# column 1 is "OCEAN", which includes part of North America...
true_country_df = (
    (true_country_totals / sf6_mm * 1e-3 * (365.25 * 24 * 3600))
    .to_series()
    .unstack()
    .T.iloc[:, 1:]
)

# %%
true_country_df

# %%
traces = [get_trace_info(year) for year in range(2015, 2025)]

# %%
post_flux_dict = defaultdict(list)
for trace, year in zip(traces, range(2015, 2025)):
    for ti in trace:
        post_flux_dict[ti.exp_num].append(
            ti.trace.posterior.x.mean(["draw", "chain"]).expand_dims(
                {"time": [f"{year}-01-01"]}
            )
        )

# %%
for bfo in bf_objs:
    print(bfo.flux.time.values)

# %%
# bf_objs[0].interpolate?

# %%
post_flux_concat = {
    k: xr.concat(
        [bfo.interpolate(ds * bfo.flux.squeeze("time")) for ds, bfo in zip(v, bf_objs)],
        dim="time",
    )
    for k, v in post_flux_dict.items()
}

# %%
post_flux_concat["97"]

# %%
post_country_totals = {}
for k, v in post_flux_concat.items():
    _, flux_aligned = xr.align(countries.area_grid, v, join="override")
    post_country_totals[k] = sparse_xr_dot(
        countries.matrix, countries.area_grid * flux_aligned
    ).compute()

# %%
post_country_dfs = {
    k: (v / sf6_mm * 1e-3 * (365.25 * 24 * 3600))
    .as_numpy()
    .to_series()
    .unstack()
    .T.iloc[:, 1:]
    for k, v in post_country_totals.items()
}

# %%
post_country_dfs.keys()

# %%
titles = [ti.title for ti in traces[0]]
titles

# %%
for df in post_country_dfs.values():
    df.index = pd.to_datetime(df.index)

# %%
print(true_country_df["DEU"].index)
print(post_country_dfs["39"]["DEU"].index)

# %%
fig, ax = plt.subplots(figsize=(15, 7))

for pcdf, title in zip(post_country_dfs.values(), titles):
    pcdf["DEU"].plot(ax=ax, label=title)

true_country_df["DEU"].iloc[2:7].plot(ax=ax, label="true")

fig.legend()
ax.set_title("Germany country totals")

# %% [markdown]
# ...forgot to do x to country matrix... need to do this year by year...

# %%

# %%
country_ds = xr.open_dataset(
    "/group/chem/acrg/LPDM/countries/country_EUROPE_EEZ_PARIS_gapfilled.nc"
)
country_ds

# %%
fig, ax = plt.subplots(figsize=(15, 7))
lon_min = country_ds.lon.min().values
lon_max = country_ds.lon.max().values
lat_min = country_ds.lat.min().values
lat_max = country_ds.lat.max().values

(country_ds.country == 0).plot(ax=ax)
world.boundary.plot(
    ax=ax, linewidth=0.6, edgecolor="white"
)  # or .plot(facecolor='none')
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# %%
country_ds.country_code.values

# %%
np.log((country_ds.country == 0) * flux.isel(time=0)).plot()

# %%
