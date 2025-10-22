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
# # Goal
#
# Define a simple model or two and see how well they work.
#
# Our current model is quite complicated, so it might be helpful to see if a simple model works at all.
#
# Also, the simple models will parameters that are easier to interpret, so tuning uncertainties should be somewhat easier.
#
# ## Simple model 1: Gaussian
#
# A fully Gaussian model would permit a "classical" analysis of the error terms.
#
# This model might also allow us to explore adding correlations while taking advantage of faster methods for fitting the model.
#
# Initially, using PyMC and MCMC will let us get started quickly.
#
# ## Simple model 2: MCMC with lognormal prior, Gaussian likelihood
#
# We can add a prior on the variance of the Gaussian likelihood, but we won't add any scaling for pollution events.
#
# ## Simple model 3(?): Gaussian with log transformed flux
#
# This is a simple way to enforce non-negative flux scaling factors.
#
# # Set-up
#
# ## Data
#
# We're going to use Helene's SF6 data config for 2015-2019, since the model has problems here and there is no flask data to complicate matters.
#
# Caching the combined data will be useful, but we need to decide when the basis functions will be applied.
# To start with, maybe we should just use the same set-up as Helene (250 basis functions, weighted).
#
# ## Modelling and Sampling
#
# - Make the models using some of the components from `likelihood_tests.py`
# - Using NUTS for 1000-2000 samples should be sufficient for testing.
# - Variational inference might be faster but I've never used it
#
#
# ## Outputs
#
# - If we use PARIS post-processing, we'll be able to use the code from the first notebook to work with fluxy, although we might need to extract some scripts.
# - Saving the the inversion output object will help with inspection (although we can just retrieve it in memory in this notebook)
#

# %% [markdown]
# # Data
#
# We'll use the script from `likelihood_tests.py` then decide how to save the data.
#
# It would be nice to parallelise this...
#
# First let's find the ini files we need.

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

sf6_path = Path("/group/chem/acrg/PARIS_inversions/sf6/")
sf6_base_nid2025_path = sf6_path / "RHIME_NAME_EUROPE_FLAT_ConfigNID2025_sf6_yearly"
# ini_files = !ls {sf6_base_nid2025_path / "*.ini"}
# get 2015-2019
ini_files = ini_files[2:7]

# %%
ini_files

# %%
# %run likelihood_tests.py

# %% editable=true slideshow={"slide_type": ""}
import dask
import dask.bag as db

data_dicts = db.from_sequence(ini_files).map(get_fp_data_dict).compute()


# %%
data_dicts[0]

# %% [markdown]
# The best way to save this data is probably using xr.DataTree, but this isn't how we currently save "combined scenarios".
#
# Let's try converting a "fp_all" dict to a DataTree.

# %%
fp_all = data_dicts[0]
print(fp_all.keys())

# %% [markdown]
# - ".species", ".scales", and ".units" are attributes
# - all others are Datasets
# - Should "MHD", "TAC", etc. be in a subgroup?

# %%
scenario = {k: v for k, v in fp_all.items() if not k.startswith(".")}
attrs = {
    k.removeprefix("."): v
    for k, v in fp_all.items()
    if k in [".species", ".scales", ".units"]
}
aux_data = {
    k.removeprefix("."): v
    for k, v in fp_all.items()
    if (k not in scenario) and (k.removeprefix(".") not in attrs)
}

# nest flux (this can be done automatically from nested dict according to xarray docs, but
# it doesn't work for me... maybe I need to update xarray
# aux_data["/flux"] = xr.DataTree.from_dict({k: v.data for k, v in aux_data["flux"].items()})
# del aux_data["flux"]

# "flux" as a dataset... this might not work if we're mixing high/low frequency fluxes
# but it works for multiple sectors
aux_data["flux"] = xr.Dataset({k: v.data.flux for k, v in aux_data["flux"].items()})

# get data from BoundaryConditionsData object... maybe we
# should put metadata in global attrs for this group?
aux_data["bc"] = aux_data["bc"].data

# add basis as data variable at root?
basis = aux_data["basis"]
del aux_data["basis"]

# add basis within group?
# aux_data["basis"] = aux_data["basis"].rename("basis").to_dataset()
dt_dict = aux_data.copy()
dt_dict["/scenario"] = xr.DataTree.from_dict({k: v for k, v in scenario.items()})

# %%
dt = xr.DataTree.from_dict(dt_dict)
dt.attrs = attrs
dt["basis"] = basis
dt

# %% [markdown]
# It might be a bit of a pain to round-trip this to the original structure, but this should work fine for storing the data, and restoring the "scenario" part will be easy.

# %%
dt.flux

# %%
dt.to_dict()


# %% [markdown]
# Okay, that seems pretty easy to revert to the "fp_all" format.
#
# Let's make a function to create a DataTree from the "fp_all" style dicts.


# %%
def fp_all_to_datatree(fp_all: dict, name: str | None = None) -> xr.DataTree:
    scenario = {k: v for k, v in fp_all.items() if not k.startswith(".")}
    attrs = {
        k.removeprefix("."): v
        for k, v in fp_all.items()
        if k in [".species", ".scales", ".units"]
    }
    aux_data = {
        k.removeprefix("."): v
        for k, v in fp_all.items()
        if (k not in scenario) and (k.removeprefix(".") not in attrs)
    }

    # nest flux (this can be done automatically from nested dict according to xarray docs, but
    # it doesn't work for me... maybe I need to update xarray
    # aux_data["/flux"] = xr.DataTree.from_dict({k: v.data for k, v in aux_data["flux"].items()})
    # del aux_data["flux"]

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
    basis = aux_data["basis"]
    del aux_data["basis"]

    # add basis within group?
    # aux_data["basis"] = aux_data["basis"].rename("basis").to_dataset()

    dt_dict = aux_data.copy()
    dt_dict["/scenario"] = xr.DataTree.from_dict({k: v for k, v in scenario.items()})

    dt = xr.DataTree.from_dict(dt_dict)
    dt.attrs = attrs
    dt["basis"] = basis

    if name is not None:
        dt.name = name

    return dt


# %%
from openghg.dataobjects import BoundaryConditionsData, FluxData


def datatree_to_fp_all(dt: xr.DataTree) -> dict:
    d = dt.to_dict()
    result = {}
    result[".flux"] = {
        dv: FluxData(data=d["/flux"][[dv]], metadata={}) for dv in d["/flux"].data_vars
    }
    result[".bc"] = BoundaryConditionsData(data=d["/bc"], metadata={})
    result[".basis"] = d["/"].basis
    result[".species"] = dt.attrs.get("species")
    result[".units"] = dt.attrs.get("units")
    result[".scales"] = dt.attrs.get("scales")
    for k, v in d.items():
        if k.startswith("/scenario/"):
            site = k.split("/")[-1]
            result[site] = v
    return result


# %%
datatree_to_fp_all(dt)

# %%
str(dt.time.dt.year.values)

# %%
# !mkdir sf6_model_testing_data

# %%
work_path = Path("/user/work/bm13805/")
data_path = work_path / "sf6_model_testing_data"

# %%
dt_and_paths = []
for dd in data_dicts:
    dt = fp_all_to_datatree(dd)
    try:
        year = str(dt.flux.time.dt.year.values[0])
    except IndexError:
        year = str(dt.flux.time.dt.year.values)
    out_path = data_path / f"sf6_combined_data_{year}.zarr"
    dt_and_paths.append((dt, out_path))

# %%
dt0, path0 = dt_and_paths[0]


# %%
def rechunk_ds(ds: xr.Dataset) -> xr.Dataset:
    default_chunks = {"lat": 293, "lon": 391, "height": 20, "bc_region": 4}
    chunks = {dim: default_chunks.get(dim) for dim in ds.dims if dim != "time"}
    if ds.sizes.get("time", 0) > 240:
        chunks["time"] = 240
    elif "time" in ds.dims:
        chunks["time"] = ds.sizes["time"]
    if "region" in ds.dims:
        chunks["region"] = ds.sizes["region"]
    return ds.chunk(chunks)


# %%
dt0 = dt0.map_over_datasets(rechunk_ds)

# %%
# dt0.to_zarr(path0)

# %%
# for dt, path in dt_and_paths[1:]:
#    dt.map_over_datasets(rechunk_ds).to_zarr(path)

# %%
# data_files = !ls {data_path} | grep combined_data
data_paths = [data_path / f for f in data_files]
data_paths

# %% [markdown]
# Now let's try reloaded the data.

# %%
# %run inversions_experimental_code/data_functions.py

# %%
fp_all_2015 = datatree_to_fp_all(xr.open_datatree(data_paths[0]))

# %%
fp_all_2015

# %% [markdown]
# ## Preparing inversion inputs
#
# We'll use the `make_inv_inputs` function from `likelihood_tests.py`. This requires the "fp_all" (or "fp_data") dict, along with some parameters from the ini file. We'll get these first.

# %%
from collections import namedtuple

InversionInfo = namedtuple("InversionInfo", "fp_data,params")

inversion_info = {}
for ini, dpath in zip(ini_files, data_paths):
    params = read_ini(ini)
    dt = xr.open_datatree(dpath)
    try:
        year = str(dt.flux.time.dt.year.values[0])
    except IndexError:
        year = str(dt.flux.time.dt.year.values)
    fp_all = datatree_to_fp_all(dt)

    inversion_info[year] = InversionInfo(fp_all, params)

# %%
inversion_info["2015"].params

# %%
InversionInput = namedtuple("InversionInput", "inv_input,params")

inversion_inputs = {}
for k, v in inversion_info.items():
    inv_input = make_inv_inputs(
        v.fp_data,
        bc_freq=v.params.get("bc_freq"),
        sigma_freq=v.params.get("sigma_freq"),
        min_error=v.params.get("min_error") or v.params.get("calculate_min_error"),
    )
    inversion_inputs[k] = InversionInput(inv_input, v.params)

# %%
inversion_inputs["2015"].inv_input

# %% [markdown]
# # Model 2: Lognormal with Gaussian likelihood
#
#

# %%
# import pytensor before pymc so we can set config values
import pytensor

pytensor.config.floatX = "float32"
pytensor.config.warn_float64 = "warn"

import arviz as az
import pymc as pm

# %%
# %run likelihood_tests.py

# %%
inv_input_obj = inversion_inputs["2015"]
inv_input = inv_input_obj.inv_input
params = inv_input_obj.params

with pm.Model() as model:
    mu = add_linear_component(
        inv_input.H,
        data_name="hx",
        prior_args=params["xprior"],
        var_name="x",
        output_name="mu",
    )
    mu_bc = add_linear_component(
        inv_input.H_bc,
        data_name="hbc",
        prior_args=params["bcprior"],
        var_name="bc",
        output_name="mu_bc",
        compute_deterministic=True,
    )

    make_offset(inv_input.site_indicator, {"pdf": "normal"})

    # make likelihood
    Y = add_model_data(inv_input.mf, "Y")
    error = add_model_data(inv_input.mf_error.astype("float32"), "error")
    min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

    sigma = make_sigma(
        inv_input.site_indicator,
        {"pdf": "inversegamma", "alpha": 2.5, "beta": 5},
        inv_input.sigma_freq_index,
    )

    epsilon = pm.Deterministic(
        "epsilon", pt.sqrt(error**2 + min_error**2 + 0.01 * sigma**2), dims="nmeasure"
    )
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")


# %%
trace = pm.sample_prior_predictive(draws=10000, model=model)

# %%
trace

# %%
prior_preds = trace.prior_predictive.y.assign_coords(
    nmeasure=inv_input.nmeasure
).squeeze("chain")

# %%
bc_prior = trace.prior.mu_bc.mean(["chain", "draw"]).assign_coords(
    nmeasure=inv_input.nmeasure
)

# %%
sites = list(np.unique(inv_input.site))
sites

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 2, figsize=(15, 15))
for site, ax in zip(sites, axs.flat):
    for i in range(10):
        prior_preds.sel(site=site).isel(draw=slice(i, None, 10)).mean("draw").plot(
            ax=ax, label="prior pred", alpha=0.05, color="blue"
        )
    inv_input.mf.sel(site=site).plot(ax=ax, label="obs", alpha=0.5, color="orange")
    bc_prior.sel(site=site).plot(ax=ax, label="bc prior", alpha=0.5, color="green")

# fig.legend()

# %% [markdown]
# These prior predictives look reasonable. We're not modelling the big pollution events, and the baseline is too high in some cases.
#
# What if we change the prior uncertainty?

# %%
print(params["xprior"])
print(params["bcprior"])

# %%
with pm.Model() as model2:
    mu = add_linear_component(
        inv_input.H,
        data_name="hx",
        prior_args={"pdf": "lognormal", "stdev": 1.0},
        var_name="x",
        output_name="mu",
    )
    mu_bc = add_linear_component(
        inv_input.H_bc,
        data_name="hbc",
        prior_args=params["bcprior"],
        var_name="bc",
        output_name="mu_bc",
        compute_deterministic=True,
    )

    make_offset(inv_input.site_indicator, {"pdf": "normal"})

    # make likelihood
    Y = add_model_data(inv_input.mf, "Y")
    error = add_model_data(inv_input.mf_error.astype("float32"), "error")
    min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

    sigma = make_sigma(
        inv_input.site_indicator,
        {"pdf": "inversegamma", "alpha": 2.5, "beta": 5},
        inv_input.sigma_freq_index,
    )

    epsilon = pm.Deterministic(
        "epsilon", pt.sqrt(error**2 + min_error**2 + 0.01 * sigma**2), dims="nmeasure"
    )
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

# %%
trace2 = pm.sample_prior_predictive(draws=1000, model=model2)


# %%
# TODO: use quantiles instead of plotting lots of traces...
def plot_prior_preds(trace, inv_input, skip=20):
    prior_preds = trace.prior_predictive.y.assign_coords(
        nmeasure=inv_input.nmeasure
    ).squeeze("chain")
    bc_prior = trace.prior.mu_bc.mean(["chain", "draw"]).assign_coords(
        nmeasure=inv_input.nmeasure
    )

    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    for site, ax in zip(sites, axs.flat):
        for i in range(skip):
            prior_preds.sel(site=site).isel(draw=slice(i, None, skip)).mean(
                "draw"
            ).plot(ax=ax, label="prior pred", alpha=0.1 / np.sqrt(skip), color="blue")
        inv_input.mf.sel(site=site).plot(ax=ax, label="obs", alpha=0.5, color="orange")
        bc_prior.sel(site=site).plot(ax=ax, label="bc prior", alpha=0.5, color="green")


# %%
plot_prior_preds(trace2, inv_input)

# %%
plot_prior_preds(trace, inv_input, skip=200)

# %%
with pm.Model() as model3:
    mu = add_linear_component(
        inv_input.H,
        data_name="hx",
        prior_args=params["xprior"],
        var_name="x",
        output_name="mu",
    )
    mu_bc = add_linear_component(
        inv_input.H_bc,
        data_name="hbc",
        prior_args=params["bcprior"],
        var_name="bc",
        output_name="mu_bc",
        compute_deterministic=True,
    )

    make_offset(inv_input.site_indicator, {"pdf": "normal"})

    # make likelihood
    Y = add_model_data(inv_input.mf, "Y")
    error = add_model_data(inv_input.mf_error.astype("float32"), "error")
    min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

    #    sigma = make_sigma(inv_input.site_indicator, {"pdf": "inversegamma", "alpha": 2.5, "beta": 5}, inv_input.sigma_freq_index)

    epsilon = pm.Deterministic(
        "epsilon", pt.sqrt(error**2 + min_error**2), dims="nmeasure"
    )
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

# %%
trace3 = pm.sample_prior_predictive(draws=5000, model=model3)

# %%
plot_prior_preds(trace3, inv_input, skip=100)

# %%
with pm.Model() as model4:
    mu = add_linear_component(
        inv_input.H,
        data_name="hx",
        prior_args={"pdf": "lognormal", "stdev": 1.0},
        var_name="x",
        output_name="mu",
    )
    mu_bc = add_linear_component(
        inv_input.H_bc,
        data_name="hbc",
        prior_args=params["bcprior"],
        var_name="bc",
        output_name="mu_bc",
        compute_deterministic=True,
    )

    make_offset(inv_input.site_indicator, {"pdf": "normal"})

    # make likelihood
    Y = add_model_data(inv_input.mf, "Y")
    error = add_model_data(inv_input.mf_error.astype("float32"), "error")
    min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

    #    sigma = make_sigma(inv_input.site_indicator, {"pdf": "inversegamma", "alpha": 2.5, "beta": 5}, inv_input.sigma_freq_index)

    epsilon = pm.Deterministic(
        "epsilon", pt.sqrt(error**2 + min_error**2), dims="nmeasure"
    )
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

    trace4 = pm.sample_prior_predictive(draws=5000)

# %%
plot_prior_preds(trace4, inv_input, skip=100)

# %%
with pm.Model() as model5:
    mu = add_linear_component(
        inv_input.H,
        data_name="hx",
        prior_args={"pdf": "exponential"},
        var_name="x",
        output_name="mu",
    )
    mu_bc = add_linear_component(
        inv_input.H_bc,
        data_name="hbc",
        prior_args=params["bcprior"],
        var_name="bc",
        output_name="mu_bc",
        compute_deterministic=True,
    )

    make_offset(inv_input.site_indicator, {"pdf": "normal"})

    # make likelihood
    Y = add_model_data(inv_input.mf, "Y")
    error = add_model_data(inv_input.mf_error.astype("float32"), "error")
    min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

    #    sigma = make_sigma(inv_input.site_indicator, {"pdf": "inversegamma", "alpha": 2.5, "beta": 5}, inv_input.sigma_freq_index)

    epsilon = pm.Deterministic(
        "epsilon", pt.sqrt(error**2 + min_error**2), dims="nmeasure"
    )
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

    trace5 = pm.sample_prior_predictive(draws=5000)

plot_prior_preds(trace5, inv_input, skip=100)

# %% [markdown]
# TODO:
# - Record deterministic for mu and compare mean/quantiles for mu with different priors. This will isolate the effects of the prior uncertainty.
# - Set up plots with quantiles/error bars and compare multiple models on one plot (fluxy functions useful?)
# - Create a class to organise experiments: include description, model, trace
# - Compute Bayesian R2 scores for prior predictives?

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Cluster setup

# %%
sf6_path = Path("/group/chem/acrg/PARIS_inversions/sf6/brendan_tests")

# !ls {sf6_path}

# %%
log_path = sf6_path / "simple_model_logs"
log_path.mkdir(exist_ok=True)

# %%
from dask_jobqueue import SLURMCluster
from dask.distributed import Client


cluster = SLURMCluster(
    processes=1,
    cores=8,
    memory="50GB",
    walltime="00:30:00",
    account="chem007981",
    log_directory=str(log_path),
)
client = Client(cluster)

# %%
client

# %%
client.restart()

# %%
client.close()
cluster.close()

# %%
cluster.scale(jobs=6)


# %%
def get_available_workers(client, cluster):
    available_workers = [
        v.get("id")
        for v in client.scheduler_info(n_workers=len(cluster.workers))[
            "workers"
        ].values()
    ]
    return available_workers


# %%
get_available_workers(client, cluster)

# %% [markdown]
# ## Model/sampling functions

# %%
params


# %%
def make_model(inv_input):
    with pm.Model() as model:
        mu = add_linear_component(
            inv_input.H,
            data_name="hx",
            prior_args={"pdf": "lognormal", "stdev": 2.0},
            var_name="x",
            output_name="mu",
        )
        mu_bc = add_linear_component(
            inv_input.H_bc,
            data_name="hbc",
            prior_args={"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1},
            var_name="bc",
            output_name="mu_bc",
            compute_deterministic=True,
        )

        mu_bc += make_offset(inv_input.site_indicator, {"pdf": "normal"})

        # make likelihood
        Y = add_model_data(inv_input.mf, "Y")
        error = add_model_data(inv_input.mf_error.astype("float32"), "error")
        min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

        sigma = make_sigma(
            inv_input.site_indicator,
            {"pdf": "inversegamma", "alpha": 2.5, "beta": 5},
            inv_input.sigma_freq_index,
        )

        epsilon = pm.Deterministic(
            "epsilon",
            pt.sqrt(error**2 + min_error**2 + 0.01 * sigma**2),
            dims="nmeasure",
        )
        pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

    return model


# %%
inversion_inputs.keys()

# %%
inv_input_obj = inversion_inputs["2015"]
inv_input = inv_input_obj.inv_input
params = inv_input_obj.params

# %%
sample_kwargs = default_sample_kwargs.copy()
sample_kwargs["blas_cores"] = 8
sample_kwargs

# %%
model = make_model(inv_input)
model

# %%
with model:
    trace = pm.sample_prior_predictive(draws=1000)

# %%
plot_prior_preds(trace, inv_input)

# %% [markdown]
# ## Running on SLURM

# %%
out_path = sf6_path / "simple_models" / "model2"
out_path.mkdir(parents=True, exist_ok=True)

# %%
model_string = """
def make_model(inv_input):
    with pm.Model() as model:
        mu = add_linear_component(
            inv_input.H,
            data_name="hx",
            prior_args={"pdf": "lognormal", "stdev": 2.0},
            var_name="x",
            output_name="mu",
        )
        mu_bc = add_linear_component(
            inv_input.H_bc,
            data_name="hbc",
            prior_args={"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1},
            var_name="bc",
            output_name="mu_bc",
            compute_deterministic=True,
        )

        mu_bc += make_offset(inv_input.site_indicator, {"pdf": "normal"})
    
        # make likelihood
        Y = add_model_data(inv_input.mf, "Y")
        error = add_model_data(inv_input.mf_error.astype("float32"), "error")
        min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

        sigma = make_sigma(inv_input.site_indicator, {"pdf": "inversegamma", "alpha": 2.5, "beta": 5}, inv_input.sigma_freq_index)

        epsilon = pm.Deterministic("epsilon", pt.sqrt(error**2 + min_error**2 + 0.01 * sigma**2), dims="nmeasure")
        pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

    return model
"""
with open(out_path / "model_code.txt", "wt") as f:
    f.write(model_string)


# %%
def run_inversion(inv_input, year, model_func=make_model, out_path=out_path, **kwargs):
    out_file = out_path / f"trace_{year}.nc"

    print("Staring sampling for year", year)
    with model_func(inv_input, **kwargs):
        idata = pm.sample(**sample_kwargs)
    print(f"Sampling for year {year} complete.")
    print(f"Writing idata for year {year}.")
    idata.to_netcdf(out_file)
    print(f"idata for year {year} saved to {out_file}")


# %%
from functools import partial

futures = []

for year in range(2015, 2020):
    inv_input = inversion_inputs[str(year)].inv_input
    func = partial(run_inversion, inv_input, year)
    future = client.submit(func)
    futures.append(future)

# %%
futures

# %%
client.refcount

# %%
from dask.distributed import Future


# %%
def get_futures(client):
    return [Future(key, client) for key in client.refcount]


# %%
get_futures(client)


# %% [markdown]
# ## Hierarchical sigma


# %%
def make_hierarchical_model(inv_input, sigma_prior: dict, sigma_hyper_prior: dict):
    with pm.Model() as model:
        mu = add_linear_component(
            inv_input.H,
            data_name="hx",
            prior_args={"pdf": "lognormal", "stdev": 2.0},
            var_name="x",
            output_name="mu",
        )
        mu_bc = add_linear_component(
            inv_input.H_bc,
            data_name="hbc",
            prior_args={"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1},
            var_name="bc",
            output_name="mu_bc",
            compute_deterministic=True,
        )

        offset = make_offset(inv_input.site_indicator, {"pdf": "normal"})

        mu_bc = mu_bc + offset

        # make likelihood
        Y = add_model_data(inv_input.mf, "Y")
        error = add_model_data(inv_input.mf_error.astype("float32"), "error")
        min_error = add_model_data(inv_input.min_error.astype("float32"), "min_error")

        sigma_hyper = parse_prior("sigma_hyper", sigma_hyper_prior)

        sigma_prior = sigma_prior or {"pdf": "halfnormal"}
        sigma0 = parse_prior("sigma0", sigma_prior, dims="nmeasure")
        sigma = pm.Deterministic("sigma", sigma_hyper * sigma0)

        epsilon = pm.Deterministic(
            "epsilon", pt.sqrt(error**2 + sigma**2 + min_error**2), dims="nmeasure"
        )
        pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")

    return model


# %%
sig_priors = dict(
    sigma_hyper_prior={"pdf": "inversegamma", "alpha": 3, "beta": 2},
    sigma_prior={"pdf": "halfstudentt", "nu": 2.0},
)

# %%
hmodel = make_hierarchical_model(
    inv_input,
    sigma_hyper_prior={"pdf": "inversegamma", "alpha": 3, "beta": 2},
    sigma_prior={"pdf": "halfnormal"},
)

with hmodel:
    htrace = pm.sample_prior_predictive(draws=1000)

# %% jupyter={"outputs_hidden": true}

plot_prior_preds(htrace, inv_input)

# %%
hout_path = sf6_path / "simple_models" / "hierarchical_model"
hout_path.mkdir(parents=True, exist_ok=True)

for year in range(2015, 2020):
    inv_input = inversion_inputs[str(year)].inv_input
    key = f"run_inversion-hierarchical_{year}"
    func = partial(
        run_inversion,
        inv_input=inv_input,
        year=year,
        model_func=make_hierarchical_model,
        out_path=hout_path,
        **sig_priors,
    )
    future = client.submit(func, key=key)
    futures.append(future)

# %%
futures = [f for f in futures if f.status != "cancelled"]

# %%
futures

# %%
client.processing()

# %%
inv_input

# %%
# #!ls -R {out_path.parent.parent/"brendan_tests"/"simple_models"}
# out_path = out_path.parent.parent/"brendan_tests"

# %%

# %%
from collections import defaultdict
import os

trace_files = defaultdict(list)


for root, _, files in os.walk(out_path / "simple_models"):
    for f in files:
        if f.endswith("nc"):
            trace_files[root].append(f)

# %%
import arviz as az

traces = defaultdict(list)

for k, v in trace_files.items():
    for f in v:
        traces[Path(k).name].append(az.InferenceData.from_netcdf(Path(k) / f))

# %%
traces

# %%
sum0 = performance_summary(traces["hierarchical_model"][0])

# %%
inv_input0 = inversion_inputs["2018"].inv_input

# %%
hmodel = make_hierarchical_model(inv_input0, **sig_priors)
trace0 = traces["hierarchical_model"][0]
trace0.posterior

# %%
# del trace0["prior"]
# del trace0["prior_predictive"]
# del trace0["posterior_predictive"]
with hmodel:
    trace0.extend(pm.sample_prior_predictive(draws=1000))
    trace0.extend(pm.sample_posterior_predictive(trace=trace0))

# %%
params = inversion_info["2018"].params

# %%
from openghg.retrieve import *

flux = get_flux(species="sf6", domain="europe", source="flat-annual-total").data.flux

# %%
inv_out = make_inv_out(inv_input0, trace0, flux, params)

# %%
from openghg_inversions.postprocessing.countries import Countries

default_country_file = Path(
    "/group/chem/acrg/LPDM/countries/country_EUROPE_EEZ_PARIS_gapfilled.nc"
)
default_countries = ["BEL", "NLD", "BENELUX", "DEU", "FRA", "GBR", "IRL", "NW_EU"]


countries = Countries.from_file(
    country_file=default_country_file, country_code="alpha3"
)

# %%
inv_out.trace = inv_out.trace.isel(chain=0)

# %%
from openghg_inversions.postprocessing.make_paris_outputs import (
    paris_concentration_outputs,
)

# conc, flux = make_paris_outputs(inv_out, default_country_file, time_point="start", inversion_grid=False)
conc = paris_concentration_outputs(inv_out)

# %%
conc = conc.compute()

# %%
conc

# %%
fig, axs = plt.subplots(3, 2, figsize=(15, 22))

for site, ax in zip(range(6), axs.flat):
    co_sel = conc.isel(nsite=site)
    co_sel.Yobs.plot(ax=ax, label="y obs")
    co_sel.Yapost.plot(ax=ax, label="a post")
    ax.fill_between(
        co_sel.time.values,
        co_sel.qYapost.isel(percentile=0).values,
        co_sel.qYapost.isel(percentile=1).values,
        alpha=0.5,
        color="orange",
        interpolate=True,
    )
    ax.legend()
    ax.set_title(conc.sitenames.values[site])

# %%
# country_df = countries.get_country_trace(inv_out).mean("draw").to_series().unstack()

# %%
# %run inversions_experimental_code/basis_functions.py

# %%
bf = BasisFunctions(inv_input0.basis_flat, flux.sel(time=["2018-01-01"]))

# %%
flux_post_mean = bf.interpolate(inv_out.trace.posterior.x.mean(["draw"]), flux=True)

# %%
import geopandas as gpd

world = gpd.read_file("natural_earth_50.zip")

# %%
fig, ax = plt.subplots(figsize=(15, 7))

lat_slice = slice(37, None)
lon_slice = slice(-14, 25)

lat_min, lat_max = lat_slice.start, lat_slice.stop
lon_min, lon_max = lon_slice.start, lon_slice.stop

# vmin, vmax = -39, -26
vmin, vmax = 0, 10e-13

flux_post_mean.sel(lat=lat_slice, lon=lon_slice).plot(ax=ax)  # , vmin=vmin, vmax=vmax)

world.boundary.plot(
    ax=ax, linewidth=0.6, edgecolor="white"
)  # or .plot(facecolor='none')
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# %%
