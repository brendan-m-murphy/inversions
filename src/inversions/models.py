from pathlib import Path
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import pytensor
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable

# need to set config before importing PyMC
pytensor.config.floatX = "float32"
pytensor.config.warn_float64 = "warn"

import pymc as pm
from pymc.distributions import continuous


from openghg_inversions.hbmcmc.run_hbmcmc import hbmcmc_extract_param
from openghg_inversions.inversion_data.get_data import data_processing_surface_notracer
from openghg.util import split_function_inputs
from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.inversion_data import (
    data_processing_surface_notracer,
    load_merged_data,
)
from openghg_inversions.filters import filtering
from openghg_inversions.model_error import (
    residual_error_method,
    percentile_error_method,
    setup_min_error,
)
from openghg_inversions.postprocessing.inversion_output import (
    make_inv_out_for_fixed_basis_mcmc,
)
from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_paris_outputs import make_paris_outputs


def get_fp_data_dict(ini_path: str | Path) -> dict:
    params = hbmcmc_extract_param(ini_path, "fixedbasisMCMC")
    dpsnt_params, _ = split_function_inputs(params, data_processing_surface_notracer)

    fp_all, sites, *_ = data_processing_surface_notracer(
        averagingerror=params["averaging_error"], **dpsnt_params
    )

    bfw_params, _ = split_function_inputs(params, basis_functions_wrapper)
    fp_data = basis_functions_wrapper(
        fp_all=fp_all, fix_outer_regions=params["fix_basis_outer_regions"], **bfw_params
    )

    # for site in sites:
    #     fp_data[site] = fp_data[site].compute()

    fp_data_filt = filtering(fp_data, params["filters"])

    return fp_data_filt


def get_fp_all(ini_path: str | Path) -> tuple[dict, list]:
    params = hbmcmc_extract_param(ini_path, "fixedbasisMCMC")
    dpsnt_params, _ = split_function_inputs(params, data_processing_surface_notracer)

    fp_all, sites, *_ = data_processing_surface_notracer(
        averagingerror=params["averaging_error"], **dpsnt_params
    )

    return fp_all, sites


def apply_basis(params: dict, fp_all: dict) -> dict:
    bfw_params, _ = split_function_inputs(params, basis_functions_wrapper)
    fp_data = basis_functions_wrapper(
        fp_all=fp_all, fix_outer_regions=params["fix_basis_outer_regions"], **bfw_params
    )
    return fp_data


# TODO: add helper for saving/loading merged data


def stack_fp_data(
    fp_data: dict, keep_vars: list[str] | None = None, sites: list[str] | None = None
) -> xr.Dataset:
    if sites is None:
        sites = [k for k in fp_data if not str(k).startswith(".")]

    if keep_vars is None:
        keep_vars = [
            "H",
            "H_bc",
            "mf",
            "mf_error",
            "mf_repeatability",
            "mf_variability",
        ]

    scenarios = [
        fp_data[site][keep_vars].expand_dims({"site": [site]}) for site in sites
    ]
    combined_scenario = (
        xr.concat(scenarios, dim="site")
        .stack(nmeasure=("site", "time"))
        .dropna("nmeasure")
    )
    return combined_scenario


def xr_unique_inv(da: xr.DataArray) -> xr.DataArray:
    def np_unique_inv(arr: np.ndarray) -> np.ndarray:
        _, inv = np.unique(arr, return_inverse=True)
        return inv

    return xr.apply_ufunc(np_unique_inv, da)


def make_site_indicator(site_coord: xr.DataArray) -> xr.DataArray:
    return xr_unique_inv(site_coord).rename("site_indicator")


def make_freq_indicator(
    time: xr.DataArray, freq: Literal["monthly"] | str
) -> xr.DataArray:
    if freq == "monthly":
        return (
            time.dt.month
            - time.min().dt.month
            + 12 * (time.dt.year - time.min().dt.year)
        )
    return xr_unique_inv(time.dt.floor(freq))


def setup_sigma_freq(
    time: xr.DataArray, freq: Literal["monthly"] | str | None = None
) -> xr.DataArray:
    if freq is None:
        res = xr.zeros_like(time)
    else:
        res = make_freq_indicator(time, freq)
    return res.rename("sigma_freq_index")


# TODO: use same coords even if freq is not None
def setup_bc(H_bc, freq=None):
    if freq is None:
        return H_bc
    freq_arr = make_freq_indicator(H_bc.time, freq)
    dums = get_xr_dummies(freq_arr, return_sparse=False, cat_dim="bc_time")
    return (H_bc.rename(bc_region="bc_curtain") * dums).stack(
        bc_region=("bc_curtain", "bc_time")
    )


# Creating data for inversion
# TODO: need to use dimension names expected by InversionOutput
# TODO: missing some min_error options (need helper function for this...)
# TODO: InversionInput object?
def make_inv_inputs(
    fp_data,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    min_error: str | dict[str, float] | float = 0.0,
    min_error_per_site: bool = True,
):
    ds = stack_fp_data(fp_data)

    # add BC freq
    if bc_freq is not None:
        temp = setup_bc(ds.H_bc, bc_freq)
        ds = ds.drop_dims("bc_region")
        ds["H_bc"] = temp

    ds["site_indicator"] = make_site_indicator(ds.site)
    ds["sigma_freq_index"] = setup_sigma_freq(ds.time, freq=sigma_freq)

    ds["basis_flat"] = fp_data[".basis"]

    # set up min error
    if isinstance(min_error, float):
        ds["min_error"] = min_error * xr.ones_like(ds.mf)
    elif isinstance(min_error, dict):
        sites = np.unique(ds.site)
        err_per_site = np.array([min_error[site] for site in sites])
        ds["min_error"] = xr.apply_ufunc(
            lambda x: setup_min_error(err_per_site, x), ds.site_indicator
        )
    elif min_error == "percentile":
        perc_err = percentile_error_method(fp_data)
        ds["min_error"] = xr.apply_ufunc(
            lambda x: setup_min_error(perc_err, x), ds.site_indicator
        )
    else:
        raise ValueError(f"Option '{min_error}' is not valid.")

    return ds


# Prior code from "model configs" branch
# type alias for prior args
PriorArgs = dict[str, str | float]


def lognormal_mu_sigma(mean: float, stdev: float) -> tuple[float, float]:
    """Return the pymc `mu` and `sigma` parameters that give a log normal distribution
    with the given mean and stdev.

    Args:
        mean: desired mean of log normal
        stdev: desired standard deviation of log normal

    Returns:
        tuple (mu, sigma), where `pymc.LogNormal(mu, sigma)` has the given mean and stdev.

    Formulas for log normal mean and variance:

    mean = exp(mu + 0.5 * sigma ** 2)
    stdev ** 2 = var = exp(2*mu + sigma ** 2) * (exp(sigma ** 2) - 1)

    This gives linear equations for `mu` and `sigma ** 2`:

    mu + 0.5 * sigma ** 2 = log(mean)
    sigma ** 2 = log(1 + (stdev / mean)**2)

    So

    mu = log(mean) - 0.5 * log(1 + (stdev/mean)**2)
    sigma = sqrt(log(1 + (stdev / mean)**2))
    """
    var = np.log(1 + (stdev / mean) ** 2)
    mu = np.log(mean) - 0.5 * var
    sigma = np.sqrt(var)
    return mu, sigma


def update_log_normal_prior(prior_params):
    if "stdev" in prior_params:
        stdev = float(prior_params["stdev"])
        mean = float(prior_params.get("mean", 1.0))

        mu, sigma = lognormal_mu_sigma(mean, stdev)
        prior_params["mu"] = mu
        prior_params["sigma"] = sigma

        del prior_params["stdev"]
        if "mean" in prior_params:
            del prior_params["mean"]


def parse_prior(name: str, prior_params: PriorArgs, **kwargs) -> TensorVariable:
    """Parses all PyMC continuous distributions:
    https://docs.pymc.io/api/distributions/continuous.html.

    Args:
        name:
          name of variable in the pymc model
        prior_params:
          dict of parameters for the distribution, including 'pdf' for the distribution to use.
          The value of `prior_params["pdf"]` must match the name of a PyMC continuous
          distribution: https://docs.pymc.io/api/distributions/continuous.html
        **kwargs: for instance, `shape` or `dims`
    Returns:
        continuous PyMC distribution

    For example:
    ```
    params = {"pdf": "uniform", "lower": 0.0, "upper": 1.0}
    parse_prior("x", params, shape=(20, 20))
    ```
    will create a 20 x 20 array of uniform random variables.
    Alternatively,
    ```
    params = {"pdf": "uniform", "lower": 0.0, "upper": 1.0}
    parse_prior("x", params, dims="nmeasure"))
    ```
    will create an array of uniform random variables with the same shape
    as the dimension coordinate `nmeasure`. This can be used if `pm.Model`
    is provided with coordinates.

    Note: `parse_prior` must be called inside a `pm.Model` context (i.e. after `with pm.Model()`)
    has an important side-effect of registering the random variable with the model.
    """
    # create dict to lookup continuous PyMC distributions by name, ignoring case
    pdf_dict = {cd.lower(): cd for cd in continuous.__all__}

    params = prior_params.copy()
    pdf = str(params.pop("pdf")).lower()  # str is just for typing...

    # special processing for lognormals
    if pdf == "lognormal":
        update_log_normal_prior(params)

        if params.get("reparameterise", False):
            temp = pm.Normal(f"{name}0", 0, 1, **kwargs)
            return pm.Deterministic(
                name, pt.exp(params["mu"] + params["sigma"] * temp), **kwargs
            )

    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except AttributeError:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        )

    return dist(name, **params, **kwargs)


# TODO: update offset code?
def make_offset(
    site_indicator: np.ndarray,
    prior_args: dict,
    name: str = "offset",
    output_dim: str = "nmeasure",
    drop_first: bool = False,
    compute_deterministic: bool = False,
) -> TensorVariable:
    """Create an offset inside a PyMC model.

    Note: this *must* be called from inside a PyMC `model` context.

    Args:
        site_indicator: array with same length as obs, with integers to indicator which site
          an observation belongs to
        prior_args: dict of prior args for offset prior
        name: name for offset in PyMC model
        output_dim: name of dimension for output
        drop_first: if True, set first site's offset to zero

    Returns:
        TensorVariable containing offset vector (to add to modelled observations).
    """
    sites = np.unique(site_indicator)

    n_sites = len(sites) - 1 if drop_first else len(sites)

    matrix = pd.get_dummies(site_indicator, drop_first=drop_first, dtype=int).values
    offset_x = parse_prior(name + "0", prior_args, shape=n_sites)

    if compute_deterministic:
        return pm.Deterministic(name, pt.dot(matrix, offset_x), dims=output_dim)
    return pt.dot(matrix, offset_x)


# Helpers for making components
def add_coords(coords: dict[str, np.ndarray]):
    with pm.modelcontext(None) as model:
        model.add_coords(coords)


def add_model_data(data: xr.DataArray, name: str | None = None):
    # TODO: extend this to a Dataset (or DataTree?)
    # TODO: name before data (like PyMC...)
    name = name or data.name

    if name is None:
        raise ValueError("Data must have a name if a name is not provided.")

    model = pm.modelcontext(None)
    if name in model:
        return model[name]

    coords = {dim: np.arange(size) for dim, size in data.sizes.items()}
    add_coords(coords)

    return pm.Data(name, data.values, dims=data.dims)


def add_linear_component(
    data: xr.DataArray,
    data_name: str,
    prior_args: dict,
    var_name: str,
    output_name: str,
    output_dim="nmeasure",
    compute_deterministic: bool = False,
):
    data = data.transpose(output_dim, ...)
    h = add_model_data(data, data_name)
    input_dims = tuple(dim for dim in data.dims if dim != output_dim)
    x = parse_prior(var_name, prior_args, dims=input_dims)
    if compute_deterministic:
        return pm.Deterministic(output_name, pt.dot(h, x), dims=output_dim)
    return pt.dot(h, x)


def make_sigma(
    site_indicator: xr.DataArray,
    prior_args: dict,
    sigma_freq_index: xr.DataArray,
    per_site: bool = True,
    name: str = "sigma",
    output_dim: str = "nmeasure",
    compute_deterministic: bool = False,
):
    if not per_site:
        raise NotImplementedError()

    freq_index = add_model_data(sigma_freq_index, "sigma_freq_index")
    sites = add_model_data(site_indicator, "site_indicator")

    coords = {
        f"n{name}_site": np.unique(site_indicator),
        f"n{name}_time": np.unique(sigma_freq_index),
    }

    add_coords(coords)

    sigma = parse_prior(name, prior_args, dims=tuple(coords.keys()))

    if compute_deterministic:
        return pm.Deterministic(
            f"{name}_obs_aligned", sigma[sites, freq_index], dims=output_dim
        )
    return sigma[sites, freq_index]


def add_new_likelihood(
    combined_scenario: xr.Dataset,
    sigma_prior: dict,
    mu: pt.TensorVariable,
    mu_bc: pt.TensorVariable,
    power: float = 1.5,
    min_error_prior: dict | None = None,
    const_min_error: float | None = None,
):
    Y = add_model_data(combined_scenario.mf, "Y")
    error = add_model_data(combined_scenario.mf_error, "error")

    if const_min_error is not None:
        min_error = add_model_data(
            const_min_error * xr.ones_like(combined_scenario.min_error), "min_error"
        )
    else:
        min_error = add_model_data(combined_scenario.min_error, "min_error")

    if min_error_prior is not None:
        min_error_scaling = make_sigma(
            name="min_error_scaling",
            site_indicator=combined_scenario.site_indicator,
            prior_args=min_error_prior,
            sigma_freq_index=combined_scenario.sigma_freq_index,
        )
    else:
        min_error_scaling = 1.0

    pe = pt.switch(pt.gt(Y, mu_bc), Y - mu_bc, 0)
    sigma = make_sigma(
        combined_scenario.site_indicator,
        sigma_prior,
        combined_scenario.sigma_freq_index,
    )
    pe_scaled = sigma * pe
    eps = pt.sqrt(error**2 + pt.pow(pe_scaled, power) + min_error**2)
    epsilon = pm.Deterministic("epsilon", eps, dims="nmeasure")
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")


def add_old_likelihood(
    combined_scenario: xr.Dataset,
    sigma_prior: dict,
    mu: pt.TensorVariable,
    mu_bc: pt.TensorVariable,
    pollution_events_from_obs: bool = True,
    min_error_prior: dict | None = None,
    const_min_error: float | None = None,
    power: float = 1.99,
):
    Y = add_model_data(combined_scenario.mf, "Y")
    error = add_model_data(combined_scenario.mf_error, "error")

    if const_min_error is not None:
        min_error = add_model_data(
            const_min_error * xr.ones_like(combined_scenario.min_error), "min_error"
        )
    else:
        min_error = add_model_data(combined_scenario.min_error, "min_error")

    if min_error_prior is not None:
        min_error_scaling = make_sigma(
            name="min_error_scaling",
            site_indicator=combined_scenario.site_indicator,
            prior_args=min_error_prior,
            sigma_freq_index=combined_scenario.sigma_freq_index,
        )
    else:
        min_error_scaling = 1.0

    pe = pt.abs(Y - mu_bc) if pollution_events_from_obs else pt.abs(mu)
    sigma = make_sigma(
        combined_scenario.site_indicator,
        sigma_prior,
        combined_scenario.sigma_freq_index,
    )
    pe_scaled = sigma * pe
    eps = pt.maximum(
        pt.sqrt(error**2 + pt.pow(pe_scaled, power)), min_error_scaling * min_error
    )
    epsilon = pm.Deterministic("epsilon", eps, dims="nmeasure")
    pm.Normal("y", mu=mu + mu_bc, sigma=epsilon, observed=Y, dims="nmeasure")


# Creating model
def make_model(
    inv_input: xr.Dataset,
    params: dict,
    new_likelihood: bool = False,
    power: float = 1.5,
) -> pm.Model:
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
        )
        if new_likelihood is True:
            add_new_likelihood(
                inv_input,
                sigma_prior=params["sigprior"],
                mu=mu,
                mu_bc=mu_bc,
                power=power,
            )
        else:
            add_old_likelihood(
                inv_input, sigma_prior=params["sigprior"], mu=mu, mu_bc=mu_bc
            )
    return model


# Sampling

default_sample_kwargs = dict(
    draws=1000,
    tune=2000,
    chains=4,
    progressbar=False,
    cores=4,
    nuts_sampler="numpyro",
    idata_kwargs={"log_likelihood": True},
)


def sample(
    model: pm.Model, draws: int = 1000, tune: int = 1000, **kwargs
) -> az.InferenceData:
    if draws in kwargs:
        draws = kwargs.pop("draws")
    if tune in kwargs:
        tune = kwargs.pop("tune")

    with model:
        trace = pm.sample(draws=draws, tune=tune, **kwargs)
        trace = pm.sample_posterior_predictive(trace, extend_inferencedata=True)
        trace.extend(pm.sample_prior_predictive(draws=draws))

    return trace


# Performance summary
def performance_summary(idata):
    print("Divergences:", idata.sample_stats.diverging.sum().values)
    summary = az.summary(idata, var_names=["x", "bc", "sigma"])
    print("r_hat:")
    print(summary.r_hat.describe())
    print()
    waic = az.waic(idata)
    loo = az.loo(idata, pointwise=True)
    print("waic:")
    print(waic)
    print("loo:")
    print(loo)
    return summary, waic, loo


# TODO bad pareto k values


# Making outputs
def flux_from_fp_data(fp_data: dict) -> xr.DataArray:
    return next(iter(fp_data[".flux"].values())).data.flux


def make_inv_out(ds, idata, flux, params):
    ds = ds.rename({"region": "nx", "bc_region": "nbc"})
    basis = get_xr_dummies(ds.basis_flat, cat_dim="nx", categories=ds.nx.values)
    inv_out = InversionOutput(
        obs=ds.mf,
        obs_err=ds.mf_error,
        obs_repeatability=ds.mf_repeatability,
        obs_variability=ds.mf_variability,
        flux=flux,
        basis=basis,
        trace=idata,
        site_indicators=ds.site_indicator,
        times=ds.time,
        start_date=params["start_date"],
        end_date=params["end_date"],
        species=params["species"],
        domain=params["domain"],
        site_names={i: s for i, s in enumerate(np.unique(ds.site))},
    )
    return inv_out


default_country_file = Path(
    "/group/chem/acrg/LPDM/countries/country_EUROPE_EEZ_PARIS_gapfilled.nc"
)
default_countries = ["BEL", "NLD", "BENELUX", "DEU", "FRA", "GBR", "IRL", "NW_EU"]


# Full pipeline


def run_inversion(
    fp_data: dict,
    params: dict,
    new_likelihood: bool = False,
    power: float = 1.5,
    sampling_kwargs: dict | None = None,
):
    # get inversion inputs, parse params
    min_error = params.get("calculate_min_error") or params.get("min_error") or 0.0
    inv_input = make_inv_inputs(
        fp_data,
        bc_freq=params.get("bc_freq"),
        sigma_freq=params.get("sigma_freq"),
        min_error=min_error,
    )

    # make model, update params (e.g. reparam lognormal)
    for prior in ("xprior", "bcprior"):
        if params[prior]["pdf"] == "lognormal" and params.get(
            "reparameterise_log_normal"
        ):
            params[prior]["reparameterise"] = True

    model = make_model(inv_input, params, new_likelihood=new_likelihood, power=power)

    # sampling
    sampling_kwargs = sampling_kwargs or default_sample_kwargs.copy()
    if "nit" in params:
        sampling_kwargs["draws"] = params["nit"]
    if "tune" in params:
        sampling_kwargs["tune"] = params["tune"]

    idata = sample(model, **sampling_kwargs)

    summary, waic, loo = performance_summary(idata)

    return model, idata, summary, waic, loo


# Cross validation

# Possible methods (should be stratified by site):
# - random splits (for K-fold or train/validate)
# - split by day of week
# - split by week (e.g. every fourth week, etc)
# - split by month (either fitting all, then testing on missing block, or train on months 1 to N and test on month N+1); could use weeks
#   for monthly inversions
# - leave one site out (could be tricky with offset) (or group of sites)


# Sensitivity tests
#
# Power scaling of prior/likelihood?
# ...this is built into arviz and called `psens`
