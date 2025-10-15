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
# # SF6 Model Testing
#
# Our model is performing poorly when modelling SF6, especially for country totals in Germany, which are too low.
#
# We assign very large uncertainties to point source (temporally spiky, but probably location fixed) emissions, essentially discounting them. Since the bulk of SF6 is emitted this way (except for certain factories), we underestimate emissions.
#
# We seem to have similar problems with PFC-218 and HFC-23(?).
#
# ## General issues with RHIME
#
# 1. Our basis functions don't split by land and sea (and this only ever worked for "weighted" basis functions)
# 2. We have no cross validation, so we can't objectively assess our model performance (or compare between models)
# 3. Our uncertainties seem miscalibrated:
#    - "min model error" is too low for SF6, and the reasoning behind our choice is somewhat weak
#    - our pollution event error is very high, and our (hyper)prior doesn't seem to decrease it
# 4. Our prior uncertainties are very large, but our country emissions have small uncertainties
# 5. We don't rescale prior uncertainties to respect the choice of basis functions, and we may be under-estimating our uncertainties by not accounting for "aggregation error" (but for our model, I'm unsure of where to add this)
# 6. We have no correlations in any of our priors
#    - ELRIS uses spatial correlation in their prior uncertainties
#    - InTEM (and ELRIS) have temporal (and spatial between sites) correlation in their likelihoods
# 7. Our boundary conditions use the same scaling factors for all sites; also our BC have no temporal correlation, we could enforce this by replacing our "dummy matrix" with one that does some temporal smoothing
# 8. We have small country total uncertainties. Could this be caused by mapping back to the transport model resolution to make the country traces without adding aggregation error?
#
#
# ## Particular issues with SF6
#
# 1. If we omit flask data, our country totals are more in line with InTEM and ELRIS from about 2021 onwards
# 2. Observations at HFD (and RGL?) have substantial biases (but we can account for this with a bias term)
#
# ## What we've looked at so far
#
# - Adding a bias is a big improvement
# - Omitting flask data improves agreement with InTEM and ELRIS, but that is worrying
# - Setting a constant value for min error tends to improve agreement (and increase uncertainty of the country totals)
#
# ## Ideas
#
# ### Use different settings for the priors
#
# - Less prior uncertainty? (Or calibrate this for a certain total uncertainty... e.g. 100% like InTEM and ELRIS)
# - More obs (model-data mismatch) uncertainty?
#
# ### Insert model error more directly
#
# Multiplying `H @ x` by random variable (matching the output dimension `nmeasure`) centered on 1. This would apply model error more directly by saying we are uncertain about the values `H @ x` of enhancement above baseline. A similar method could be used to account for error at baseline times (uncertainty would have to be *very* small here). The likelihood would just contain instrument errors, etc.
#
# - One question here is whether multiplicative errors are appropriate here.
# - We could try to apply the uncertainty to the elements of `H` itself, but this is complicated somewhat by combining footprint and flux ahead of time. Also, if we don't introduce correlation between this new error aligned with `nmeasure` and the existing prior on `x`, then there isn't really anything gained.
# - The form for this new uncertainty isn't clear. It would make sense to consider influences from farther away from the measurement site, or further back in time, as less certain. But I'm not sure we have access to the right data to compute this. Most likely it will be (truncated) Gaussian with mean 1 and fairly small uncertainties
# - It might make sense for each site to have extra scaling factors for particular basis regions (so... basically partial pooling by site?)
#
# ### Fix basis functions
#
# 1. fix land/sea split
# 2. filter first to avoid using footprints that aren't used in the inversion to determine the average footprint times flux
# 3. adjust uncertainties appropriately
#    - If we're reporting at the transport model resolution, there is an aggregation error term we probably need to add (but where, I'm not sure... literature is all for Gaussian models)
#    - A sum of lognormal RVs isn't lognormal (although there is a decent approximation using a lognormal), so the adjustment might not be straightforward
# 4. (more work) use local search/simulated annealing to optimise results from quadtree or weighted
#
# ### Use cross validation
#
# - This is probably not too hard to setup with the helper functions from `likelihood_tests.py`.
# - If it turns out that WAIC is a reasonable proxy, we could use that instead of K-fold CV
#
# ### Look at synthetic data
#
# How does our model handle obs generated by an intermittent point source?
#
# ### Look at a simpler model
#
# - Can we do reasonably well with a Gaussian likelihood? Or Student-t likelihood?
# - Use CV to tune the ratio between prior uncertainty and model-data mismatch uncertainty?
#
# ### Add correlations
#
# - temporal correlation between consecutive obs
# - simpler option: increase averaging period
# - spatial correlation between stations
# - spatial correlation between basis regions
#
# ### Flask uncertainties
#
# We haven't added any variability term, like InTEM (and ELRIS?).
#
# ### Data selection
#
# - Are there problematic observations?
# - Should we select e.g. 0:00-4:00 local time for mountain sites?
#
# ## Plan/Workflow
#
# ### How are we assessing performance?
# 1. prior/posterior fit to obs
# 2. "agreement" (with other models, inventory) for country totals
# 3. CV scores
# 4. Degrees of freedom for signal? (Need to balance with other error contributions?)
# 5. WAIC or LOO-PIT?
#
# ### How will we assess convergence?
#
# - Arviz summary stats
# - Divergences
# - Possibly running extra long runs to verify shorter runs are consistent
#
# ### What data will we train on? validate/test on?
#
# - We should probably run multiple years. 2015-2019 seem problematic for RHIME.
# - We don't have flask data for 2015-2019, so that simplifies things.
#

# %% [markdown]
# # Baseline
#
# Goals:
# - get a sense of the performance of our current model
# - look at Helene's work

# %%
from pathlib import Path
sf6_path = Path("/group/chem/acrg/PARIS_inversions/sf6/")
# !ls -lst {sf6_path}

# %%
# model_dirs = !ls {sf6_path} | grep "RHIME_"
for i, md in enumerate(model_dirs):
    print(i, md)

# %%
model_dict = {"base": model_dirs[9], "bias": model_dirs[14], "pefalse": model_dirs[8], "biaspefalse": model_dirs[11]}

# %% [markdown]
# We need to read in data, but the fluxy `read_model_output` function assumes a different directory structure, so we'll just try to replicate its output.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# %%
ds_all = {k: xr.open_mfdataset(str(sf6_path / v / "*flux*.nc")) for k, v in model_dict.items()}

# %%
import logging
from fluxy.config import set_print_settings
from fluxy.io import read_config_files
from fluxy.test_utils import data_dir
from fluxy.config import set_model_colors
from fluxy.config import set_model_labels

# Logging settings, fluxy uses the standard logging module
logging.basicConfig(level=logging.WARNING)


# Path to test files
# data_dir = data_dir


# Set presentation_mode to True for bigger fonts
presentation_mode = False

# If False, uses default labels. If True, uses labels in models_info.json.
get_labels_from_file = False


# Group the models of interest in meaningful experiment names
models = list(ds_all.keys())

# This is the substance available in the test data.
species = "sf6" 

# Read the default configuration files
config_data = read_config_files()

annotate_coords = set_print_settings(presentation_mode)
model_colors = set_model_colors(models)
model_labels = set_model_labels(models, config_data, get_labels_from_file)

# %%
model_labels = {"base": "PEFO True", "bias": "PEFO True + Bias", "pefalse": "PEFO False", "biaspefalse": "PEFO False + Bias"}

# %%
from fluxy.io import edit_vars_and_attributes

ds_all = {k : edit_vars_and_attributes(v, k, "yearly", "flux", config_data.get("regions_info", {}), config_data.get("site_info", {}), species=species)
          for k, v in ds_all.items()}

# %%
# from fluxy "example basics" ipynb
from fluxy.io import read_model_output
from fluxy.operators.select import slice_flux


###################################
### edit variables in this block
regions = ["DEU", "FRA", "IRL", "GBR", "BENELUX", "NW_EU2"]
# inversion period, must be a string or a list of the same length as models, e.g. ['monthly','yearly']
period = "yearly"
# desired units for the plot. Add "CO2-eq" to convert the mass to CO2 equivalent (e.g. "Gg CO2-eq yr-1")
country_flux_units_print = "Tg yr-1 CO2-eq"
# inclusive
start_date = "2015-01-01"
# not inclusive
end_date = "2020-01-01"


###################################



ds_all_flux = ds_all #read_model_output(...


# Select only the time period and regions of interest
ds_all_flux_scaled = slice_flux(
    ds_all_flux,
    config_data,
    start_date,
    end_date,
    species=species,
    country_flux_units_print=country_flux_units_print,
)

# %% [markdown]
# ### Plot timeseries of country flux

# %%
from fluxy.plots.flux_timeseries import plot_country_flux

###################################
### edit variables in this block
plot_inventory = False
inventory_years = (
    None  # If None, plots most recent. Or can choose list of years: ['2022','2023']
)
inventory_filename = "UNFCCC_inventory"  # Full filename = {inventory_filename}_{species}_{inventory_year}
fix_y_axes = False  # if True: all y axis limits are the same, if False: each y axis is relative to the data
# if a list of floats (e.g. [0,0.1]) applies these limit to all axes
add_prior = True  # if True: plots prior as dashed lines
add_prior_unc = False  # if True: plots prior uncertainty as shaded area
set_global_leg = (
    True  # If True, plots one single legend instead of one legend per subplot.
)
country_codes_as_titles = False  # If True, lists 3-letter country codes under region names in subplot titles. Set to None for no title.
plot_separate = True  # If True, includes all model results as separate lines (or insert a list of boolean of the same length as models to specify which models to plot)
plot_combined = False  # If True, combined results, averaged from all models (or insert a list of boolean of the same length as models to specify which models to combine)
resample = None  # If None, no resample is done. Else resample the data to the given period (options 'year' and 'season' for yearly and seasonal averages)
resample_uncert_correlation = False  # If True, uses mean uncertainty during resampling, if False, recalculates uncertainty assuming no correlation.
plot_resample_and_original = (
    False  # If True, plots both the resampled and original data
)
annex_mode = (
    False  # If True, replace the labels with more concise versions for NID Annexes.
)
rolling_mean = False  ##If True, calculates a rolling mean of the data (insert a list of boolean of the same length as models to specify the models to smooth)
###################################

fig = plot_country_flux(
    ds_all_flux_scaled,
    species,
    plot_regions=regions,
#    config_data=config_data,  # turn off, else it looks for BEL-LUX-NEL, etc instead of BENELUX
    model_colors=model_colors,
    model_labels=model_labels,
    start_date=start_date,
    end_date=end_date,
    annex_mode=annex_mode,
    plot_inventory=plot_inventory,
    inventory_years=inventory_years,
    inventory_filename=inventory_filename,
    data_dir=data_dir,
    fix_y_axes=fix_y_axes,
    add_prior=add_prior,
    add_prior_unc=add_prior_unc,
    set_global_leg=set_global_leg,
    country_codes_as_titles=country_codes_as_titles,
    plot_separate=plot_separate,
    plot_combined=plot_combined,
    resample=resample,
    resample_uncert_correlation=resample_uncert_correlation,
    plot_resample_and_original=plot_resample_and_original,
    rolling_mean=rolling_mean,
)

# %% [markdown]
# We'll use PEFO as shorthand for "pollution events from obs".
#
# So we can see:
# 1. For DEU, and to some extent FRA, there is a large gap between PEFO = True (lower totals) and PEFO = False (higher totals)
# 2. In DEU, adding a bias picks up a spike in 2017 (for both PEFO T/F)
# 3. In GBR, PEFO T/F doesn't matter too much, but the bias has a large effect. (Except in 2015, totals are *higher* for PEFO True)
# 4. In FRA, the effects are mixed: the bias shifts emissions up, and flattens out the trend in 2018-2019, and PEFO False shifts up emissions too, especially in early years. These effects are *not additive*, so emissions with PEFO False with bias are increased slightly (vs. w/o bias), and flattened, but the shift isn't as large as the sum of the separate increases from PEFO F and bias.

# %% [markdown]
# ### Concentration timeseries
#
# The best method we have to assess our model is comparing the observations to the posterior modelled observations.
#
# So... let's plot these, along with modelled baselines.

# %%
ds_conc_all = {k: xr.concat([xr.open_dataset(sf6_path / v / f"SF6_EUROPE_PARIS_conc_20{yr}-01-01.nc").rename(sitenames="site").swap_dims(nsite="site") for yr in ["15", "16", "17", "18", "19"]], dim="time", join="outer") for k, v in model_dict.items()}

# %%
ds_conc_all = {k: v.swap_dims(site="nsite").rename(site="sitenames") for k, v in ds_conc_all.items()}

# %%
ds_conc_all = {k : edit_vars_and_attributes(v, k, "yearly", "concentration", config_data.get("regions_info", {}), config_data.get("site_info", {}), species=species)
          for k, v in ds_conc_all.items()}

# %%
ds_conc_all["base"].platform.values

# %%
from fluxy.operators.select import slice_mf

###################################
site = "JFJ"
# inversion period, must be a string or a list of the same length as models, e.g. ['monthly','yearly']
period = "yearly"
mf_units_print = "ppt"
start_date = "2018-01-01"
end_date = "2019-01-01"
#'MHD', 'JFJ' or 'CMN'. If None, does not mask by baseline time
baseline_site = None
baseline_filename = "InTEM_baseline_timestamps"
###################################

ds_all_mf = ds_conc_all

ds_all_mf_sliced = slice_mf(
    ds_all_mf.copy(),
    start_date,
    end_date,
    site,
    baseline_site=baseline_site,
    baseline_filename=baseline_filename,
    data_dir=data_dir,
    mf_units_print=mf_units_print,
)

# %%
# trying to restore stacked index coord...
ds = next(iter(ds_all_mf_sliced.values()))
mindex = pd.MultiIndex.from_arrays([ds.platform.values[ds.number_of_identifier.values], ds.time.values], names=["platform", "time"])
ds = ds.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, "index"))
ds

# %%
ds.sel(time=slice("2018-01-01", "2018-02-01"))

# %% [markdown]
# ### Timeseries plot, separated by model

# %%
from fluxy.plots.mf_timeseries import plot_timeseries

###################################
### edit variables in this block
# Variables and respective uncertainties to plot
include = {"mf_observed": None, 
           "mf_posterior": "percentile_mf_posterior",
           "mf_prior": "percentile_mf_prior",
#           "mf_bc_posterior": None,
          }

# To plot the histogram of the variables in "include", set "diff_include" to None
# To plot the histogram of Obs-variable, set "diff_include" to the desired variable to be subtracted
diff_include = ["mf_posterior"]

# To choose y-axis limits set y_lim=[min_value,max_value]
y_lim = None
###################################

###################################
### options for variables to include
# mf_observed         - total observed mole fraction
# mf_prior            - prior total mole fraction
# mf_posterior        - posterior total mole fraction
# mf_bc_prior         - prior baseline
# mf_bc_posterior     - posterior baseline
# mf_bias_prior       - prior bias added to site
# mf_bias_posterior   - posterior bias added to site
# mf_outer_prior      - prior mole fractions only from outer regions
# mf_outer_posterior  - posterior mole fractions only from outer regions
# stdev_mf_observed_repeatability - observed repeatability mole fraction uncertainty
# stdev_mf_observed_variability   - observed variability mole fraction uncertainty
# stdev_mf_model                  - model mole fraction uncertainty
# stdev_mf_total                  - total mole fraction uncertainty

### options for uncertainties to plot as error bars/shaded area
# stdev_mf_observed_repeatability - observed repeatability mole fraction uncertainty
# stdev_mf_observed_variability   - observed variability mole fraction uncertainty
# stdev_mf_model                  - model mole fraction uncertainty
# stdev_mf_total                  - total mole fraction uncertainty
# percentile_mf_prior             - prior mole fraction uncertainty
# percentile_mf_posterior         - posterior mole fraction uncertainty
###################################

fig = plot_timeseries(
    ds_all_mf_sliced,
    include,
    species,
    site,
    model_colors,
    model_labels,
    config_data,
    annotate_coords,
    presentation_mode,
    plot_type="separate",
    diff_include=diff_include,
    y_lim=y_lim,
)

# %% [markdown]
# It looks like the fit to the obs doesn't really change, but the uncertainty decreases massively, except at pollution event times. 
#
# - Why is uncertainty high initially?
# - Why does it decrease? (In Gaussian model, the posterior predictive uncertainty is the uncertainty of the likelihood combined with the uncertainty of the posterior... so I guess we're seeing just the uncertainty from the likelihood?)
# - Does this have anything to do with sigma? (e.g. is it smaller in the posterior?)
#
# To check if it is due to the prior flux uncertainty, we can compare the prior predictives from runs where Helene used 2 for the stdev instead of 8.
# We could also look at the effect of constant min model error...

# %%
model_dirs


# %%
def get_conc_all(model_dict: dict) -> dict[str, xr.Dataset]:
    ds_conc_all = {k: xr.concat([xr.open_dataset(sf6_path / v / f"SF6_EUROPE_PARIS_conc_20{yr}-01-01.nc").rename(sitenames="site").swap_dims(nsite="site") for yr in ["15", "16", "17", "18", "19"]], dim="time", join="outer") for k, v in model_dict.items()}
    ds_conc_all = {k: v.swap_dims(site="nsite").rename(site="sitenames") for k, v in ds_conc_all.items()}
    ds_conc_all = {k : edit_vars_and_attributes(v, k, "yearly", "concentration", config_data.get("regions_info", {}), config_data.get("site_info", {}), species=species)
          for k, v in ds_conc_all.items()}
    return ds_conc_all


# %%
model_dict_std_2 = {"pefo_true_std_2": 'RHIME_NAME_EUROPE_FLAT_PARISNID2026_wi_bias_prior_std_2_sf6_yearly',
                    "pefo_false_std_2": 'RHIME_NAME_EUROPE_FLAT_PARISNID2026_wi_bias_prior_std_2_poll_events_from_obs_FALSE_sf6_yearly',
                   }

models_std_2 = list(model_dict_std_2.keys())
model_colors_std_2 = set_model_colors(models_std_2)
model_labels_std_2 = {"pefo_true_std_2": "PEFO True, stdev 2", "pefo_false_std_2": "PEFO False, stdev 2"}

# %%
ds_all_mf_std_2 = get_conc_all(model_dict_std_2)

# %%
ds_all_mf_std_2_sliced = slice_mf(
    ds_all_mf_std_2.copy(),
    start_date,
    end_date,
    site,
    baseline_site=baseline_site,
    baseline_filename=baseline_filename,
    data_dir=data_dir,
    mf_units_print=mf_units_print,
)

# %%
fig = plot_timeseries(
    ds_all_mf_std_2_sliced,
    include,
    species,
    site,
    model_colors_std_2,
    model_labels_std_2,
    config_data,
    annotate_coords,
    presentation_mode,
    plot_type="separate",
    diff_include=diff_include,
    y_lim=y_lim,
)

# %% [markdown]
# These look the same as runs that used a bias and stdev 8... I'm not sure if these runs used a bias.

# %% [markdown]
# # Info on SF6 production
#
# From searches on production and industrial use around 2010:

# %%
sf6_info = """
"category","company_name","location","latitude","longitude","description"
"SF6 production and major recycling","Solvay","Bad Wimpfen, Germany","49.23","9.17","Major site for SF6 manufacturing and recycling."
"Electrical equipment manufacturing (using SF6)","ABB","Zürich, Switzerland","47.37","8.54","Global headquarters; likely had SF6-related activities."
"Electrical equipment manufacturing (using SF6)","ABB","Klagenfurt, Austria","46.62","14.31","Plant location mentioned in 2017 documents, likely involved in SF6 handling in 2010."
"Electrical equipment manufacturing (using SF6)","ABB","Ostrava, Czech Republic","49.83","18.29","Operations center; likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Siemens","Anderlecht, Belgium","50.82","4.31","Subsidiary location, potentially involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Siemens","Germany","51.17","10.45","Multiple facilities; major hub for electrical equipment manufacturing and SF6 handling."
"Electrical equipment manufacturing (using SF6)","Schneider Electric","Rueil-Malmaison, France","48.88","2.18","Head office; likely involved in SF6-related manufacturing in 2010."
"Electrical equipment manufacturing (using SF6)","Schneider Electric","France","46.60","1.88","Multiple production sites in 2010; likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Alstom Grid (acquired by GE)","Boulogne-Billancourt, France","48.83","2.24","Headquarters in 2010; likely involved in SF6-related manufacturing."
"Electrical equipment manufacturing (using SF6)","Alstom Grid (acquired by GE)","Scandale, Italy","39.12","16.96","Operations, likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Ormazabal","Igorre, Spain","43.16","357.22","Facility location; likely manufactured SF6-insulated switchgear."
"Electrical equipment manufacturing (using SF6)","Ormazabal","Merseyside, England","53.42","357.00","Facility location; likely manufactured SF6-insulated switchgear."
"Gas suppliers and handlers","Air Liquide","Sines, Portugal","37.96","351.14","Liquified air gas unit producing gases, potentially supplying SF6."
"Gas suppliers and handlers","Air Liquide","Rozenburg, Netherlands","51.92","4.25","Facilities potentially involved in gas supply, including SF6."
"Gas suppliers and handlers","Air Liquide","Geleen, Netherlands","50.97","5.83","Facilities potentially involved in gas supply, including SF6."
"Gas suppliers and handlers","Air Liquide","Hésingue, France","47.58","7.52","Cryostar facility; likely involved in gas handling."
"Gas suppliers and handlers","Air Liquide","Germany","51.17","10.45","Numerous chemical park sites potentially supplying SF6."
"Gas suppliers and handlers","Linde Group","Germany","51.17","10.45","Numerous chemical park sites potentially supplying SF6."
"Gas suppliers and handlers","Linde Group","Pullach im Isartal, Germany","48.06","11.51","Engineering facility; potentially involved in gas handling."
"Gas suppliers and handlers","Linde Group","Dresden, Germany","51.05","13.74","Engineering facility; potentially involved in gas handling."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Crolles, France","45.28","5.88","R&D and manufacturing facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Rousset, France","43.48","5.62","Wafer fab and R&D facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Kirkop, Malta","35.84","14.49","Assembly and testing facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Agrate, Italy","45.57","9.35","Fab lines and R&D center; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","Infineon Technologies","Villach, Austria","46.61","13.85","High power semiconductor segment facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","Infineon Technologies","Dresden, Germany","51.05","13.74","Facility location; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","Infineon Technologies","Cegléd, Hungary","47.17","19.80","Fabrication unit; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Eindhoven, Netherlands","51.42","5.46","Headquarters; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Hamburg, Germany","53.55","9.99","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Nijmegen, Netherlands","51.84","5.86","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Glasgow, Scotland","55.86","4.25","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Southampton, England","50.90","1.40","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Leuven, Belgium","50.88","4.70","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Caen, France","49.18","0.37","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Sophia Antipolis, France","43.61","7.07","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Toulouse, France","43.60","1.44","Facility location; potentially involved in semiconductor manufacturing using SF6."
"Other uses","Magnesium die-casting","Various locations","N/A","N/A","SF6 used as a protective atmosphere in this process."
"Other uses","Soundproof windows","Various locations","N/A","N/A","SF6 used as an insulating gas between glass panes."
"Other uses","Tracer gas","Various locations","N/A","N/A","SF6 used as a tracer gas for leak detection in various industries."
"Waste management","Celtic Recycling","Newport, Wales","51.58","356.99","Specialist facility for SF6 gas recovery and disposal."
"""

# %%
import io
import pandas as pd


info_df = pd.read_csv(io.StringIO(sf6_info), sep=",")
info_df

# %%
sf6_info_2015="""
category,company_name,location,latitude,longitude,description
"SF6 production and major recycling","Solvay","Bad Wimpfen, Germany",49.23,9.17,"Major site for SF6 manufacturing and recycling."
"Electrical equipment manufacturing (using SF6)","ABB","Zürich, Switzerland",47.37,8.54,"Global headquarters; continued use of SF6 in electrical equipment manufacturing."
"Electrical equipment manufacturing (using SF6)","ABB","Klagenfurt, Austria",46.62,14.31,"Plant location, likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","ABB","Ostrava, Czech Republic",49.83,18.29,"Operations center, likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Siemens","Anderlecht, Belgium",50.82,4.31,"Subsidiary location, potentially involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Siemens","Germany",51.17,10.45,"Multiple facilities; major hub for electrical equipment manufacturing and SF6 handling."
"Electrical equipment manufacturing (using SF6)","Schneider Electric","Rueil-Malmaison, France",48.88,2.18,"Head office, likely involved in SF6-related manufacturing."
"Electrical equipment manufacturing (using SF6)","Schneider Electric","France",46.60,1.88,"Multiple production sites; likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Alstom Grid (acquired by GE in 2015)","Boulogne-Billancourt, France",48.83,2.24,"Headquarters; likely involved in SF6-related manufacturing at the time of GE acquisition."
"Electrical equipment manufacturing (using SF6)","Alstom Grid (acquired by GE)","Scandale, Italy",39.12,16.96,"Operations, likely involved in SF6 handling."
"Electrical equipment manufacturing (using SF6)","Ormazabal","Igorre, Spain",43.16,-2.78,"Facility location, likely manufactured SF6-insulated switchgear."
"Electrical equipment manufacturing (using SF6)","Ormazabal","Merseyside, England",53.42,-3.00,"Facility location, likely manufactured SF6-insulated switchgear."
"Gas suppliers and handlers","Air Liquide","Sines, Portugal",37.96,-8.86,"Liquified air gas unit, potentially supplying SF6."
"Gas suppliers and handlers","Air Liquide","Rozenburg, Netherlands",51.92,4.25,"Facilities potentially involved in gas supply, including SF6."
"Gas suppliers and handlers","Air Liquide","Geleen, Netherlands",50.97,5.83,"Facilities potentially involved in gas supply, including SF6."
"Gas suppliers and handlers","Linde Group","Germany",51.17,10.45,"Numerous chemical park sites potentially supplying SF6."
"Gas suppliers and handlers","Linde Group","Pullach im Isartal, Germany",48.06,11.51,"Engineering facility; potentially involved in gas handling."
"Gas suppliers and handlers","Linde Group","Dresden, Germany",51.05,13.74,"Engineering facility; potentially involved in gas handling."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Crolles, France",45.28,5.88,"R&D and manufacturing facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Rousset, France",43.48,5.62,"Wafer fab and R&D facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Kirkop, Malta",35.84,14.49,"Assembly and testing facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","STMicroelectronics","Agrate, Italy",45.57,9.35,"Fab lines and R&D center; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","Infineon Technologies","Villach, Austria",46.61,13.85,"High power semiconductor segment facility; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","Infineon Technologies","Dresden, Germany",51.05,13.74,"Facility location; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","Infineon Technologies","Cegléd, Hungary",47.17,19.80,"Fabrication unit; used SF6 as an etching gas."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Eindhoven, Netherlands",51.42,5.46,"Headquarters; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Hamburg, Germany",53.55,9.99,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Nijmegen, Netherlands",51.84,5.86,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Glasgow, Scotland",55.86,-4.25,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Southampton, England",50.90,-1.40,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Leuven, Belgium",50.88,4.70,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Caen, France",49.18,-0.37,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Sophia Antipolis, France",43.61,7.07,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Semiconductor manufacturing (using SF6)","NXP Semiconductors","Toulouse, France",43.60,1.44,"Facility location; potentially involved in semiconductor manufacturing using SF6."
"Waste management","Celtic Recycling","Newport, Wales",51.58,-3.01,"Specialist facility for SF6 gas recovery and disposal."
"""

# %%
info_df_2015 = pd.read_csv(io.StringIO(sf6_info_2015), sep=",")
info_df_2015

# %% [markdown]
# Link to google AI search for this info: https://share.google/aimode/2g8lJUJgGgiK5XheO

# %%
info_df.to_csv("sf6_model_testing_data/sf6_info_2010.csv")
info_df_2015.to_csv("sf6_model_testing_data/sf6_info_2015.csv")


# %%
