import os
import getpass
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.common import is_chunked_array
from sparse import SparseArray

from openghg.retrieve import search
from openghg.retrieve import get_footprint, get_flux, get_obs_surface, search
from openghg.util import find_domain

from openghg_inversions.array_ops import get_xr_dummies, sparse_xr_dot
from openghg_inversions.basis.algorithms._weighted import load_landsea_indices
from openghg_inversions.basis.algorithms import quadtree_algorithm, weighted_algorithm
from openghg_inversions.basis._functions import (
    _flux_fp_from_fp_all,
    _mean_fp_times_mean_flux,
)
from openghg_inversions.config.paths import Paths
from openghg_inversions.utils import read_netcdfs


openghginv_path = Paths.openghginv


def masked_basis(
    da,
    mask,
    basis_fn,
    masked_options: dict | None = None,
    unmasked_options: dict | None = None,
    masked_first: bool = True,
    **kwargs,
):
    masked, unmasked = split_by_mask(da, mask)
    masked_options = masked_options or {}
    masked_options.update(kwargs)
    unmasked_options = unmasked_options or {}
    unmasked_options.update(kwargs)
    masked_fn = partial(basis_fn, **masked_options)
    unmasked_fn = partial(basis_fn, **unmasked_options)
    bf1 = xr.apply_ufunc(masked_fn, masked.compute())
    bf2 = xr.apply_ufunc(unmasked_fn, unmasked.compute())

    # remove any gaps after masking by using "unique inverse"
    # array; the region zero corresponds to the part where the mask
    # is zero, since the basis algorithms start with region 1
    def uinv(bf, mask):
        _, inv = np.unique(bf * mask, return_inverse=True)
        inv = inv.reshape(mask.shape)
        return inv

    bf1 = xr.apply_ufunc(uinv, bf1, mask)
    bf2 = xr.apply_ufunc(uinv, bf2, 1 - mask)
    # find value to shift by so that region numbers do not
    # overlap
    if masked_first:
        shift = bf1.max() * (1 - mask)
    else:
        shift = bf2.max() * mask
    return bf1 + bf2 + shift


def apply_mask(da, mask):
    da, mask = xr.align(da, mask, join="override")
    return da.where(mask, drop=True)


def split_by_mask(da, mask):
    da, mask = xr.align(da, mask, join="override")
    return da.where(mask, 0, drop=False), da.where(1 - mask, 0, drop=False)


class BasisFunctions:
    def __init__(self, basis_flat, flux, chunks: dict | None = None):
        self.basis_flat = (
            basis_flat.isel(time=0) if "time" in basis_flat.dims else basis_flat
        )
        if "time" in flux.dims and flux.sizes["time"] == 1:
            self.flux = flux.squeeze("time", drop=True)
        else:
            self.flux = flux

        self.basis_matrix = get_xr_dummies(basis_flat, cat_dim="region")

        if chunks is not None:
            self.basis_matrix = self.basis_matrix.chunk(**chunks)
        else:
            self.basis_matrix = self.basis_matrix.chunk()

        self.labels = np.unique(basis_flat)
        self.labels_shuffled = np.unique(basis_flat)
        np.random.shuffle(self.labels_shuffled)

        self.interpolation_matrix = self.basis_matrix * self.flux

        self.projection_weightings = xr.dot(
            self.basis_matrix, self.flux**2, dim=["lat", "lon"]
        )

        # if "time" in self.projection_weightings.dims:
        #     self.projection_weightings = self.projection_weightings.squeeze(
        #         "time", drop=True
        #     )

        # make factors to rescale prior uncertainty
        # normalise by mean of flux to avoid floating point issues
        flux_scaling = 1 / self.flux.mean()
        self.uncertainty_rescaling = (
            xr.dot(
                (flux_scaling * self.flux) ** 4, self.basis_matrix, dim=["lat", "lon"]
            )
            / (flux_scaling**2 * self.projection_weightings) ** 2
        )

        # if "time" in self.uncertainty_rescaling.dims:
        #     self.uncertainty_rescaling = self.uncertainty_rescaling.squeeze(
        #         "time", drop=True
        #     )

        # make aggregation error factor; the aggregation error is then
        # prior_sigma * np.sqrt((fp_x_flux)**2 @ agg_err_factor)
        self.agg_err_factor = (
            1
            - 2
            * (flux_scaling * self.flux) ** 2
            / self.interpolate(flux_scaling**2 * self.projection_weightings)
            + self.interpolate(self.uncertainty_rescaling)
        )  #.squeeze("time", drop=True)

    def interpolate(self, data, flux: bool = False):
        """Map from regions to lat/lon."""
        if flux:
            return xr.dot(self.interpolation_matrix, data, dim="region")
        return xr.dot(self.basis_matrix, data, dim="region")

    def project(self, data, flux: bool = False, normalise: bool = False):
        if flux:
            return (
                xr.dot(data, self.interpolation_matrix, dim=["lat", "lon"])
                / self.projection_weightings
            )
        if normalise:
            return xr.dot(
                data, self.basis_matrix, dim=["lat", "lon"]
            ) / self.basis_matrix.sum(["lat", "lon"])
        return xr.dot(data, self.basis_matrix, dim=["lat", "lon"])

    def sensitivities(self, fp):
        """Pre-multiply by interpolation matrix."""
        if "time" in self.basis_matrix.dims and self.basis_matrix.sizes["time"] < 2:
            interp = self.basis_matrix.squeeze("time", drop=True)
        else:
            interp = self.basis_matrix
        interp = interp.reindex_like(fp, method="ffill")
        interp = interp.transpose("lat", "lon", ...)
        return xr.dot(fp, interp, dim=["lat", "lon"]).as_numpy()

    def plot(self, shuffle=False, **kwargs):
        if not shuffle:
            return self.basis_flat.plot(**kwargs)
        else:
            bf_shuf = self.basis_flat.copy()
            bf_shuf.values = self.labels_shuffled[
                self.basis_flat.values.astype(int) - 1
            ]
            return bf_shuf.plot(**kwargs)


def test_setup():
    res = search(store="inversions_tests")
    lsi = load_landsea_indices("europe")
    lat, lon = find_domain("europe")
    lsda = xr.DataArray(lsi, dims=["lat", "lon"], coords=[lat, lon])

    fp = get_footprint(store="inversions_tests", site="tac", domain="europe")
    flux = get_flux(
        store="inversions_tests",
        species="ch4",
        domain="europe",
        source="total-ukghg-edgar7",
    )
    fpflux = _mean_fp_times_mean_flux(flux.data.flux, [fp.data.fp])

    bf_da = masked_basis(
        fpflux,
        lsda,
        quadtree_algorithm,
        masked_options={"nbasis": 200},
        unmasked_options={"nbasis": 50},
    )

    bf = BasisFunctions(bf_da, flux.data.flux)

    return lsda, fp, flux, fpflux, bf_da, bf
