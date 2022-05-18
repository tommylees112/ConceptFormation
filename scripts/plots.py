from typing import Optional, Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def adjust_yaxis(ax: Any, variable: np.ndarray) -> Any:
    xmin = ax.get_ylim()[0]
    xmax = ax.get_ylim()[1] + (1.5 * np.std(variable))
    ax.set_ylim(xmin, xmax)

    return ax


def plot_hydrograph(
    data: xr.Dataset,
    ax: Optional[Any] = None,
    discharge_var: str = "discharge",
    precip_var: Optional[str] = "Precipitation",
    time_var: str = "time",
    basin_dim: str = "basin",
    discharge_kwargs: Dict = {},
    rainfall_kwargs: Dict = {},
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    ## Plot the discharge variable
    discharge = data[discharge_var]
    ax.plot(data[time_var], discharge.values.flatten(), label=discharge_var, **discharge_kwargs)
    ax.set_ylabel(discharge_var)

    #  adjust y axis
    ax = adjust_yaxis(ax, discharge)

    ## Plot the rainfall inverted
    if precip_var is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel(precip_var)
        ax2.invert_yaxis()

        precip = data[precip_var]
        ax2.bar(data["time"], precip, alpha=0.4, **rainfall_kwargs)

        #  adjust y axis
        ax2 = adjust_yaxis(ax2, precip)

    ax.set_title(f"Hydrograph StationID: {int(data[basin_dim].values)}")
    return ax


def scatter_plot(obs: np.ndarray, sim: np.ndarray, ax = None, scatter_kwargs: Dict = {"marker": "x", "color": "C0", "alpha": 0.3}):
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))
    
    
    ax.scatter(obs, sim, **scatter_kwargs)
    # plot 1_to_1 line (1:1 line)
    lim = (min([np.nanmin(obs), np.nanmin(sim)]), max([np.nanmax(obs), np.nanmax(sim)]))
    ax.plot(lim, lim, ls="--", color="k")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("obs")
    ax.set_ylabel("sim")


def empirical_cdf(errors: np.ndarray, ax: Any, kwargs: Dict[str, Any] = {}):
    x = np.sort(errors)
    y = np.arange(len(x))/float(len(x))
    ax.plot(x, y, **kwargs)


def plot_context(static: xr.Dataset, variables: List[str], sids: Optional[List[int]] = None, axvline_kwargs: Dict[str, Any] = {}) -> Any:
    n = len(variables)
    nrows = n // 2 + (n % 2)
    f, axs = plt.subplots(nrows, 2, figsize=(12, 2*nrows), tight_layout=True)

    for ix, var_ in enumerate(variables):
        ax = axs[np.unravel_index(ix, (nrows, 2))]
        ax.hist(static[var_], color="grey", alpha=0.6, bins=30, density=True)
        ax.set_title(var_)

        if sids is not None:
            for sid in sids:
                ax.axvline(static[var_].sel(station_id=sid), **axvline_kwargs)
    return axs


def plot_precip_as_bar(precip: xr.DataArray, ax: Any, despine: bool = True, kwargs: Dict = {}):
    ax.bar(precip["time"], precip, alpha=0.4, **kwargs)
    ax.set_ylim([0, precip.max() + 3 * precip.std()])
    if despine:
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel("")
    ax.invert_yaxis()