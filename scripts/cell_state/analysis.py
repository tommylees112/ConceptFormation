from typing import List, Tuple, Optional, Any, Dict, Union, DefaultDict
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from sklearn.linear_model import SGDRegressor

import sys

sys.path.append("/home/tommy/ConceptFormation/neuralhydrology")
from neuralhydrology.utils.config import Config
from scripts.cell_state.cell_state_dataset import CellStateDataset
from scripts.cell_state.cell_state_model import LinearModel


def finite_flat(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def empirical_cdf(errors: np.ndarray, ax: Optional[Any] = None, kwargs: Dict[str, Any] = {}) -> Any:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    x = np.sort(errors)
    y = np.arange(len(x))/float(len(x))
    ax.plot(x, y, **kwargs)
    return ax


def histogram_plot(
    arr: np.ndarray, ax=None, hist_kwargs={}, zero_benchmark: bool = False
):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    ax.hist(arr, alpha=0.6, bins=50, density=True, **hist_kwargs)

    try:
        color = hist_kwargs["color"]
    except KeyError:
        color = "k"
    ax.axvline(np.median(arr), ls="--", color=color, alpha=0.8)

    if zero_benchmark:
        ax.axvline(0, ls=":", color="k", alpha=0.4, label="Mean Benchmark")

    sns.despine()
    ax.set_xlabel("Metric")
    return ax


def plot_weights(
    ws: np.ndarray,
    ax: Optional[Any] = None,
    kwargs: Dict[str, Any] = {"vmin": 0, "vmax": 0.3},
    cbar: bool = True,
):
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 2))

    if len(ws.shape) == 1:
        #  row vector
        ws = ws.reshape(1, -1)
    im = ax.pcolormesh(ws, **kwargs)
    if cbar:
        plt.colorbar(im, orientation="horizontal", pad=0.2)
        plt.tight_layout()
    return ax


def get_model_weights(model: Union[LinearModel, RegressorMixin]) -> Tuple[np.ndarray]:
    if isinstance(model, LinearModel):
        parameters = list(model.parameters())
        w = parameters[0].cpu().detach().numpy()
        b = parameters[1].cpu().detach().numpy()
    elif isinstance(model, RegressorMixin):
        w = model.coef_
        b = model.intercept_
    else:
        assert False, "Only works with Pytorch and Sklearn models"
    return w, b


def get_all_models_weights(models: List[nn.Linear]) -> Tuple[np.ndarray]:
    model_outputs = defaultdict(dict)
    for sw_ix in range(len(models)):
        w, b = get_model_weights(models[sw_ix])
        model_outputs[f"swvl{sw_ix+1}"]["w"] = w
        model_outputs[f"swvl{sw_ix+1}"]["b"] = b

    ws = np.stack([model_outputs[swl]["w"] for swl in model_outputs.keys()]).reshape(
        len(models), 64
    )
    bs = np.stack([model_outputs[swl]["b"] for swl in model_outputs.keys()])
    return ws, bs


def _correlations_each_dimension(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # [timesteps, n_dimensions]
    assert X.shape[0] == y.shape[0], "Expect equal number of timesteps"
    assert X.shape[-1] > 1, "Expect more than 1 dimension (64 state vector) in X"

    correlations = []
    pbar = tqdm(range(X.shape[-1]), desc="Calculating correlations for each dimension")
    for i in pbar:
        corr = np.corrcoef(X[:, i], y, rowvar=False)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    return correlations


def calculate_raw_correlations(
    norm_sm: xr.Dataset,
    cs_data: xr.Dataset,
    variable_str: Optional[str] = "sm",
    device: str = "cpu",
    time_dim: str = "time",
) -> np.ndarray:
    """Calculate the correlation coefficient for each feature of cs_data
    using: `np.corrcoef`.

    Args:
        norm_sm (xr.Dataset): The target soil moisture data (1D)
        cs_data (xr.Dataset): The input cell state data (64D)
        variable_str (Optional[str]): (defaults to "cell_state")

    Returns:
        np.ndarray: (4, 64) correlation coefficient for each feature (64),
         for each soil water level (4).
    """
    #  Create the datasets for each feature
    datasets = []

    #  features can be soil level or just one soil moisture estimate
    features = (
        list(norm_sm.data_vars) if len(list(norm_sm.data_vars)) > 1 else [variable_str]
    )
    for feature in features:
        # target data = SOIL MOISTURE
        target_data = norm_sm[feature]
        target_data["station_id"] = [int(sid) for sid in target_data["station_id"]]

        #  input data
        input_data = cs_data
        input_data["station_id"] = [int(sid) for sid in input_data["station_id"]]

        start_date = pd.to_datetime(cs_data[time_dim].min().values)
        end_date = pd.to_datetime(cs_data[time_dim].max().values)
        print(f"Creating CellStateDataset for feature {feature}")
        sm_dataset = CellStateDataset(
            input_data=input_data,
            target_data=target_data,
            device=device,
            start_date=start_date,
            end_date=end_date,
        )
        datasets.append(sm_dataset)

    # Calculate the correlations for each level
    all_correlations = np.zeros((len(features), 64))

    pbar = tqdm(np.arange(len(datasets)), desc="Calculating Correlation")
    for feature in pbar:
        pbar.set_postfix_str(feature)

        #  get the DATA for that feature
        all_cs_data = np.array(
            [x.detach().cpu().numpy() for (_, (x, _)) in DataLoader(datasets[feature])]
        )
        all_sm_data = np.array(
            [y.detach().cpu().numpy() for (_, (_, y)) in DataLoader(datasets[feature])]
        )
        Y = all_sm_data.reshape(-1, 1)

        correlations = []
        for cs in np.arange(all_cs_data.shape[-1]):
            X = all_cs_data[:, :, cs]
            correlations.append(np.corrcoef(X, Y, rowvar=False)[0, 1])

        #  save correlations to matrix (n_features, 64)
        correlations = np.array(correlations)
        all_correlations[feature, :] += correlations

    return all_correlations


def save_train_test_data(
    run_dir: Path,
    all_train_test: DefaultDict[str, Dict[str, np.ndarray]],
):
    pass


def save_probe_components(
    run_dir: Path,
    all_models_preds: DefaultDict[str, Dict[str, Union[xr.Dataset, SGDRegressor]]],
):
    #  save defaultdict components
    if not (run_dir / "probe").exists():
        (run_dir / "probe").mkdir(exist_ok=True)

    for var in tqdm([v for v in all_models_preds.keys()], desc=f"Saving data"):
        results = all_models_preds[var]
        if results != {}:
            save_dir = run_dir / f"probe/{var}"
            if not save_dir.exists():
                save_dir.mkdir(exist_ok=True, parents=True)

            _preds = results["preds"]  #  xr.Dataset
            _errors = results["errors"]  #  xr.Dataset
            _model = results["model"]  # SGDRegressor, nn.Module

            _preds.to_netcdf(save_dir / "preds.nc")
            _errors.to_netcdf(save_dir / "errors.nc")
            pickle.dump(_model, (save_dir / "model.pkl").open("wb"))
        else:
            print(f"No data from {var}")


def load_probe_components(
    run_dir: Path,
) -> DefaultDict[str, Dict[str, Union[xr.Dataset, SGDRegressor]]]:
    #  load defaultdict
    _all_models_preds = defaultdict(dict)
    target_vars = [
        d.name
        for d in (run_dir / "probe").iterdir()
        if (d.is_dir() & ((d / "preds.nc").exists()))
    ]
    assert (
        target_vars != []
    ), f"Target Variables are empty. Has the default dict been saved in {run_dir / 'probe'}"
    for var in target_vars:
        save_dir = run_dir / f"probe/{var}"
        _all_models_preds[var]["model"] = pickle.load(
            (save_dir / "model.pkl").open("rb")
        )
        _all_models_preds[var]["preds"] = xr.open_dataset(save_dir / "preds.nc")
        _all_models_preds[var]["errors"] = xr.open_dataset(save_dir / "errors.nc")
    return _all_models_preds


def get_hydrological_year(year: int) -> Tuple[pd.Timestamp, ...]:
    """return Oct-Oct https://nrfa.ceh.ac.uk/news-and-media/news/happy-new-water-year"""
    min_ts = pd.Timestamp(day=10, month=10, year=year - 1)
    max_ts = pd.Timestamp(day=30, month=9, year=year)
    return (min_ts, max_ts)
