from typing import Dict, Optional, List, Tuple, Any, DefaultDict
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from scipy import sparse
from torch.utils.data import Dataset
from tqdm import tqdm

import sys

sys.path.append("/home/tommy/ConceptFormation/neuralhydrology")
from scripts.read_nh_results import calculate_all_error_metrics
from scripts.cell_state.timeseries_model import _round_time_to_hour
from scripts.cell_state.timeseries_dataset import create_train_test_datasets
from scripts.cell_state.timeseries_dataset import get_time_basin_aligned_dictionary


def init_linear_model(kwargs: Dict = {}, sgd: bool = False):
    if sgd:
        _kwargs = dict(
            loss="squared_loss",  # "huber"
            penalty="elasticnet",
            alpha=0.01,
            l1_ratio=0.15,  # default
            fit_intercept=True,
            n_iter_no_change=3,
            early_stopping=True,
            verbose=1,
            random_state=100,
        )

        if kwargs != {}:
            for new_key, new_val in kwargs.items():
                _kwargs.update({new_key: new_val})

        model = SGDRegressor(**_kwargs)
    else:
        _kwargs = dict(n_jobs=-1, fit_intercept=True)
        if kwargs != {}:
            for new_key, new_val in kwargs.items():
                _kwargs.update({new_key: new_val})
        
        model = LinearRegression(**_kwargs)

    return model


def init_nonlinear_model(
    hidden_sizes: List[int], activation: str = "relu", kwargs: Dict = {}
):
    _kwargs = dict(
        alpha=0.15,
        solver="adam",
        learning_rate="invscaling",
        random_state=100,
        max_iter=30,
        early_stopping=True,
        n_iter_no_change=3,
        verbose=1,
    )
    if kwargs != {}:
        for new_key, new_val in kwargs.items():
            _kwargs.update({new_key: new_val})

    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes, activation=activation, **_kwargs
    )
    return model


def create_sparse_ohe_appended_data(data_dict: Dict[str, np.ndarray], ohe_data: pd.DataFrame, mode: str = "train"):
    _ohe = ohe_data.loc[data_dict["station_ids"].ravel()].values
    print(f"{mode.upper()} Made dataframe of one hot encodings")
    sparse_ohe = sparse.csr_matrix(_ohe)
    print(f"{mode.upper()} Made Sparse OHE data")
    sparse_x = sparse.csr_matrix(data_dict["X"])
    n_dims = sparse_x.shape[-1] + sparse_ohe.shape[-1]
    print(f"N Dimensions: {n_dims}")
    sparse_X_ohe = sparse.hstack([sparse_x, sparse_ohe])
    
    return sparse_X_ohe


def make_predictions(model, data: Dict[str, np.ndarray], ohe_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ohe_data is None:
        y_hat = model.predict(data["X"])
    else:
        batch_size = 100
        batches = create_batch_object(data, batch_size=batch_size)
        all_yhats = []
        for batch in tqdm(batches, desc="Running Forward Predictions"):
            _ohe = ohe_data.loc[batch["station_ids"].ravel()].values
            # replace the values not found in ohe_data with the missing column
            _ohe[~np.isin(batch["station_ids"].ravel(), ohe_data.index)] = 999999

            X_ohe = np.hstack([batch["X"], _ohe])
            yhat = model.predict(X_ohe)
            all_yhats.append(yhat)
        y_hat = np.concatenate(all_yhats)
    
    preds = (
        pd.DataFrame(
            {
                "station_id": data["station_ids"].ravel(),
                "time": data["times"].astype("datetime64[ns]").ravel(),
                "obs": data["y"].ravel(),
                "sim": y_hat.ravel(),
            }
        )
        .set_index(["station_id", "time"])
        .to_xarray()
    )

    return preds


def create_analysis_dataset(data: Dict[str, np.ndarray]) -> xr.Dataset:
    _x_df = pd.DataFrame(
        {f"dim{i}": data["X"][:, i] for i in range(data["X"].shape[-1])}
    )
    _df = pd.DataFrame(
        {
            "station_id": data["station_ids"].ravel(),
            "time": data["times"].astype("datetime64[ns]").ravel(),
            "y": data["y"].ravel(),
        }
    )
    analysis_df = _df.join(_x_df).set_index(["time", "station_id"])
    analysis_ds = analysis_df.to_xarray()
    analysis_ds["time"] = _round_time_to_hour(analysis_ds["time"].values)

    return analysis_ds


def evaluate(
    model,
    data: Dict[str, np.ndarray],
    basin_coord: str = "station_id",
    time_coord: str = "time",
    obs_var: str = "obs",
    sim_var: str = "sim",
    metrics: List[str] = ["NSE", "Pearson-r", "RMSE", "Alpha-NSE", "Beta-NSE"],
    ohe_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, xr.Dataset]:
    preds = make_predictions(model, data, ohe_data=ohe_data)

    errors = calculate_all_error_metrics(
        preds,
        basin_coord=basin_coord,
        time_coord=time_coord,
        obs_var=obs_var,
        sim_var=sim_var,
        metrics=metrics,
    )

    return preds, errors


def create_batch_object(data: Dict[str, np.ndarray], batch_size: int = 5) -> List[Dict[str, np.ndarray]]:
    batches = []
    batch_iterator = batch_arrays_from_dict(data, batch_size=batch_size)
    for data in batch_iterator:
        batches.append(data)
        
    return batches


def batch_arrays_from_dict(data: Dict[str, np.ndarray], batch_size: int = 5):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    keys = list(data.keys())
    try:
        n_samples = len(data[keys[0]])
    except TypeError:
        n_samples = data[keys[0]].shape[0]
    for i in range(0, n_samples, batch_size):

        yield {key: data[key][i: i+batch_size] for key in keys}


def fit_and_predict(
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    random_seed: int = 100,
    linear: bool = True,
    hidden_sizes: Optional[List[int]] = [10, 10],
    ohe_data: Optional[pd.DataFrame] = None,
    n_epochs: int = 5,  # only used if appending ohe data
    sgd: bool = True,
) -> Tuple[Any, xr.Dataset, xr.Dataset]:

    if not linear:
        assert hidden_sizes is not None, "Must pass hidden sizes if `linear=False`"
    np.random.seed(random_seed)

    # intiialise and fit the model
    if linear:
        model = init_linear_model(sgd=sgd)
    else:
        model = init_nonlinear_model(hidden_sizes)

    if ohe_data is None:
        model.fit(train["X"], train["y"].ravel())
    else:
        if sgd:
            model = init_linear_model({"early_stopping": False, "verbose": 0, "max_iter": 1})
            
            # batch the data to avoid memory issues
            batch_size = 100
            batches = create_batch_object(train, batch_size=batch_size)
            for epoch in range(n_epochs):
                for batch in tqdm(batches, desc=f"Epoch {epoch}"):
                    # append the ohe data to each batch
                    _ohe = ohe_data.loc[batch["station_ids"].ravel()].values
                    # replace the values not found in ohe_data with the missing column
                    _ohe[~np.isin(batch["station_ids"].ravel(), ohe_data.index)] = 999999

                    X_ohe = np.hstack([batch["X"], _ohe])
                    model.partial_fit(X_ohe, batch["y"].ravel())
        else:
            print("Starting OHE with standard linear regression")
            # ohe on data with no regularisation terms 
            # have to create sparse matrices
            sparse_X_ohe = create_sparse_ohe_appended_data(train, ohe_data, mode="Train")
            train["X"] = sparse_X_ohe
            
            # standard linear regression
            model = init_linear_model(sgd=False)
            print("Model training")
            model.fit(train["X"], train["y"].ravel())
            print("Model fit on sparse OHE X data")

            sparse_test_X_ohe = create_sparse_ohe_appended_data(test, ohe_data, mode="Test")
            test["X"] = sparse_test_X_ohe

    #  make predictions from the fitted model
    if not sgd:
        preds, errors = evaluate(model, test)
    else:
        preds, errors = evaluate(model, test, ohe_data=ohe_data)
    print("\n")

    return model, preds, errors


if __name__ == "__main__":
    data_dir = Path("/datadrive/data")
    #  initialise the train-test split
    train_start_date = pd.to_datetime("1998-01-01")
    train_end_date = pd.to_datetime("2006-09-30")
    test_start_date = pd.to_datetime("2006-10-01")
    test_end_date = pd.to_datetime("2009-10-01")

    input_variables = [f"dim{i}" for i in np.arange(64)]
    target_var = "sm"
    subset_pixels = None

    #  get target data `era5_sm`
    # era5_sm = xr.open_dataset(data_dir / "SOIL_MOISTURE/FINAL/era5land_normalized.nc")
    esa_ds = xr.open_dataset(
        data_dir / "SOIL_MOISTURE/FINAL/esa_ds_interpolated_normalised.nc"
    )

    #  get input data `cs`
    cs = xr.open_dataset(data_dir / "SOIL_MOISTURE/FINAL/cs_normalised_64variables.nc")

    # create train-test dataloaders and Dict of X, y arrays
    train_dataset, test_dataset = create_train_test_datasets(
        target_var=target_var,
        input_variables=input_variables,
        target_ds=esa_ds,
        input_ds=cs,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        subset_pixels=subset_pixels,
        seq_length=1,
        basin_dim="station_id",
        time_dim="time",
    )

    train = get_time_basin_aligned_dictionary(train_dataset)
    test = get_time_basin_aligned_dictionary(test_dataset)

    #  initalise the model
    model = init_linear_model()
    #  fit the model
    model.fit(train["X"], train["y"].ravel())

    #  make predictions from the fitted model
    preds, errors = evaluate(model, test)
    train_preds, train_errors = evaluate(model, train)

    # create an easy to work with analysis dataset
    test_analysis = create_analysis_dataset(test)
    train_analysis = create_analysis_dataset(train)
    analysis_ds = xr.concat([train_analysis, test_analysis], dim="time")

    assert False
