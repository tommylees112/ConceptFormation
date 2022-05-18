from typing import List, Dict, DefaultDict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict
import xarray as xr 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys

sys.path.append("/home/tommy/ConceptFormation/neuralhydrology")
from scripts.cell_state.timeseries_dataset import TimeSeriesDataset, get_time_basin_aligned_dictionary
from scripts.cell_state.sklearn_models import fit_and_predict
from scripts.cell_state.normalize import normalize_cstate
from scripts.cell_state.cell_state_dataset import dataset_dimensions_to_variable
from neuralhydrology.utils.config import Config
from scripts.cell_state.extract_cell_state import get_cell_states



def _create_timeseries_dataset(
    train_cs: xr.Dataset, 
    test_cs: xr.Dataset,
    train_target_ds: xr.Dataset,
    test_target_ds: xr.Dataset,
    input_variables: List[str],
    seq_length: int = 1,
    basin_dim: str = "station_id",
    time_dim: str = "time",
    calculate_test_set: bool = True,
) -> DefaultDict[str, Dict[str, Dict[str, TimeSeriesDataset]]]:
    all_train_test_timeseries_datasets = defaultdict(dict)
    for target_var in [v for v in train_target_ds.data_vars]:
        train_dataset = TimeSeriesDataset(
            input_data=train_cs,
            target_data=train_target_ds,
            target_variable=target_var,
            input_variables=input_variables,
            seq_length=seq_length,
            basin_dim=basin_dim,
            time_dim=time_dim,
            desc="Creating Train Samples",
        )
        assert train_dataset.num_samples != 0 

        if calculate_test_set:
            test_dataset = TimeSeriesDataset(
                input_data=test_cs,
                target_data=test_target_ds,
                target_variable=target_var,
                input_variables=input_variables,
                seq_length=seq_length,
                basin_dim=basin_dim,
                time_dim=time_dim,
                desc="Creating Test Samples",
            )
            assert test_dataset.num_samples != 0 
        else:
            test_dataset = None

        all_train_test_timeseries_datasets[target_var]["train"] = train_dataset
        all_train_test_timeseries_datasets[target_var]["test"] = test_dataset

    return all_train_test_timeseries_datasets


def _create_train_test_numpy_arrays(all_train_test_timeseries_datasets: DefaultDict[str, Dict[str, Dict[str, TimeSeriesDataset]]]) -> DefaultDict[str, Dict[str, Dict[str, np.ndarray]]]:
    all_train_test = defaultdict(dict)
    for target_var in [v for v in all_train_test_timeseries_datasets.keys()]:
        train = get_time_basin_aligned_dictionary(all_train_test_timeseries_datasets[target_var]["train"])
        
        if all_train_test_timeseries_datasets[target_var]["test"] is not None:
            test = get_time_basin_aligned_dictionary(all_train_test_timeseries_datasets[target_var]["test"])
        else:
            test = None

        all_train_test[target_var]["train"] = train
        all_train_test[target_var]["test"] = test
    
    return all_train_test


def create_train_test_default_dict_for_all_target_vars(
    train_cs: xr.Dataset, 
    test_cs: xr.Dataset,
    train_target_ds: xr.Dataset,
    test_target_ds: xr.Dataset,
    input_variables: List[str],
    seq_length: int = 1,
    basin_dim: str = "station_id",
    time_dim: str = "time",
    calculate_test_set: bool = True,
) -> DefaultDict[str, Dict[str, Dict[str, np.ndarray]]]:
    all_train_test = defaultdict(dict)
    assert all(np.isin(train_target_ds.data_vars, test_target_ds.data_vars)), "" + \
        "Expect same target variables." + \
        f"\nTrain: {list(train_target_ds.data_vars)} -- Test: {list(test_target_ds.data_vars)}"
    
    for target_var in [v for v in train_target_ds.data_vars]:
        print(f"** STARTING {target_var} **")

        all_train_test_timeseries_datasets = _create_timeseries_dataset(
            train_cs=train_cs,
            test_cs=test_cs,
            train_target_ds=train_target_ds[[target_var]],
            test_target_ds=test_target_ds[[target_var]],
            input_variables=input_variables,
            seq_length=seq_length,
            basin_dim=basin_dim,
            time_dim=time_dim,
            calculate_test_set=calculate_test_set,
        )

        all_train_test[target_var] = _create_train_test_numpy_arrays(all_train_test_timeseries_datasets)[target_var]
    return all_train_test


def get_ohe_of_station_ids_to_X_data(
    all_train_test: DefaultDict[str, Dict[str, Dict[str, np.ndarray]]],
    station_ids: Optional[List[int]] = None,
    include_missing: bool = True,
) -> DefaultDict[str, Dict[str, Dict[str, np.ndarray]]]:
    target_vars = list(all_train_test.keys())
    
    ohe = OneHotEncoder()
    if station_ids is None:
        sids = np.unique(all_train_test[target_vars[0]]["train"]["station_ids"])
    else: 
        sids = station_ids

    ohe_sids = ohe.fit_transform(sids.reshape(-1, 1)).toarray()
    ohe_data = pd.DataFrame(ohe_sids.T, index=pd.Series(ohe.categories_[0], name="station_id"))

    # add a missing data column
    if include_missing: 
        ohe_data[999999] = np.zeros(ohe_data.shape[0])
        _extra = pd.Series(np.zeros(ohe_data.shape[1]), name=999999).rename({669: 999999})
        _extra[999999] = 1.0
        ohe_data = ohe_data.append(_extra)
        ohe_data = ohe_data.dropna(axis=1)

    return ohe_data


def train_and_evaluate_models(
    all_train_test: DefaultDict[str, Dict[str, Dict[str, np.ndarray]]], 
    random_seed: int = 100,             
    hidden_sizes: Optional[List[int]] = None,
    evaluate_set: str = "test",
    include_station_ohe: bool = False,
    station_ids: List[int] = None,
    sgd: bool = True,
) -> DefaultDict[str, Dict[str, Any]]:
    assert evaluate_set in ["test", "train"], f'Require evaluate_set to be one of ["test", "train"], currently: {evaluate_set}'
    # initalise the model 
    all_models_preds = defaultdict(dict)
    target_vars = list(all_train_test.keys())
    linear: bool = True if hidden_sizes is None else False

    if include_station_ohe:
        ohe_data = get_ohe_of_station_ids_to_X_data(all_train_test, station_ids=station_ids)
        print("Created OneHotEncoding of Station ID variables")
    else:
        ohe_data = None

    for target_var in target_vars:
        print(f"** {target_var} {'linear' if linear else 'non-linear'} model **")
        model, preds, errors = fit_and_predict(
            all_train_test[target_var]["train"], 
            all_train_test[target_var][evaluate_set], 
            random_seed=random_seed, 
            linear=linear, 
            hidden_sizes=hidden_sizes,
            ohe_data=ohe_data,
            sgd=sgd,
        )
        all_models_preds[target_var]["model"] = model
        all_models_preds[target_var]["preds"] = preds
        all_models_preds[target_var]["errors"] = errors
    
    return all_models_preds


def normalize_and_convert_dimension_to_variable_for_cell_state_data(
    cn: xr.Dataset, out_dir: Path, per_basin: bool, train_test: str,
    variable_str: str = "c_n", time_dim: str = "time", dimension_dim: str = "dimension",
    reload: bool = False, mean_: Optional[xr.Dataset] = None, std_: Optional[xr.Dataset] = None
) -> Tuple[xr.Dataset, Tuple[xr.Dataset, xr.Dataset]]:
    assert all(np.isin([time_dim, dimension_dim], list(cn.dims.keys()))), f"Expect these dimensions: {list(cn.dims.keys())}. Got: {[time_dim, dimension_dim]}"
    assert train_test in ["test", "train"], f'Require train_test to be one of ["test", "train"], currently: {train_test}'
    if train_test == "test":
        assert all(
            [is_none is not None for is_none in [mean_, std_]]
        ), f"Found that mean_ ({mean_ is None}) or std_ ({std_ is None}) is None when calculating train errors"

    fname_base = "per_basin" if per_basin else "global"
    if ((out_dir / f"{fname_base}_{train_test}_cs.nc").exists()) and (not reload):
        cs = xr.open_dataset(out_dir / f"{fname_base}_{train_test}_cs.nc")
    else:
        print(f"Calculating Normalisation for `{train_test}` data: {fname_base}")
        if train_test == "train":
            norm_cs_data, (mean_, std_) = normalize_cstate(cn, variable_str=variable_str, per_basin=per_basin, time_dim=time_dim)
        else:
            norm_cs_data, (mean_, std_) = normalize_cstate(cn, variable_str=variable_str, per_basin=per_basin, time_dim=time_dim, mean_=mean_, std_=std_)
        cs = dataset_dimensions_to_variable(
            ds=norm_cs_data.to_dataset() if isinstance(norm_cs_data, xr.DataArray) else norm_cs_data,
            variable=variable_str,
            dimension_to_convert_to_variable_dim=dimension_dim,
            time_dim=time_dim
        )
        cs.to_netcdf(out_dir / f"{fname_base}_{train_test}_cs.nc")
    return cs, (mean_, std_)


def read_basin_list(txt_path: Path) -> pd.DataFrame:
    return pd.read_csv(txt_path, header=None).rename({0: "station_id"}, axis=1)


def get_train_test_cell_states(run_dir: Path, cfg: Config, reload: bool = False) -> Tuple[xr.Dataset, xr.Dataset]:
    out_dir = run_dir / f"cell_states"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    for subset in ["train", "test"]:
        if not (out_dir / f"{subset}_cell_states.nc").exists() or reload:
            cn = get_cell_states(cfg, run_dir, subset=subset)
            cn.to_netcdf(out_dir / f"{subset}_cell_states.nc")
        else:
            cn = xr.open_dataset(out_dir / f"{subset}_cell_states.nc")
        cn = cn if (not "date" in cn.coords) else cn.rename({"date": "time"})

        if subset == "train":
            train_cn = xr.open_dataset(out_dir / f"train_cell_states.nc")
        else:
            test_cn = xr.open_dataset(out_dir / f"test_cell_states.nc")

    return train_cn, test_cn


if __name__ == "__main__":
    from scripts.cell_state.normalize import normalize_2d_dataset


    # SETTINGS
    TARGET = "ERA5"  # ERA5 ESA
    PER_BASIN = False  # True = PER BASIN

    # PATHS / Experiment
    data_dir = Path("/datadrive/data")
    run_dir = data_dir / "runs/complexity_AZURE/hs_064_0306_205514"
    out_dir = run_dir / "cell_states"
    cfg = Config(run_dir / "config.yml")

    # CHECK STATIONS TO BE PROBED
    train_sids = read_basin_list(cfg.train_basin_file)
    test_sids = read_basin_list(cfg.test_basin_file)
    out_of_sample = not all(np.isin(test_sids, train_sids))
    print(f"Out of Sample: {not all(np.isin(test_sids, train_sids))}")

    # CALCULATE INPUT DATA 
    train_cn, test_cn = get_train_test_cell_states(run_dir, cfg, reload=False)

    # normalize input data
    train_cs = normalize_and_convert_dimension_to_variable_for_cell_state_data(
        cn=train_cn,
        out_dir=out_dir, 
        per_basin=PER_BASIN,
        train_test="train",
        time_dim="date",
        reload=True
    )

    test_cs = normalize_and_convert_dimension_to_variable_for_cell_state_data(
        cn=test_cn,
        out_dir=out_dir, 
        per_basin=PER_BASIN,
        train_test="test",
        time_dim="date",
        reload=True
    )

    # GET TARGET DATA 
    if TARGET == "ERA5":
        filepath = data_dir / "camels_basin_ERA5Land_sm.nc"
        era5_ds = xr.open_dataset(filepath)

        if not isinstance(era5_ds, xr.Dataset):
            era5_ds = era5_ds.to_dataset()

        for var in era5_ds.data_vars:
            era5_ds[var] = normalize_2d_dataset(era5_ds, variable_str=var, per_basin=PER_BASIN)

        era5_ds["station_id"] = era5_ds["station_id"].astype(int)

        # NOT for snow depth ..?
        era5_ds = era5_ds.drop("sd")
        target_ds = era5_ds

    elif TARGET == "ESA":
        filepath = data_dir / "SOIL_MOISTURE/interp_full_timeseries_esa_cci_sm.nc"
        esa_ds = xr.open_dataset(filepath).drop("spatial_ref")
        if not isinstance(esa_ds, xr.Dataset):
            esa_ds = esa_ds.to_dataset()

        for var in esa_ds.data_vars:
            esa_ds[var] = normalize_2d_dataset(esa_ds, variable_str=var, per_basin=PER_BASIN)

        esa_ds["station_id"] = esa_ds["station_id"].astype(int)
        target_ds = esa_ds
    else:
        assert False
        