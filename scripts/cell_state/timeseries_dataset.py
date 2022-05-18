from typing import List, Tuple, Union, Dict, Optional
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
import torch
from numba import njit, prange
from torch.utils.data import DataLoader
from sklearn.utils import shuffle


def get_matching_dim(ds1, ds2, dim: str) -> Tuple[np.ndarray]:
    return (
        np.isin(ds1[dim].values, ds2[dim].values),
        np.isin(ds2[dim].values, ds1[dim].values),
    )


@njit
def validate(
    x_d: List[np.ndarray], y: List[np.ndarray], seq_length: int, is_train: bool = True
):
    n_samples = len(y)
    flag = np.ones(n_samples)

    # if any condition met then go to next iteration of loop
    for target_index in prange(n_samples):
        start_input_idx = target_index - seq_length

        # 1. not enough history (seq_length > history)
        if start_input_idx < 0:
            flag[target_index] = 0
            continue

        #  2. NaN in the dynamic inputs
        _x_d = x_d[start_input_idx:target_index]
        if np.any(np.isnan(_x_d)):
            flag[target_index] = 0
            continue

        #  3. NaN in the outputs (TODO: only for training period)
        if is_train:
            _y = y[start_input_idx:target_index]
            if np.any(np.isnan(_y)):
                flag[target_index] = 0
                continue

    return flag


class TimeSeriesDataset(Dataset):
    # TODO: how to add forecast horizon
    def __init__(
        self,
        input_data: xr.Dataset,
        target_data: xr.Dataset,
        target_variable: str,
        input_variables: List[str],
        seq_length: int = 64,
        basin_dim: str = "station_id",
        time_dim: str = "time",
        desc: str = "Creating Samples",
    ):
        self.target_variable = target_variable
        self.input_variables = input_variables
        self.seq_length = seq_length
        self.basin_dim = basin_dim
        self.time_dim = time_dim

        # matching data
        target_times, input_times = get_matching_dim(
            target_data, input_data, self.time_dim
        )
        target_sids, input_sids = get_matching_dim(
            target_data, input_data, self.basin_dim
        )

        input_data = input_data.sel(
            {self.time_dim: input_times, self.basin_dim: input_sids}
        )
        target_data = target_data.sel(
            {self.time_dim: target_times, self.basin_dim: target_sids}
        )

        self.input_data = input_data
        self.target_data = target_data

        self.create_lookup_table_of_valid_samples(
            input_data=self.input_data, target_data=self.target_data, desc=desc,
        )

    def create_lookup_table_of_valid_samples(
        self,
        input_data: Union[xr.Dataset, xr.DataArray],
        target_data: xr.DataArray,
        desc: str = "Creating Samples",
    ) -> None:
        lookup: List[Tuple[str, int]] = []
        spatial_units_without_samples: List[Union[str, int]] = []
        self.x_d: Dict[str, np.ndarray] = {}
        self.y: Dict[str, np.ndarray] = {}
        self.times: List[float] = []

        # spatial_unit = target_data[self.basin_dim].values[0]
        pbar = tqdm(target_data[self.basin_dim].values, desc=desc)

        #  iterate over each basin
        for spatial_unit in pbar:
            #  create pd.Dataframe timeseries from [xr.Dataset, xr.DataArray]
            in_df = input_data.sel({self.basin_dim: spatial_unit}).to_dataframe()
            out_df = target_data.sel({self.basin_dim: spatial_unit}).to_dataframe()

            #  create np.ndarray
            _x_d = in_df[self.input_variables].values
            _y = out_df[self.target_variable].values

            #  keep pointer to the valid samples
            flag = validate(x_d=_x_d, y=_y, seq_length=self.seq_length)
            valid_samples = np.argwhere(flag == 1)
            [lookup.append((spatial_unit, smp)) for smp in valid_samples]

            # STORE DATA if spatial_unit has at least ONE valid sample
            if valid_samples.size > 0:
                self.x_d[spatial_unit] = _x_d.astype(np.float32)
                self.y[spatial_unit] = _y.astype(np.float32)
            else:
                spatial_units_without_samples.append(spatial_unit)

            if self.times == []:
                #  store times as float32 to keep pytorch happy
                # assert False
                self.times = (
                    in_df.index.values.astype(np.float32)
                    # .astype(np.float32)
                )

        #  save lookup from INT: (spatial_unit, index) for valid samples
        self.lookup_table: Dict[int, Tuple[str, int]] = {
            i: elem for i, elem in enumerate(lookup)
        }
        self.num_samples = len(self.lookup_table)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spatial_unit, target_ix = self.lookup_table[idx]
        # X, y samples
        # [seq_length, input_features]
        x_d = self.x_d[spatial_unit][int(target_ix - self.seq_length) : int(target_ix)]
        # [seq_length, 1]
        y = np.expand_dims(
            self.y[spatial_unit][int(target_ix - self.seq_length) : int(target_ix)], -1
        )

        #  to torch.Tensor
        x_d = Tensor(x_d)
        y = Tensor(y)

        #  metadata
        time = self.times[int(target_ix - self.seq_length) : int(target_ix)]
        meta = dict(spatial_unit=spatial_unit, time=time)

        data = dict(x_d=x_d, y=y, meta=meta)

        return data


def get_train_test_dataloader(
    input_data: xr.Dataset,
    target_data: xr.Dataset,
    target_variable: str,
    input_variables: List[str],
    seq_length: int = 64,
    basin_dim: str = "station_id",
    time_dim: str = "time",
    batch_size: int = 256,
    train_start_date: pd.Timestamp = pd.to_datetime("01-01-1998"),
    train_end_date: pd.Timestamp = pd.to_datetime("12-31-2006"),
    test_start_date: pd.Timestamp = pd.to_datetime("01-01-2007"),
    test_end_date: pd.Timestamp = pd.to_datetime("01-01-2009"),
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TimeSeriesDataset(
        input_data=input_data.sel(time=slice(train_start_date, train_end_date)),
        target_data=target_data.sel(time=slice(train_start_date, train_end_date)),
        target_variable=target_variable,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Train Samples",
    )
    test_dataset = TimeSeriesDataset(
        input_data=input_data.sel(time=slice(test_start_date, test_end_date)),
        target_data=target_data.sel(time=slice(test_start_date, test_end_date)),
        target_variable=target_variable,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Test Samples",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def get_time_basin_aligned_samples(
    dataset: TimeSeriesDataset,
    batch_size: int = 256,
    num_workers: int = 0,
    final_n: Optional[int] = None,
) -> Tuple[np.ndarray]:
    # initialise dataloader
    #   TODO: batch_size=dataset.__len__()
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    #  initialise arrays
    all_x = []
    all_y = []
    all_times = []
    all_station_id = []

    #  extract arrays from dataset
    pbar = tqdm(dl, desc="Extracting Data")
    for data in pbar:
        all_x.append(data["x_d"].detach().cpu().numpy())
        _y = data["y"].detach().cpu().numpy()
        if final_n is None:
            all_y.append(_y)
        else:
            all_y.append(_y[-final_n:])

        all_times.append(data["meta"]["time"].detach().cpu().numpy())
        all_station_id.append(data["meta"]["spatial_unit"].detach().cpu().numpy())

    #  merge the arrays into the correct sizes (n_samples, :)
    print("Merging and reshaping arrays")
    #  [sample, dimension]
    X = np.vstack(all_x).squeeze()
    #  [sample, 1]
    y = np.vstack(all_y).squeeze().reshape(-1, 1)
    times = np.vstack(all_times).squeeze().reshape(-1, 1)
    station_ids = np.concatenate(all_station_id).squeeze().reshape(-1, 1)

    return (X, y, times, station_ids)


def get_time_basin_aligned_dictionary(
    dataset: TimeSeriesDataset,
    batch_size: int = 256,
    num_workers: int = 0,
    final_n: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    X, y, times, station_ids = get_time_basin_aligned_samples(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        final_n=final_n,
    )
    out = dict(X=X, y=y, times=times, station_ids=station_ids)
    return out


def get_data_samples(
    input_data: xr.Dataset,
    target_data: xr.Dataset,
    target_variable: str,
    input_variables: List[str],
    seq_length: int = 1,
    basin_dim: str = "station_id",
    time_dim: str = "time",
    batch_size: int = 256,
    num_workers: int = -1,
    start_date: pd.Timestamp = pd.to_datetime("01-01-1998"),
    end_date: pd.Timestamp = pd.to_datetime("12-31-2006"),
) -> Dict[str, np.ndarray]:
    # 1. create dataset
    dataset = TimeSeriesDataset(
        input_data=input_data.sel(time=slice(start_date, end_date)),
        target_data=target_data.sel(time=slice(start_date, end_date)),
        target_variable=target_variable,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Samples",
    )

    # 2. extract samples as numpy arrays
    X, y, times, station_ids = get_time_basin_aligned_samples(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    out = {
        "X": X,
        "y": y,
        "times": times,
        "station_ids": station_ids,
    }
    return out


def get_matching_station_ids(
    ds1: xr.Dataset, ds2: xr.Dataset, basin_dim: str = "station_id"
) -> Tuple[xr.Dataset, xr.Dataset]:
    ds1[basin_dim] = ds1[basin_dim].astype(int)
    ds2[basin_dim] = ds2[basin_dim].astype(int)

    # matching_stations
    ds1_sids, ds2_sids = get_matching_dim(ds1, ds2, basin_dim)
    #  select matching stations only
    ds1 = ds1.sel(dict(basin_dim=ds1_sids))
    ds2 = ds2.sel(dict(basin_dim=ds2_sids))

    return ds1, ds2


def shuffle_basin_dim(
    input_data: xr.Dataset, target_data: xr.Dataset, basin_dim: str = "station_id"
) -> Tuple[Dict[str, int], xr.Dataset, xr.Dataset]:
    """Create a mapping (dict) that randomly shuffles the basin dim and returns a copy of 
    the two datasets with matching dimensions which can then be shuffled using:

    input_data[basin_dim] = [mapping[sid] for sid in input_data[basin_dim].values]
        `OR`
    target_data[basin_dim] = [mapping[sid] for sid in target_data[basin_dim].values]

    Args:
        input_data (xr.Dataset): [description]
        target_data (xr.Dataset): [description]
        basin_dim (str, optional): [description]. Defaults to "station_id".

    Returns:
        Tuple[Dict[str, int], xr.Dataset, xr.Dataset]: [description]
    """
    input_copy = input_data.copy()
    target_copy = target_data.copy()

    target_copy, input_copy = get_matching_station_ids(
        target_copy, input_copy, basin_dim=basin_dim
    )

    #  TODO: how can we shuffle to increase the spatial distances between the points?
    mapping = dict(
        zip(target_copy[basin_dim].values, shuffle(target_copy[basin_dim].values),)
    )
    inv_mapping = {
        #  shuffled key: original value
        v: k
        for (k, v) in mapping.items()
    }

    # input_copy[basin_dim] = [mapping[sid] for sid in input_copy[basin_dim].values]
    # target_copy[basin_dim] = [mapping[sid] for sid in target_copy[basin_dim].values]

    return mapping, input_copy, target_copy


def _get_train_test_target_input_datasets(
    target_ds: xr.Dataset,
    input_ds: xr.Dataset,
    train_start_date: pd.Timestamp = pd.to_datetime("1998-01-01"),
    train_end_date: pd.Timestamp = pd.to_datetime("2006-09-30"),
    test_start_date: pd.Timestamp = pd.to_datetime("2006-10-01"),
    test_end_date: pd.Timestamp = pd.to_datetime("2009-10-01"),
    subset_pixels: Optional[List[int]] = None,
) -> Tuple[xr.Dataset]:
    #  subset data
    if subset_pixels is not None:
        target_data = target_ds.sel(
            time=slice(train_start_date, train_end_date), station_id=subset_pixels
        )
        input_data = input_ds.sel(
            time=slice(train_start_date, train_end_date), station_id=subset_pixels
        )

        test_target_data = target_ds.sel(
            time=slice(test_start_date, test_end_date), station_id=subset_pixels
        )
        test_input_data = input_ds.sel(
            time=slice(test_start_date, test_end_date), station_id=subset_pixels
        )

    else:
        target_data = target_ds.sel(time=slice(train_start_date, train_end_date))
        input_data = input_ds.sel(time=slice(train_start_date, train_end_date))

        test_target_data = target_ds.sel(time=slice(test_start_date, test_end_date))
        test_input_data = input_ds.sel(time=slice(test_start_date, test_end_date))

    return (target_data, input_data, test_target_data, test_input_data)


def create_train_test_datasets(
    target_var: str,
    input_variables: List[str],
    target_ds: xr.Dataset,
    input_ds: xr.Dataset,
    train_start_date: pd.Timestamp = pd.to_datetime("1998-01-01"),
    train_end_date: pd.Timestamp = pd.to_datetime("2006-09-30"),
    test_start_date: pd.Timestamp = pd.to_datetime("2006-10-01"),
    test_end_date: pd.Timestamp = pd.to_datetime("2009-10-01"),
    subset_pixels: Optional[List[int]] = None,
    seq_length: int = 1,
    basin_dim: str = "station_id",
    time_dim: str = "time",
) -> Tuple[Dataset, Dataset]:

    (
        target_data,
        input_data,
        test_target_data,
        test_input_data,
    ) = _get_train_test_target_input_datasets(
        target_ds=target_ds,
        input_ds=input_ds,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        subset_pixels=subset_pixels,
    )

    # create pytorch dataloaders
    train_dataset = TimeSeriesDataset(
        input_data=input_data,
        target_data=target_data,
        target_variable=target_var,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Train Samples",
    )

    test_dataset = TimeSeriesDataset(
        input_data=test_input_data,
        target_data=test_target_data,
        target_variable=target_var,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Test Samples",
    )

    return (train_dataset, test_dataset)


if __name__ == "__main__":
    #  load data
    from pathlib import Path

    data_dir = Path("/datadrive/data")
    target_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/interpolated_esa_cci_sm.nc")
    input_data = xr.open_dataset(
        data_dir / "SOIL_MOISTURE/interpolated_normalised_camels_gb.nc"
    )

    # #  initialize dataset
    # td = TimeSeriesDataset(
    #     input_data=input_data,
    #     target_data=target_data,
    #     target_variable="sm",
    #     input_variables=["precipitation"],
    #     seq_length=64,
    #     basin_dim="station_id",
    #     time_dim="time",
    # )

    # initialize dataloaders]
    batch_size = 256
    seq_length = 64
    num_workers = 4
    input_variables = ["precipitation"]

    train_dl, test_dl = get_train_test_dataloader(
        input_data=input_data,
        target_data=target_data,
        target_variable="sm",
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim="station_id",
        time_dim="time",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data = train_dl.__iter__().__next__()
    assert data["x_d"].shape == (batch_size, seq_length, len(input_variables))

    times = data["meta"]["time"].numpy().astype("datetime64[ns]")
    assert False
