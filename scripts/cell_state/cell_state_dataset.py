from typing import Tuple, Any, Optional, Union
import numpy as np
import torch
from torch.utils.data import Subset, Dataset, SubsetRandomSampler
import xarray as xr
import pandas as pd
from tqdm import tqdm


def _fill_gaps_da(da: xr.DataArray, fill: Optional[str] = None) -> xr.DataArray:
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    variable = da.name
    if fill is None:
        return da
    else:
        # fill gaps
        if fill == "median":
            # fill median
            da = da.fillna(da.median())
        elif fill == "interpolate":
            #  fill interpolation
            da_df = da.to_dataframe().interpolate()
            coords = [c for c in da_df.columns if c != variable]
            da = da_df.to_xarray().assign_coords(
                dict(zip(coords, da_df[coords].iloc[0].values))
            )[variable]
    return da


def fill_gaps(
    ds: Union[xr.DataArray, xr.Dataset], fill: Optional[str] = None
) -> Union[xr.DataArray, xr.Dataset]:
    if fill is None:
        return ds
    if isinstance(ds, xr.Dataset):
        pbar = tqdm(ds.data_vars, desc=f"Filling gaps with method {fill}")
        for v in pbar:
            pbar.set_postfix_str(v)
            ds[v] = _fill_gaps_da(ds[v], fill=fill)
    else:
        ds = _fill_gaps_da(ds, fill=fill)
    return ds


def dataset_dimensions_to_variable(
    ds: xr.Dataset,
    variable: str = "cell_state",
    time_dim: str = "time",
    pixel_dim: str = "station_id",
    dimension_to_convert_to_variable_dim: str = "dimension",
) -> xr.Dataset:
    """Convert dimensions to variables.

    Args:
        ds (xr.Dataset): [description]
        variable (str, optional): [description]. Defaults to "cell_state".
        time_dim (str, optional): [description]. Defaults to "time".
        pixel_dim (str, optional): [description]. Defaults to "station_id".
        dimension_to_convert_to_variable_dim (str, optional): [description]. Defaults to "dimension".

    Returns:
        xr.Dataset: [description]
    """
    #  set the order of dimensions
    ds = ds.transpose(time_dim, pixel_dim, dimension_to_convert_to_variable_dim)
    new_ds = xr.Dataset(
        {
            f"dim{i}": ((time_dim, pixel_dim), ds[variable].values[:, :, i])
            for i in range(ds[variable].shape[-1])
        },
        coords={time_dim: ds[time_dim], pixel_dim: ds[pixel_dim]},
    )
    return new_ds


class CellStateDataset(Dataset):
    def __init__(
        self,
        input_data: xr.Dataset,  #  cell state (`hs` dimensions)
        target_data: xr.DataArray,  #  soil moisture
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        device: str = "cpu",
        variable_str: str = "cell_state",
        fill: Optional[str] = None,
    ):
        assert all(np.isin(["time", "dimension", "station_id"], input_data.dims))
        assert "cell_state" in input_data

        #  drop missing / non matching basins
        if not all(
            np.isin(input_data.station_id.values, target_data.station_id.values)
        ):
            input_data = input_data.sel(
                station_id=np.isin(
                    input_data.station_id.values, target_data.station_id.values
                )
            )

        self.input_data = input_data
        self.device = device
        self.variable_str = variable_str

        #  All times that we have data for
        times = pd.date_range(start_date, end_date, freq="D")
        bool_input_times = np.isin(input_data.time.values, times)
        bool_target_times = np.isin(target_data.time.values, times)
        self.all_times = list(
            set(target_data.time.values[bool_target_times]).intersection(
                set(input_data.time.values[bool_input_times])
            )
        )
        self.all_times = sorted(self.all_times)

        # get input/target data
        self.input_data = self.input_data.sel(time=self.all_times)
        self.input_data = fill_gaps(self.input_data, fill=fill)
        target_data = target_data.sel(time=self.all_times)
        self.target_data = fill_gaps(target_data, fill=fill)

        # basins
        self.basins = input_data.station_id.values

        # dimensions
        self.dimensions = len(input_data.dimension.values)

        # create x y pairs
        self.create_samples()

    def __len__(self):
        return len(self.samples)

    def create_samples(self):
        self.samples = []
        self.basin_samples = []
        self.time_samples = []

        for basin in self.basins:
            # read the basin data
            X = (
                self.input_data[self.variable_str]
                .sel(station_id=basin)
                .values.astype("float64")
            )

            Y = self.target_data.sel(station_id=basin).values.astype("float64")

            # Ensure time is the 1st (0 index) axis
            X_time_axis = int(
                np.argwhere(~np.array([ax == len(self.all_times) for ax in X.shape]))
            )
            if X_time_axis != 1:
                X = X.transpose(1, 0)

            # drop nans over time (1st axis)
            finite_indices = np.logical_and(np.isfinite(Y), np.isfinite(X).all(axis=1))
            X, Y = X[finite_indices], Y[finite_indices]
            times = self.input_data["time"].values[finite_indices].astype(float)

            # convert to Tensors
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()

            # convert to device (gpu=cuda:0, cpu)
            X = X.to(self.device)
            Y = Y.to(self.device)

            # create unique samples [(64,), (1,)]
            samples = [(x, y.reshape(-1)) for (x, y) in zip(X, Y)]
            self.samples.extend(samples)
            self.basin_samples.extend([basin for _ in range(len(samples))])
            self.time_samples.extend(times)

        #  SORT DATA BY TIME (important for train-test split)
        sort_idx = np.argsort(self.time_samples)
        self.time_samples = np.array(self.time_samples)[sort_idx]
        try:
            self.samples = np.array(self.samples)[sort_idx]
        except TypeError:
            self.samples = np.array([(x, y) for (x, y) in self.samples])[sort_idx]
        self.basin_samples = np.array(self.basin_samples)[sort_idx]

    def __getitem__(self, item: int) -> Tuple[Tuple[str, Any], Tuple[torch.Tensor]]:
        basin = str(self.basin_samples[item])
        time = self.time_samples[item]
        x, y = self.samples[item]
        data = {"x_d": x, "y": y, "meta": {"basin": basin, "time": time}}

        return data


# train-test split
def get_train_test_dataset(
    dataset: Dataset, test_proportion: float = 0.2
) -> Tuple[Subset, Subset]:
    #  Subset = https://stackoverflow.com/a/59414029
    #  random_split = https://stackoverflow.com/a/51768651
    all_data_size = len(dataset)
    train_size = int((1 - test_proportion) * all_data_size)
    test_size = all_data_size - train_size
    test_index = all_data_size - int(np.floor(test_size))

    #  test data is from final_sequence : end
    test_dataset = Subset(dataset, range(test_index, all_data_size))
    # train data is from start : test_index
    train_dataset = Subset(dataset, range(0, test_index))
    #  sense-check
    assert len(train_dataset) + len(test_dataset) == all_data_size

    return train_dataset, test_dataset


def train_validation_split(
    dataset: Dataset,
    validation_split: float = 0.1,
    shuffle_dataset: bool = True,
    random_seed: int = 42,
) -> Tuple[SubsetRandomSampler]:
    #  SubsetRandomSampler = https://stackoverflow.com/a/50544887
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


if __name__ == "__main__":
    #  generate random but small target/input data
    #  create dataset
    cfg = None
    dataset = CellStateDataset(
        input_data=None,
        target_data=None,
        start_date=cfg.test_start_date,
        end_date=cfg.test_end_date,
    )

    #  train-val, test split data (using the subset)
    train_ds, test_ds = get_train_test_dataset(dataset, test_proportion=0.2)
    train_ds, validation = train_validation_split(train_ds, validation_split=0.1)
    pass
