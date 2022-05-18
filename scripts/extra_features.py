from typing import Union, List, Tuple
import numpy as np
import xarray as xr
from pathlib import Path


def encode_doys(
    doys: Union[int, List[int]], start_doy: int = 1, end_doy: int = 366
) -> Tuple[List[float], List[float]]:
    """
    encode (list of) date(s)/doy(s) to cyclic sine/cosine values
    int is assumed to represent a day of year
    
    it is possible to change the encoding period by passing `start_doy` or
    `end_doy` to the function
    (e.g. if you want to have cyclic values for the vegetation period only)
    returns two lists, one with sine-encoded and one with cosine-encoded doys
    """
    if not isinstance(doys, list):
        doys = [doys]

    doys_sin = []
    doys_cos = []
    for doy in doys:
        if doy > 366 or doy < 1:
            raise ValueError(f'Invalid date "{doy}"')

        doys_sin.append(
            np.sin(2 * np.pi * (doy - start_doy) / (end_doy - start_doy + 1))
        )
        doys_cos.append(
            np.cos(2 * np.pi * (doy - start_doy) / (end_doy - start_doy + 1))
        )

    return doys_sin, doys_cos


def join_camels_to_era5land(data_dir: Path):
    #  load era5 land data
    filepath = data_dir / "camels_basin_ERA5Land_sm.nc"
    era5_ds = xr.open_dataset(filepath)

    if not isinstance(era5_ds, xr.Dataset):
        era5_ds = era5_ds.to_dataset()

    # load camels data
    ds = xr.open_dataset(data_dir / "RUNOFF/ALL_dynamic_ds.nc")

    _valid_times = np.isin(ds.time, era5_ds.time)
    valid_times = era5_ds["time"][np.isin(era5_ds["time"], ds["time"][_valid_times])]

    extended_ds = xr.merge(
        [
            ds.sel(station_id=era5_ds.station_id, time=valid_times),
            era5_ds.sel(time=valid_times),
        ]
    )
    return extended_ds


if __name__ == "__main__":
    pass
    data_dir = Path("/datadrive/data")
    extended_ds = join_camels_to_era5land(data_dir)
    extended_ds.to_netcdf(data_dir / "extended_camels_data.nc")
