from typing import List
from pathlib import Path
import time 
import shutil
import xarray as xr
import pandas as pd
import numpy as np
import socket


def get_data_dir() -> Path:
    location = socket.gethostname()
    if location == "Tommy-Lees-MacBook-Air.local":
        print("On Machine: laptop")
        data_dir = Path(".").home() / "github/spatio_temporal/data"
    elif location == "MantleNucleus":
        print("On Machine: nucleus")
        data_dir = Path("/cats/datastore/data")
    elif location == "LinVmGPU":
        print("On Machine: iiasa")
        data_dir = Path("/DataDrive200/data")
    elif location == "GPU-MachineLearning":
        print("On Machine: azure")
        data_dir = Path("/datadrive/data")
    elif ".ouce.ox.ac.uk" in location:
        print("On Machine: ouce")
        data_dir = Path("/lustre/soge1/projects/crop_yield/data")
    elif "nvidia-ngc-pytorch-test-b-1-vm" == location:
        print("On Machine: oxeo-pytorch")
        data_dir = Path("/home/thomas_lees112/data")

    else:
        assert False, "What machine are you on?"

    assert (
        data_dir.exists()
    ), f"Expect data_dir: {data_dir} to exist. Current Working Directory: {Path('.').absolute()}"
    return data_dir


def move_directory(
    from_path: Path, to_path: Path, with_datetime: bool = False
) -> None:
    if with_datetime:
        dt = time.gmtime()
        dt_str = f"{dt.tm_year}_{dt.tm_mon:02}_{dt.tm_mday:02}:{dt.tm_hour:02}{dt.tm_min:02}{dt.tm_sec:02}"
        name = "/" + dt_str + "_" + to_path.as_posix().split("/")[-1]
        to_path = "/".join(to_path.as_posix().split("/")[:-1]) + name
        to_path = Path(to_path)
    shutil.move(from_path.as_posix(), to_path.as_posix())
    print(f"MOVED {from_path} to {to_path}")


def round_time_to_hour(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    #  beacause of imprecise storage of datetime -> float
    ds[time_dim] = [pd.to_datetime(t).round("H") for t in ds[time_dim].values]
    return ds


def save_basin_list_to_txt_file(fname, basins: List[int]):
    np.savetxt(fname, basins, fmt="%s")
