from typing import List, Any
from pathlib import Path
from pyflwdir import basins
import xarray as xr
from tqdm import tqdm


def convert_dataset_to_camels_gb_csv_files(ds: xr.Dataset, out_dir: Path, basin_dim: str = "station_id", time_dim: str = "time"):
    #  create out_dir
    [
        (out_dir / _dir).mkdir(exist_ok=True, parents=True)
        for _dir in ["attributes", "timeseries"]
    ]

    # write the filename to fit neuralhydrology/datasetzoo/camelsgb.py
    file_base = f"CAMELS_GB_hydromet_timeseries_"

    #  convert into station-timeseries csv files
    pbar = tqdm(ds[basin_dim].values, desc="Writing Dataset to csv")
    for sid in pbar:
        pbar.set_postfix_str(sid)
        sid_ds = ds.sel({basin_dim: sid}).rename({time_dim: "date", basin_dim: "basin"})
        filename = file_base + str(int(sid)) + "_CWATM.csv"
        sid_ds.to_dataframe().to_csv(out_dir / "timeseries" / filename)


def convert_static_xarray_to_attributes_csv(static: xr.Dataset, out_dir: Path, basin_dim: str = "station_id"):
    # need to save in the (out_dir / attributes) folder
    (out_dir / "attributes").mkdir(exist_ok=True, parents=True)
    static = static.rename({basin_dim: "basin"})
    static = static.to_dataframe().reset_index()
    # need a gauge_id columns
    static = static.rename({"basin": "gauge_id"}, axis=1)
    # need to match the "*_attributes.csv" glob pattern
    static.to_csv(out_dir / "attributes" / "ALL_attributes.csv")


def write_train_test_basins_to_txt(format_str: str, basins: List[Any], data_dir: Path):
    output_file = f"{format_str}_basin_list.txt"
    with open(data_dir / output_file, "w") as f:
        f.write("\n".join(basins))


if __name__ == "__main__":
    nh_repo_dir = Path("/home/tommy/neuralhydrology/data")
    data_dir = Path("/datadrive/data")
    format_str = "stage"

    #  WRITE out train data file
    if format_str == "camels":
        #  load dataset
        ds = xr.open_dataset(data_dir / "cwatm.nc")
        out_dir = data_dir / "cwatm_camels_gb"
        # write timseries as csv files
        convert_dataset_to_camels_gb_csv_files(ds=ds, out_dir=out_dir)

    elif format_str == "original":
        ds = xr.open_dataset(data_dir / "202108171219_cwatm_ORIG_GRID.nc")
        static = xr.open_dataset(data_dir / "cwatm_static.nc")
        out_dir = data_dir / "cwatm_orig_grid"
        convert_dataset_to_camels_gb_csv_files(ds=ds, out_dir=out_dir, basin_dim="basin", time_dim="time")
        convert_static_xarray_to_attributes_csv(static, out_dir, basin_dim="basin")

        basins = [str(sid) for sid in ds.basin.values]
        write_train_test_basins_to_txt(format_str=format_str + "grid", basins=basins, data_dir=nh_repo_dir)

    elif format_str == "stage":
        # stage dataset
        ds = xr.open_dataset(data_dir / "camels_river_level_data.nc")
        if "station_id" in ds.dims.keys():
            ds = ds.rename({"station_id": "basin"})
        static = xr.open_dataset(data_dir / "camels_static.nc")
        print(static)
        if "station_id" in static.dims.keys():
            static = static.rename({"station_id": "basin"})
        out_dir = data_dir / "stage_camels_gb"

        # convert to camelsgb format
        convert_dataset_to_camels_gb_csv_files(ds=ds, out_dir=out_dir, basin_dim="basin", time_dim="time")
        convert_static_xarray_to_attributes_csv(static, out_dir, basin_dim="basin")

        basins = [str(sid) for sid in ds.basin.values]
        write_train_test_basins_to_txt(format_str=format_str, basins=basins, data_dir=nh_repo_dir)

    else:
        assert False
