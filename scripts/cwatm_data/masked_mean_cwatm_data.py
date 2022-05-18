from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import dask
import pickle
import pandas as pd
import geopandas as gpd
from datetime import datetime
from scripts.utils import get_data_dir


def save_basin_list_to_txt_file(fname, basins: List[int]):
    np.savetxt(fname, basins, fmt="%s")


def mean_basin_cwatm_data_with_dask(
    var: str,
    filepath: Path,
    mask: xr.Dataset,
    basin_dim: str = "basin",
    chunks: Dict[str, Any] = {"time": 10},
):
    # load in the gridded input data
    input_ds = xr.open_dataset(filepath, chunks=chunks)
    if "lambert_azimuthal_equal_area" in input_ds.data_vars:
        input_ds = input_ds.drop("lambert_azimuthal_equal_area")

    input_da = input_ds[[v for v in input_ds.data_vars][0]]

    all_mean = []
    for basin in tqdm(mask[basin_dim].values, desc=f"{var} Chopping Basin"):
        basin_mask = mask.sel({basin_dim: basin})
        if "lambert_azimuthal_equal_area" in basin_mask.coords:
            basin_mask = basin_mask.drop("lambert_azimuthal_equal_area")

        basin_ds = input_da.where(basin_mask)
        mean_data = basin_ds.mean(dim=["x", "y"])
        all_mean.append(mean_data)

    all_mean = xr.concat(all_mean, dim=basin_dim)

    return all_mean


def create_timestamped_filename(filepath: Path) -> Path:
    yr, mn, day, hr, min = (
        datetime.now().year,
        datetime.now().month,
        datetime.now().day,
        datetime.now().hour,
        datetime.now().minute,
    )
    new_filepath_name = f"{yr}{mn:02}{day:02}{hr:02}{min:02}_{filepath.name}"
    return filepath.parent / new_filepath_name


def save_to_filepath_else_to_home_directory(ds: xr.Dataset, filepath: Path) -> None:
    try:
        ds.to_netcdf(filepath)
        print(f"Saved data to: {filepath}")
    except PermissionError:
        new_filepath = Path(".").home() / filepath.name
        ds.to_netcdf(new_filepath)
        print(f"Saved data to: {new_filepath}")


if __name__ == "__main__":
    INPUT = False
    TARGET = False
    HIDDEN = False
    JOIN = True
    STATIC = False
    NUM_WORKERS = 4
    DEBUG = False

    data_dir = get_data_dir()
    cwat_dir = data_dir / "CWATM"
    assert cwat_dir.exists()

    #  load the basin mask
    mask = xr.open_dataset(cwat_dir / "basin_mask.nc")["mask"]

    # initialise_datasets
    input_files = {
        "Precipitation": cwat_dir / "Precipitation_daily.nc",
        "Tavg": cwat_dir / "Tavg_daily.nc",
    }
    target_file = cwat_dir / "discharge_daily.nc"
    hidden_files = {
        "channelStorage": cwat_dir / "channelStorage_daily.nc",
        "storGroundwater": cwat_dir / "storGroundwater_daily.nc",
        "sum_interceptStor": cwat_dir / "sum_interceptStor_daily.nc",
        "SnowCover": cwat_dir / "SnowCover_daily.nc",
        "sum_w1": cwat_dir / "sum_w1_daily.nc",
        "sum_w2": cwat_dir / "sum_w2_daily.nc",
        "sum_w3": cwat_dir / "sum_w3_daily.nc",
    }

    #  create folder structure
    output_dir = data_dir / "cwatm_new_basins"
    output_dir.mkdir(exist_ok=True, parents=True)
    assert output_dir.exists()

    # linux14.ouce.ox.ac.uk

    if INPUT:
        all_vars = []
        for var, filepath in input_files.items():
            all_mean = mean_basin_cwatm_data_with_dask(
                var=var, filepath=filepath, mask=mask
            )
            all_vars.append(all_mean)

        all_vars = xr.merge(all_vars)
        basin_coord = "basin"
        all_vars[basin_coord] = all_vars[basin_coord].astype(int)

        with ProgressBar():
            #  join all variables in input files
            if DEBUG:
                dask.config.set(scheduler="single-threaded")
            print("\n*** Computing the spatial means for input data ***\n")
            results = all_vars.compute(num_workers=NUM_WORKERS)

        input_filepath = output_dir / "input_data.nc"
        if input_filepath.exists():
            input_filepath = create_timestamped_filename(input_filepath)

        save_to_filepath_else_to_home_directory(results, input_filepath)

    if HIDDEN:
        assert all(
            [hidden_file.exists() for hidden_file in hidden_files.values()]
        ), f"Expect hidden files to be found here: {cwat_dir}. Found: \n{list(cwat_dir.iterdir())}"
        all_vars = []
        for var, filepath in hidden_files.items():
            all_mean = mean_basin_cwatm_data_with_dask(
                var=var, filepath=filepath, mask=mask
            )
            all_vars.append(all_mean)

        basin_coord = "basin"
        all_vars = xr.merge(all_vars)
        all_vars[basin_coord] = all_vars[basin_coord].astype(int)

        with ProgressBar():
            #  join all variables in input files
            print("\n*** Computing the spatial means for hidden data ***\n")
            results = all_vars.compute(num_workers=NUM_WORKERS)

        hidden_filepath = output_dir / "hidden_data.nc"
        if hidden_filepath.exists():
            hidden_filepath = create_timestamped_filename(hidden_filepath)

        save_to_filepath_else_to_home_directory(results, hidden_filepath)

    if TARGET:
        #  choose how to chunk data
        chunks = {"x": 10, "y": 10} if True else {"time": 10}
        ds = xr.open_dataset(target_file, chunks=chunks).transpose("time", "x", "y")
        locations = xr.open_dataset(cwat_dir / "basin_locations.nc")

        all_basins = []
        for basin in tqdm(locations.basin.values, desc="Extracting gauge discharge"):
            x, y = locations.sel(basin=basin)["location"].values
            basin_data = ds.isel(x=int(x), y=int(y)).assign_coords({"basin": basin})
            basin_data = basin_data.drop(["x", "y", "lambert_azimuthal_equal_area"])[
                "discharge"
            ]
            all_basins.append(basin_data)

        basin_coord = "basin"
        all_basins = xr.concat(all_basins, dim=basin_coord)
        all_basins[basin_coord] = all_basins[basin_coord].astype(int)
        all_basins.attrs = {}

        # write to disk in delayed form
        target_filepath = output_dir / "target_data.nc"
        if target_filepath.exists():
            target_filepath = create_timestamped_filename(target_filepath)

        with ProgressBar():
            print("\n*** Computing the extracted gauge discharge ***\n")
            results = all_basins.compute(num_workers=NUM_WORKERS)
        print("Computed!")

        pickle.dump(results, open(target_filepath.parent / "target_dump.pkl", "wb"))
        save_to_filepath_else_to_home_directory(results, target_filepath)
        assert results['discharge'].isnull().mean() != 1, "All NaN values extracted!"

    if JOIN:
        nc_files = [f for f in list(output_dir.glob("*.nc"))]
        all_ds = [xr.open_dataset(f) for f in nc_files]

        basin_coord = "basin"
        # get only the matching station_ids
        list_of_sids = [all_ds[i][basin_coord].values for i in range(len(nc_files))]
        shortest_sids_idx = np.argmin([len(lst) for lst in list_of_sids])
        sids = list_of_sids[shortest_sids_idx]

        ds = xr.merge([ds.sel({basin_coord: sids}) for ds in all_ds])
        ds[basin_coord] = ds[basin_coord].astype(int)

        whole_data_filepath = data_dir / "cwatm_ORIG_GRID.nc"
        if whole_data_filepath.exists():
            whole_data_filepath = create_timestamped_filename(whole_data_filepath)

        save_to_filepath_else_to_home_directory(ds, whole_data_filepath)

    if STATIC:
        train_start_date, train_end_date = (
            pd.Timestamp("1990-01-01 00:00:00"),
            pd.Timestamp("2000-12-31 00:00:00"),
        )
        test_start_date, test_end_date = (
            pd.Timestamp("2001-01-01 00:00:00"),
            pd.Timestamp("2010-12-31 00:00:00"),
        )

        #  load shapefiles
        shp_dir = cwat_dir / "shapefile"
        basin_shapes = gpd.read_file(list(shp_dir.glob("*.shp"))[0])
        basin_shapes = basin_shapes.rename({"value": "basin"}, axis=1)
        basin_shapes = basin_shapes.set_index("basin")

        #  create mean over dynamic variables
        whole_data_filepath = data_dir / "cwatm_ORIG_GRID.nc"
        assert whole_data_filepath.exists()
        ds = xr.open_dataset(whole_data_filepath)

        static_df = (
            ds.sel(time=slice(train_start_date, train_end_date))
            .mean(dim="time")
            .drop("discharge")
            .to_dataframe()
        )
        static_df = static_df.rename(
            {c: f"{c}_mean" for c in static_df.columns}, axis=1
        )
        #  and area
        areas = basin_shapes.to_crs("epsg:3035").area
        static_df = static_df.join(areas.rename("area"))

        #  save static datas
        static = static_df.to_xarray()
        static_filepath = data_dir / "cwatm_static.nc"
        if static_filepath.exists():
            static_filepath = create_timestamped_filename(static_filepath)

        save_to_filepath_else_to_home_directory(static, static_filepath)

        fname = Path(".").home() / "neuralhydrology/data/all_cwatm_basins_orig_grid.txt"
        save_basin_list_to_txt_file(fname, basins=static.basin.values.astype(int))
