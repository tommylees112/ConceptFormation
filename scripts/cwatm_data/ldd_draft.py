from typing import Tuple, Dict, List, Optional
from pathlib import Path
import pyflwdir
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr
from pyflwdir import FlwdirRaster
from affine import Affine
from rasterio import features
import geopandas as gpd
import pandas as pd

from scripts.cwatm_data.cwatm_to_camels_dataset import (
    initalise_rio_geospatial,
    reproject_ds,
)
from scripts.utils import get_data_dir


def load_basin_file_ldd(
    data_dir: Path, init_crs: str = "epsg:3035", transform_crs: Optional[str] = None
) -> xr.Dataset:
    ds = xr.open_dataset(data_dir / "ldd.nc")
    uk_mask = xr.open_dataset(data_dir / "cwatm_output_data_grid.nc")
    ldd = ds.sel(x=uk_mask.x, y=uk_mask.y)["ldd"]

    ldd = initalise_rio_geospatial(ldd, crs=init_crs, lon_dim="x", lat_dim="y")
    if transform_crs is not None:
        ldd = reproject_ds(ldd, reproject_crs=transform_crs)  #  EPSG:4326
    return ldd


def convert_each_unique_value_in_dataarray_to_dimension(
    ds: xr.DataArray,
) -> xr.DataArray:
    """Create a new dimension, with a boolean mask for each unique value in ds
    """
    # Create a boolean mask for each unique value in ds
    unique_values = np.unique(ds.values)
    all_masks = []
    for value in unique_values:
        mask = np.zeros(ds.shape, dtype=bool)
        mask[ds.values == value] = True
        all_masks.append(mask)

    ma = np.array(all_masks)

    #  create xarray object
    ds_masks = xr.Dataset(
        {"mask": (["basin", ds.dims[0], ds.dims[1]], ma)},
        coords={"basin": unique_values, "x": ds.x, "y": ds.y},
    )
    return ds_masks["mask"]


def _reset_basin_ids(basins: np.ndarray) -> np.ndarray:
    # this is just to reduce the indices of the leftover basins
    unique_basin_ids = np.unique(basins)[1:]  # remove first value, should be -1
    basin_ids_index = np.full(basins.max() + 2, -1, dtype=np.int32)
    basin_ids_index[unique_basin_ids] = np.arange(unique_basin_ids.size)
    basins = basin_ids_index[basins]

    return basins


def filter_basins_by_size(
    basins: np.ndarray, min_basin_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    # get size of basins
    basin_ids, basin_inverse, basin_size = np.unique(
        basins, return_counts=True, return_inverse=True
    )

    # create map where each pixel of basin has the size of the basin
    basin_size_map = basin_size[basin_inverse].reshape(basins.shape)

    # remove basins smaller than MIN_BASIN_SIZE
    basins[basin_size_map < min_basin_size] = -1

    return basins, basin_size_map


def get_pour_point_locations_for_basins_from_upstream_area(
    basins: np.ndarray, flw: FlwdirRaster
) -> Tuple[Dict[str, Tuple[int, int]], np.ndarray]:
    unique_basins = np.unique(basins)
    gauge_rowcols = dict()
    errors = []
    #  calculate upstream area
    upstr_area = flw.upstream_area()

    #  get the point of largest upstream area for that basin
    for basin_id in tqdm(unique_basins, desc="Calculating basin pour-points"):
        try:
            #  create masked array for that basin
            ma = (
                np.ma.array(upstr_area, mask=basins != basin_id)
                .astype(float)
                .filled(np.nan)
            )
            max_idx = np.nanargmax(ma)
            pour_point_xy = np.unravel_index(max_idx, basins.shape)
            gauge_rowcols[basin_id] = pour_point_xy[::-1]
        except IndexError:
            # print(f"Outlet for BasinID: {basin_id} Not found")
            errors.append(basin_id)
            pass

    return gauge_rowcols, errors


def delineate_subbasins(
    flwdir: np.ndarray, min_basin_size: int
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    # delineate subbasins
    #  By default the method uses pits from the flow direction raster as outlets to delineate basins
    flw = pyflwdir.from_array(flwdir.astype("uint8"), ftype="ldd", cache=True)

    # create basins
    basins = flw.basins().astype(np.int32)
    basins, _ = filter_basins_by_size(basins, min_basin_size)
    basins = _reset_basin_ids(basins)

    #  get the pit locations for each basin
    gauge_rowcols, _ = get_pour_point_locations_for_basins_from_upstream_area(
        basins, flw=flw
    )

    return basins, gauge_rowcols


def gauge_rowcols_dict_to_xarray(
    gauge_rowcols: Dict[str, Tuple[int, int]]
) -> xr.Dataset:
    gauge_xy_array = np.array([v for v in gauge_rowcols.values()])
    basin_ids = np.array([v for v in gauge_rowcols.keys()])
    locations = xr.Dataset(
        {"location": (["basin", "xy"], gauge_xy_array)},
        coords={"basin": basin_ids, "xy": ["x", "y"]},
    )

    return locations


def vectorize(data: np.ndarray, nodata: int, transform: Affine, crs: int = 4326):
    feats_gen = features.shapes(
        data, mask=data != nodata, transform=transform, connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {"value": val}}
        for geom, val in list(feats_gen)
    ]

    # parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    return gdf


def vectorize_rowcol_to_latlon_points(
    gauge_rowcols: Dict[str, Tuple[int, int]], ldd: xr.Dataset, crs: int = 3035
) -> gpd.GeoDataFrame:
    # translate rowcol indexes to latlon
    gauge_latlons = {}
    for basin_id, rowcol in gauge_rowcols.items():
        x, y = rowcol
        x_lon = ldd.isel(x=x, y=y)["x"]
        y_lat = ldd.isel(x=x, y=y)["y"]
        gauge_latlons[basin_id] = (x_lon, y_lat)
    
    # convert to GeoSeries
    d = pd.DataFrame(gauge_latlons).rename({0: "lat", 1: "lon"}).T
    points = gpd.GeoSeries(
        gpd.points_from_xy(d["lat"], d["lon"]), index=d.index, crs=crs,
    ).to_crs("epsg:4326")
    points.name = "geometry"
    points = points.to_frame()

    # add the original metadata for the points
    points["original_loc"] = gpd.GeoSeries(gpd.points_from_xy(d["lat"], d["lon"]), index=d.index, crs=crs)
    points["original_x"] = [p.x for p in points["original_loc"]]
    points["original_y"] = [p.y for p in points["original_loc"]]
    points["x"] = [xy[0] for xy in gauge_rowcols.values()]
    points["y"] = [xy[1] for xy in gauge_rowcols.values()]
    points = points.drop("original_loc", axis=1)
    
    return points


if __name__ == "__main__":
    PLOT = False
    write_shapefile: bool = True
    MIN_BASIN_SIZE = 20  # should be minimum of N pixels (total = 30: 385; 20: 551)

    data_dir = get_data_dir()
    cwat_dir = data_dir / "CWATM"

    # load ldd data (flow direction)
    ldd = load_basin_file_ldd(cwat_dir)
    transform = ldd.rio.transform()
    crs = ldd.rio.crs
    latlon = crs.to_epsg() == 4326
    flwdir = ldd.values

    basins, gauge_rowcols = delineate_subbasins(
        flwdir=flwdir, min_basin_size=MIN_BASIN_SIZE
    )

    #  TODO: convert and save to a shapefile ...
    if write_shapefile:
        shapefile_out_dir = cwat_dir / "shapefile"
        shapefile_out_dir.mkdir(exist_ok=True, parents=True)
        ldd_gdf = vectorize(data=basins, nodata=-1, transform=transform, crs=crs)
        ldd_gdf = ldd_gdf.loc[[sid for sid in list(gauge_rowcols.keys()) if sid != -1]]
        ldd_gdf = ldd_gdf.to_crs("EPSG:4326")  #  convert to latlon
        ldd_gdf.to_file(shapefile_out_dir / "catchment_boundaries.shp")

    #  remove basins where the pour-point selection has failed
    if PLOT:
        _, basin_size_map = filter_basins_by_size(basins, MIN_BASIN_SIZE)
        max_size_basin = np.unique(basins[basin_size_map == basin_size_map.max()])[0]
        bool_mask = (basins == max_size_basin).astype(int)
        xy = gauge_rowcols[max_size_basin]
        bool_mask[xy] = 10

        plt.close("all")
        plt.imshow(bool_mask)
        plt.show()

    if PLOT:
        # finally plot
        plt.close("all")
        im = plt.imshow(basins)
        plt.colorbar(im)
        plt.show()

    #  OUTPUT BASIN DATA
    ds = xr.ones_like(ldd) * basins
    print(f"Number of Station-IDs: {len(gauge_rowcols.keys())}")

    locations = gauge_rowcols_dict_to_xarray(gauge_rowcols)

    points = vectorize_rowcol_to_latlon_points(gauge_rowcols, ldd=ldd)
    points = points.reset_index()
    points_outfile = cwat_dir / "gauge_latlons"
    points.to_file(points_outfile)

    # drop the missing data basin
    locations = locations.sel(basin=slice(0, locations.basin.max()))
    locations.to_netcdf(cwat_dir / "basin_locations.nc")

    if PLOT:
        plt.close("all")
        ds.plot(vmin=0)
        plt.show()

    ds_mask = convert_each_unique_value_in_dataarray_to_dimension(ds)
    ds_mask = (
        ds_mask.drop("lambert_azimuthal_equal_area")
        if "lambert_azimuthal_equal_area" in ds_mask.coords
        else ds_mask
    )
    #  drop the missing data basins
    ds_mask = ds_mask.sel(basin=locations.basin.values)

    #  save to netcdf
    ds_mask.to_netcdf(cwat_dir / "basin_mask.nc")
    
