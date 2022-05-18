from typing import List, Optional, Union, Any, Dict
import pickle
import geopandas as gpd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def initialise_gb_spatial_plot(ax=None, data_dir: Path = Path("/datadrive/data")):
    #  read UK outline data
    assert (
        data_dir / "RUNOFF/natural_earth_hires/ne_10m_admin_0_countries.shp"
    ).exists(), "Download the natural earth hires from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"

    world = gpd.read_file(
        data_dir / "RUNOFF/natural_earth_hires/ne_10m_admin_0_countries.shp"
    )
    uk = world.query("ADM0_A3 == 'GBR'")

    #  plot UK outline
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 8))
    uk.plot(facecolor="none", edgecolor="k", ax=ax, linewidth=0.3)

    ax.set_xlim([-8.2, 2.1])
    ax.set_ylim([50, 59.5])
    ax.axis("off")
    return ax


def load_latlon_points(data_dir: Path) -> gpd.GeoSeries:
    static = xr.open_dataset(data_dir / "camels_static.nc")
    d = static[["gauge_lat", "gauge_lon"]].to_dataframe()
    points = gpd.GeoSeries(
        gpd.points_from_xy(d["gauge_lon"], d["gauge_lat"]),
        index=d.index,
        crs="epsg:4326",
    )
    points.name = "geometry"
    return points


region_abbrs = {
    "Western Scotland": "WS",
    "Eastern Scotland": "ES",
    "North-east England": "NEE",
    "Severn-Trent": "ST",
    "Anglian": "ANG",
    "Southern England": "SE",
    "South-west England & South Wales": "SWESW",
    "North-west England & North Wales (NWENW)": "NWENW",
}


def get_region_station_within(
    stations: gpd.GeoDataFrame,
    hydro_regions: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
) -> List[str]:
    # find the region that a station belongs WITHIN
    # create a list of strings/nans for each station
    region_dict = {}
    for region, geom in zip(hydro_regions["NAME"], hydro_regions["geometry"]):
        isin_region = [p.within(geom) for p in stations]
        region_dict[region] = [(region if item else np.nan) for item in isin_region]

    region_cols = pd.DataFrame(region_dict)

    # copy non-null values from the right column into the left and select left
    # https://stackoverflow.com/a/49908660
    regions_list = (region_cols.bfill(axis=1).iloc[:, 0]).rename("region")
    regions_list.index = points.index

    return regions_list


def assign_region_coordinate(ds: xr.Dataset, regions_data: pd.DataFrame) -> xr.Dataset:
    # get regions data
    # convert to xarray
    region_xr = regions_data.to_xarray()
    # assign coordinate to ds
    ds = ds.assign_coords(region=regions_data.to_xarray()["region"]).assign_coords(
        region=regions_data.to_xarray()["region_abbr"]
    )
    return ds


def get_regions_data(gis_data_dir: Path, data_dir: Path) -> pd.DataFrame:
    assert (gis_data_dir / "UK_hydroclimate_regions_Harrigan_et_al_2018").exists(), f"Expect to find dir: UK_hydroclimate_regions_Harrigan_et_al_2018 in {gis_data_dir}"
    
    hydro_regions = gpd.read_file(gis_data_dir / "UK_hydroclimate_regions_Harrigan_et_al_2018/UK_Hydro_Regions_ESP_HESS.shp").to_crs(epsg=4326)
    hydro_regions = hydro_regions.loc[~np.isin(hydro_regions["NAME"], ["Northern Ireland", "Republic of Ireland"])]

    points = load_latlon_points(data_dir)
    regions_list = get_region_station_within(points, hydro_regions, points=points)
    regions_data = regions_list.to_frame().join(regions_list.map(region_abbrs).rename("region_abbr"))
    return regions_data


def plot_spatial_location(sids: Union[List[int], int], points: gpd.GeoDataFrame, plot_kwargs: Dict[str, Any] = {}):
    if isinstance(sids, int):
        sids = [sids]
    ax = initialise_gb_spatial_plot()
    points.plot(color="grey", alpha=0.6, markersize=4, ax=ax)
    points.loc[sids].plot(ax=ax, markersize=100, **plot_kwargs)