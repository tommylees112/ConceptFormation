"""[summary]
1) Load the LDD map rasterio
2) Load numpy array into pyflwdir 
    PCRASTER convention
    Value: 1-9 (5 is a bit, 1-8 is direction)
    (alternate=arcgis 0-128)
3) Execute flwdir 
    `conda install pyflwdir -c conda-forge`

Documentation:
https://deltares.gitlab.io/wflow/pyflwdir/flwdir.html#Basins 

Convention:
https://pcraster.geo.uu.nl/pcraster/4.3.1/documentation/pcraster_manual/sphinx/secdatbase.html#formldd

Data:
https://iiasahub-my.sharepoint.com/personal/debruijn_iiasa_ac_at2/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9paWFzYWh1Yi1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9kZWJydWlqbl9paWFzYV9hY19hdDIvRWl5N3p2Ri1RVUZDZzFET0NBWXVEajhCdU9PanNRY1Qwby02bkdxSDJZazRyUT9ydGltZT02dG5uOVo1VjJVZw&id=%2Fpersonal%2Fdebruijn%5Fiiasa%5Fac%5Fat2%2FDocuments%2FCWatM%2F1keurope%2Finput%5Fnetcdf%2Flandsurface%2Ftopo 
"""
from typing import final
from affine import Affine
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import cartopy.crs as ccrs
import xarray as xr
import descartes
import numpy as np
from rasterio import features
import rasterio
import geopandas as gpd
import pyflwdir
from tqdm import tqdm

from scripts.cwatm_data.cwatm_to_camels_dataset import (
    initalise_rio_geospatial,
    reproject_ds,
)
from scripts.geospatial import initialise_gb_spatial_plot, load_latlon_points


def quickplot(gdfs=[], maps=[], hillshade=True, title="", filename="flw", save=False):
    fig = plt.figure(figsize=(8, 15))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    # plot hillshade background
    # if hillshade:
    # ls = matplotlib.colors.LightSource(azdeg=115, altdeg=45)
    # hillshade = ls.hillshade(np.ma.masked_equal(elevtn, -9999), vert_exag=1e3)
    # ax.imshow(hillshade, origin='upper', extent=flw.extent, cmap='Greys', alpha=0.3, zorder=0)
    # plot geopandas GeoDataFrame
    for gdf, kwargs in gdfs:
        gdf.plot(ax=ax, **kwargs)
    for data, nodata, kwargs in maps:
        ax.imshow(
            np.ma.masked_equal(data, nodata),
            origin="upper",
            extent=flw.extent,
            **kwargs,
        )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize="large")
    ax.text(
        0.01, 0.01, "created with pyflwdir", transform=ax.transAxes, fontsize="large"
    )
    if save:
        plt.savefig(f"{filename}.png")
    return ax


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


def decide_area_threshold_plot(THRESH: float):
    prop_excluded = float((static["area"] < THRESH).mean()) * 100
    f, ax = plt.subplots(figsize=(12, 4))
    ax.hist(static["area"], density=True, bins=100)
    q1 = float(static["area"].quantile(q=0.1))
    q5 = float(static["area"].quantile(q=0.5))
    ax.axvline(q1, color="r", linestyle="--", label=f"Q0.1: {q1:.2f}")
    ax.axvline(
        THRESH,
        color="k",
        linestyle="--",
        label=f"{THRESH}km$^2$: {prop_excluded:.2f}% excluded",
    )
    ax.axvline(q5, color="b", linestyle="--", label=f"Q0.5: {q5:.2f}")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # data_dir = Path("/Users/tommylees/Downloads/")
    # static_path = data_dir / "camels_static.nc"
    data_dir = Path("/DataDrive200/data/CWATM")
    static_path = data_dir.parent / "static.nc"
    assert data_dir.exists()

    # load ldd data and initialise basins
    ds = xr.open_dataset(data_dir / "ldd.nc")
    uk_mask = xr.open_dataset(data_dir / "cwatm_output_data_grid.nc")
    ldd = ds.sel(x=uk_mask.x, y=uk_mask.y)["ldd"]

    ldd = initalise_rio_geospatial(ldd, crs="epsg:3035", lon_dim="x", lat_dim="y")
    ldd = reproject_ds(ldd, reproject_crs="EPSG:4326")
    transform = ldd.rio.transform()
    crs = ldd.rio.crs
    latlon = crs.to_epsg() == 4326

    #  NOTE: replace water/missing with 5 (according to PCRASTER convention)
    # save a sea mask for later ... (defined by the soil moisture data)
    sea_mask = xr.open_dataset(data_dir / "cwatm_latlon_sm_mask.nc")
    sea_mask = sea_mask[[v for v in sea_mask.data_vars][0]].rename("mask")

    ldd_pcraster = ldd.where(~(abs(ldd) > 9), 5)

    #  convert to FlwdirRaster object
    #  NOTE: has to be of the correct dtype (uint8)
    flw = pyflwdir.from_array(
        ldd_pcraster.values.astype("uint8"),
        ftype="ldd",
        transform=transform,
        latlon=latlon,
        cache=True,
    )

    # delineate subbasins
    #  By default the method uses pits from the flow direction raster as outlets to delineate basins
    #    pits = locations where the value == 5
    #   subbasins = flw.basins(xy=(lons[:1],lats[:1]), streams=flw.stream_order()>=4)
    basins = flw.basins()

    # 1) remove the sea-data by using the sea mask
    masked_basins = np.ma.array(basins, mask=sea_mask)
    unique, counts = np.unique(masked_basins, return_counts=True)

    # 2) remove the basins with fewer than THRESH pixels
    THRESH = 20  # km^2 -> 588 basins
    keep_basins = unique[counts > THRESH]
    keep_basin_mask = ~keep_basins.mask
    keep_basins = keep_basins.data[keep_basin_mask]

    #  keep only the basins with a min area > 1 pixel
    final_counts = counts[counts > THRESH][keep_basin_mask]
    #  sort from largest to smallest (invert default small to large)
    argsort = np.argsort(final_counts)[::-1]

    final_basins = np.ma.array(basins, mask=~np.isin(basins, keep_basins))
    filled_basins = final_basins.astype("float32").filled(np.nan)

    # recode to {0: N} integer
    basin_array = np.zeros_like(filled_basins)
    recode_dict = {v: k for k, v in dict(enumerate(keep_basins)).items()}
    lookup_counts = {k: final_counts[ix] for (ix, k) in enumerate(recode_dict.keys())}

    for row_ix in tqdm(
        np.arange(filled_basins.shape[0]), desc="Recode values in array"
    ):
        for col_ix in np.arange(filled_basins.shape[1]):
            try:
                basin_array[row_ix][col_ix] = recode_dict[filled_basins[row_ix][col_ix]]
            except KeyError:
                assert np.isnan(filled_basins[row_ix][col_ix])  #  should be nan
                basin_array[row_ix][col_ix] = np.nan

    masked_basin_array = np.ma.array(basin_array, mask=np.isnan(basin_array))

    # vectorize
    gdf_bas = vectorize(
        data=masked_basin_array.astype(np.int32),
        nodata=masked_basin_array.fill_value,
        transform=flw.transform,
    )
    # plot
    streams = (gdf[gdf["strord"] >= 6], dict(color="grey"))
    bas = (gdf_bas, dict(edgecolor="black", facecolor="none", linewidth=0.8))
    subbas = (subbasins, 0, dict(cmap="Set3", alpha=0.5))
    ax = quickplot(
        [streams, bas],
        [subbas],
        title="Basins from point outlets",
        filename="flw_basins",
    )

    # # define output locations = get XY of large basins
    static = xr.open_dataset(static_path)
    THRESH = 100  # km^2
    large_basins = static.sel(station_id=static["area"] > THRESH)
    lons, lats = large_basins["gauge_lon"].values, large_basins["gauge_lat"].values

    #  check that the lats/lons fall within the ldd grid
    lon_range = float(ldd.x.min()), float(ldd.x.max())
    lat_range = float(ldd.y.min()), float(ldd.y.max())

    assert np.all([(lon >= lon_range[0]) & (lon <= lon_range[1]) for lon in lons])
    assert np.all([(lat >= lat_range[0]) & (lat <= lat_range[1]) for lat in lats])

    subbasins = flw.basins(xy=(lons, lats))
    subbasins_ma = np.ma.array(subbasins, mask=sea_mask)
