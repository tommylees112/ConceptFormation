import xarray as xr
from pathlib import Path
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    # data_dir = Path("/Users/tommylees/github/spatio_temporal/data")
    data_dir = Path(".")
    data_dir = Path("/lustre/soge1/projects/crop_yield/SOIL_MOISTURE")

    # join all of the same variable
    all_output_ds = []
    pbar = tqdm(np.arange(1, 6), desc="Joining Time: ")
    for level in pbar:
        if level < 5:
            level_fps = list(
                (data_dir / "DAILY").glob(f"*volumetric_soil_water*{level}*.nc")
            )
        else:
            level_fps = list(
                (data_dir / "DAILY").glob("*snow_depth_water_equivalent.nc")
            )
        pbar.set_postfix_str(f"Level: {level if level < 5 else 'snow'}")

        all_ds = [xr.open_dataset(fp) for fp in level_fps]
        output_ds = xr.concat(all_ds, dim="time")
        all_output_ds.append(output_ds)

    out = xr.merge(all_output_ds)
    out.to_netcdf("gb_soil_moisture_1234_snow.nc")
    