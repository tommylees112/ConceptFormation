import xarray as xr
from pathlib import Path
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    # data_dir = Path("/Users/tommylees/github/spatio_temporal/data")
    data_dir = Path(".")
    data_dir = Path("/lustre/soge1/projects/crop_yield/SOIL_MOISTURE")
    (data_dir / "DAILY").mkdir(exist_ok=True)

    hourly_files = sorted(list(data_dir.glob("*water*.nc")))
    assert hourly_files != []

    #  resample
    pbar = tqdm(hourly_files, desc="Resampling H->D")
    for fp in pbar:
        outfile = data_dir / "DAILY" / fp.name
        pbar.set_postfix_str(f"Resampling file: {fp.name}")

        if not outfile.exists():
            resampled = xr.open_dataset(fp).resample(time="D").mean()

            #  save resampled output
            resampled.to_netcdf(outfile)
        else:
            print(f"{fp.name} Already Created!")
