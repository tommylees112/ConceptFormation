import cdsapi
import numpy as np
from pathlib import Path
from pprint import pprint


download_dict = {
    "format": "netcdf",
    "variable": "nan",
    "year": [
        "1980",
        "1981",
        "1982",
        "1983",
        "1984",
        "1985",
        "1986",
        "1987",
        "1988",
        "1989",
    ],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",],
    "day": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
    ],
    "time": [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
    "area": [58.64, -7.57, 49.96, 1.68,],
}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    MAX_YEARS = 10
    years = np.arange(1980, 2020)
    variables = [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
    variables = variables + ["snow_depth_water_equivalent"]

    for variable in variables:
        # for variable in ["volumetric_soil_water_layer_2"]:
        for subset_years in list(chunks(years, MAX_YEARS)):
            fname = f"{subset_years.min()}_{subset_years.max()}_{variable}.nc"

            if Path(fname).exists():
                print(f"** {fname} already exists! Skipping **")
                continue
            else:
                c = cdsapi.Client()
                download_dict["year"] = [str(year) for year in subset_years]
                download_dict["variable"] = variable

                pprint(download_dict)
                c.retrieve("reanalysis-era5-land", download_dict, fname)
