"""To use the api one has to
1. Have an account and log in
2. Go to the data page and accept Terms of Use: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
3. Copy the api url and key from this page: https://cds.climate.copernicus.eu/how-to-api
4. Create a file called ".cdsapirc" containing the url and key in ones home/user folder. The file should have two lines.
5. You may be missing eccodes and cfgrib, which are needed to read the grib files. You can install them using brew install eccodes and pip install cfgrib. 
"""
import os
import cdsapi
import xarray as xr

def download_era5_data(months: list[str]=["01","02","12"], days: list[str]=[str(i).zfill(2) for i in range(1,32)], test_mode: bool=False, full: bool=False):
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "potential_vorticity",
            "temperature",
            "geopotential"
        ],
        "year": [ "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
        "month": months,
        "day": days,
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "pressure_level": ["450", "500","550"],
        "data_format": "grib",
        #"download_format": "unarchived"
    }
    if not full:
        request["area"]=[70, -180, 46, 180]  # North, West, South, East

    grib_file = "./data/era5/era5.grib"
    os.makedirs(os.path.dirname(grib_file), exist_ok=True)

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(grib_file)
    print(f"Downloaded ERA5 data to {grib_file}")

    ds = xr.open_dataset(grib_file, engine='cfgrib')
    save_path = './data/era5/zarr'
    if test_mode:
        save_path += "_test"
    if full:
        save_path += "_full"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ds.to_zarr(save_path,mode="w")
    print(f"Converted GRIB file to Zarr format at {save_path}")

    #delete the grib file to save space
    os.remove(grib_file)
    print(f"Deleted temporary GRIB file {grib_file}")

if __name__ == "__main__":
    download_era5_data(test_mode=False)