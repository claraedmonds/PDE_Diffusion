"""
Preprocessing script: converts raw per-year ERA5 zarr stores into a training-optimised
format with the following properties:

  - Variables extracted in canonical channel order: (u, v, pv, t, z) × 3 pressure levels
    → 15 channels, layout matching the model's expected (var lev) channel ordering
  - Spatial downsampling (factor 3) and lat/lon cropping to multiples of 16 applied once
  - Raw (un-normalised) float32 fields stored with chunks=(1, 15, nlon, nlat) so that
    each single-timestep read decompresses exactly one zarr chunk (~0.9 MB)
  - Unix timestamps stored alongside for clock-feature computation at training time

Output layout per year:
    data/era5_proc/zarr_proc_<YEAR>/
        fields/     (T, 15, nlon, nlat)  float32  chunks=(1,15,nlon,nlat)
        time_unix/  (T,)                 int64    Unix seconds since epoch
        longitude/  (nlon,)              float32
        latitude/   (nlat,)              float32

Run via Slurm job array (see sh_files/prepare_data.sh) or directly:
    python src/pde_diff/data/prepare_data.py --year 2015
"""

import argparse
import numpy as np
import xarray as xr
import zarr
from pathlib import Path

FEATURES = ['u', 'v', 'pv', 't', 'z']
DOWNSAMPLE = 3
TIME_CHUNK = 720  # 30 days — limits peak memory per iteration to ~5 GB


def prepare_year(year: int, data_dir: Path, out_dir: Path) -> None:
    src = data_dir / f"zarr_{year}"
    dst = out_dir / f"zarr_proc_{year}"

    if not src.exists():
        raise FileNotFoundError(f"Source zarr not found: {src}")

    print(f"[{year}] Opening {src}")
    ds = xr.open_zarr(str(src))

    # Apply the same spatial transforms as ERA5Dataset.__init__
    ds = ds.isel(
        longitude=slice(0, None, DOWNSAMPLE),
        latitude=slice(0, None, DOWNSAMPLE),
    )
    nlon = ds.sizes['longitude']
    nlat = ds.sizes['latitude']
    if nlon % 16 != 0:
        ds = ds.isel(longitude=slice(0, nlon - nlon % 16))
    if nlat % 16 != 0:
        ds = ds.isel(latitude=slice(0, nlat - nlat % 16))

    nlon = ds.sizes['longitude']
    nlat = ds.sizes['latitude']
    T    = ds.sizes['time']
    print(f"[{year}] Grid: T={T}, lon={nlon}, lat={nlat}")

    # Unix timestamps (int64, seconds since epoch)
    time_unix = ds['time'].values.astype('datetime64[s]').astype(np.int64)

    dst.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(dst), mode='w')

    # Coordinate arrays (small, write once)
    store['longitude']       = ds['longitude'].values.astype(np.float32)
    store['latitude']        = ds['latitude'].values.astype(np.float32)
    store['time_unix']       = time_unix
    store['pressure_levels'] = np.array(ds['isobaricInhPa'].values, dtype=np.float32)

    # Pre-allocate the fields array with chunk size = 1 along time
    n_channels = len(FEATURES) * ds.sizes['isobaricInhPa']
    fields_arr = store.create_array(
        'fields',
        shape=(T, n_channels, nlon, nlat),
        chunks=(1, n_channels, nlon, nlat),
        dtype='float32',
        overwrite=True,
    )

    # Write in time chunks to keep peak memory manageable
    for t_start in range(0, T, TIME_CHUNK):
        t_end = min(t_start + TIME_CHUNK, T)
        chunk_vars = []
        for var in FEATURES:
            # ds[var] dims: (time, isobaricInhPa, latitude, longitude)
            arr = ds[var].isel(time=slice(t_start, t_end)).values  # (chunk, lev, nlat, nlon)
            arr = arr.transpose(0, 1, 3, 2)                        # (chunk, lev, nlon, nlat)
            chunk_vars.append(arr.astype(np.float32))
        # Concatenate along channel dim: (chunk, 15, nlon, nlat)
        chunk_fields = np.concatenate(chunk_vars, axis=1)
        fields_arr[t_start:t_end] = chunk_fields
        print(f"[{year}] Written {t_end}/{T} timesteps")

    print(f"[{year}] Done → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',     type=int, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/era5')
    parser.add_argument('--out-dir',  type=str, default='./data/era5_proc')
    args = parser.parse_args()

    prepare_year(
        year=args.year,
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
    )
