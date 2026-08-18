"""
Merge per-year zarr stores (zarr_20**) into a single consolidated zarr.

Usage:
    python src/pde_diff/data/merge_zarrs.py \
        --data-dir ./data/era5 \
        --years 2015 2016 ... 2024 \
        --out-path ./data/era5/zarr
"""
import argparse
from pathlib import Path
import xarray as xr


def merge_zarrs(data_dir: Path, years: list[str], out_path: Path) -> None:
    zarr_paths = []
    for year in sorted(years):
        p = data_dir / f"zarr_{year}"
        if not p.exists():
            raise FileNotFoundError(f"Expected zarr store not found: {p}")
        zarr_paths.append(str(p))

    print(f"Opening {len(zarr_paths)} zarr stores...")
    ds = xr.open_mfdataset(
        zarr_paths,
        engine="zarr",
        combine="nested",
        concat_dim="time",
        chunks={},          # keeps lazy/dask arrays
        consolidated=False,
    )
    print(f"Combined dataset: {ds}")

    # Rechunk to match the source zarr encoding (270 timesteps per chunk)
    ds = ds.chunk({"time": 270})

    # Per-year coordinate encodings (e.g. valid_time chunk=2160) are meaningless
    # after merging and conflict with our rechunked dask layout — drop them.
    for coord in ds.coords:
        ds[coord].encoding.pop("chunks", None)

    print(f"Writing merged zarr to {out_path} ...")
    ds.to_zarr(
        str(out_path),
        mode="w",
        consolidated=True,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./data/era5"))
    parser.add_argument(
        "--years",
        nargs="+",
        default=[str(y) for y in range(2015, 2025)],
        help="Years to include (default: 2015–2024)",
    )
    parser.add_argument("--out-path", type=Path, default=Path("./data/era5/zarr"))
    args = parser.parse_args()

    merge_zarrs(args.data_dir, args.years, args.out_path)
