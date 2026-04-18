"""
Download LeakDB and BattLeDIM datasets from Zenodo.

LeakDB: Realistic leakage scenarios for water distribution networks.
  - Zenodo: https://zenodo.org/records/13985057
  
BattLeDIM: Battle of the Leakage Detection and Isolation Methods (2020).
  - Zenodo: https://zenodo.org/records/4017659

Both are academic benchmark datasets under open licenses.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
LEAKDB_DIR = DATA_DIR / "leakdb"
BATTLEDIM_DIR = DATA_DIR / "battledim"


# ── Download URLs ──────────────────────────────────────────────────────────
# LeakDB: full dataset ZIP from Zenodo
LEAKDB_URL = "https://zenodo.org/records/13985057/files/LeakDB.zip?download=1"

# BattLeDIM: individual files from Zenodo record 4017659
BATTLEDIM_BASE = "https://zenodo.org/records/4017659/files"
BATTLEDIM_FILES = [
    "2018_SCADA.xlsx",
    "2019_SCADA.xlsx",
    "2018_Leakages.csv",
    "2019_Leakages.csv",
    "L-TOWN.inp",
    "2018_Fixed_Leakages_Report.txt",
]


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress indication."""
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return

    print(f"  ↓ Downloading {desc or dest.name}...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                mb_done = downloaded / 1024 / 1024
                mb_total = total / 1024 / 1024
                print(f"\r    {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

    print(f"\n  ✓ Saved: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def download_leakdb() -> None:
    """Download and extract the LeakDB dataset."""
    print("\n" + "=" * 60)
    print("DOWNLOADING LeakDB (Leakage Diagnosis Benchmark)")
    print("=" * 60)

    LEAKDB_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = LEAKDB_DIR / "LeakDB.zip"

    download_file(LEAKDB_URL, zip_path, "LeakDB.zip from Zenodo")

    # Extract if not already extracted
    # Look for a marker that extraction is done
    marker = LEAKDB_DIR / ".extracted"
    if marker.exists():
        print("  ✓ Already extracted")
    else:
        print("  ↓ Extracting LeakDB.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(LEAKDB_DIR)
        marker.touch()
        print("  ✓ Extraction complete")

    # Show what we got
    print("\n  Contents:")
    for item in sorted(LEAKDB_DIR.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            n_children = len(list(item.rglob("*")))
            print(f"    📁 {item.name}/ ({n_children} files)")
        else:
            size_mb = item.stat().st_size / 1024 / 1024
            print(f"    📄 {item.name} ({size_mb:.1f} MB)")


def download_battledim() -> None:
    """Download the BattLeDIM dataset files."""
    print("\n" + "=" * 60)
    print("DOWNLOADING BattLeDIM (Battle of Leak Detection Methods)")
    print("=" * 60)

    BATTLEDIM_DIR.mkdir(parents=True, exist_ok=True)

    for fname in BATTLEDIM_FILES:
        url = f"{BATTLEDIM_BASE}/{fname}?download=1"
        dest = BATTLEDIM_DIR / fname
        download_file(url, dest, fname)

    # Show what we got
    print("\n  Contents:")
    for item in sorted(BATTLEDIM_DIR.iterdir()):
        size_mb = item.stat().st_size / 1024 / 1024
        print(f"    📄 {item.name} ({size_mb:.2f} MB)")


def verify_leakdb() -> None:
    """Verify LeakDB structure — find the Hanoi network scenarios."""
    print("\n" + "=" * 60)
    print("VERIFYING LeakDB STRUCTURE")
    print("=" * 60)

    # LeakDB may extract with nested folders — find the Hanoi data
    hanoi_candidates = list(LEAKDB_DIR.rglob("*Hanoi*"))
    if not hanoi_candidates:
        # Try looking for any CSV with pressure data
        csv_files = list(LEAKDB_DIR.rglob("*.csv"))
        print(f"  Found {len(csv_files)} CSV files total")
        if csv_files:
            print(f"  First few: {[f.name for f in csv_files[:5]]}")
        # Show top-level structure
        print("\n  Top-level structure:")
        for item in sorted(LEAKDB_DIR.iterdir()):
            if item.name.startswith(".") or item.name.endswith(".zip"):
                continue
            print(f"    {'📁' if item.is_dir() else '📄'} {item.name}")
            if item.is_dir():
                for sub in sorted(item.iterdir())[:5]:
                    print(f"      {'📁' if sub.is_dir() else '📄'} {sub.name}")
                remaining = len(list(item.iterdir())) - 5
                if remaining > 0:
                    print(f"      ... and {remaining} more")
    else:
        print(f"  Found Hanoi data in: {hanoi_candidates[0]}")
        # Explore structure
        hanoi_dir = hanoi_candidates[0] if hanoi_candidates[0].is_dir() else hanoi_candidates[0].parent
        print(f"  Path: {hanoi_dir}")
        scenario_dirs = sorted([d for d in hanoi_dir.iterdir() if d.is_dir()])
        if scenario_dirs:
            print(f"  Number of scenario folders: {len(scenario_dirs)}")
            # Show first scenario contents
            first = scenario_dirs[0]
            print(f"\n  First scenario ({first.name}) contents:")
            for f in sorted(first.iterdir()):
                print(f"    📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")


def verify_battledim() -> None:
    """Verify BattLeDIM files exist and are non-empty."""
    print("\n" + "=" * 60)
    print("VERIFYING BattLeDIM FILES")
    print("=" * 60)

    all_ok = True
    for fname in BATTLEDIM_FILES:
        fpath = BATTLEDIM_DIR / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"  ✓ {fname} ({fpath.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ✗ {fname} — MISSING or EMPTY")
            all_ok = False

    if all_ok:
        print("\n  All BattLeDIM files verified ✓")
    else:
        print("\n  ⚠ Some files are missing!")


if __name__ == "__main__":
    print("HydraWatch Data Download Script")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")

    download_leakdb()
    download_battledim()
    verify_leakdb()
    verify_battledim()

    print("\n" + "=" * 60)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 60)
