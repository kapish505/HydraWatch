"""
Extract the full LeakDB Hanoi_CMH dataset (500 scenarios).

The main LeakDB.zip has been downloaded already. Inside it there's a nested
Hanoi_CMH.zip (~4GB) that contains all 500 scenarios. This script extracts
that inner zip to get the full dataset.
"""

import zipfile
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEAKDB_DIR = PROJECT_ROOT / "data" / "raw" / "leakdb" / "LeakDB"

def extract_hanoi_zip():
    """Extract the inner Hanoi_CMH.zip to get all 500 scenarios."""
    zip_path = LEAKDB_DIR / "Hanoi_CMH.zip"
    extract_to = LEAKDB_DIR / "Hanoi_CMH"
    
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found!")
        sys.exit(1)
    
    print(f"ZIP file: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Extracting to: {extract_to}")
    print()
    
    # Check what's already extracted
    existing = sorted([
        d.name for d in extract_to.iterdir() 
        if d.is_dir() and d.name.startswith("Scenario-")
    ]) if extract_to.exists() else []
    
    # Also check nested dir
    nested_dir = extract_to / "Hanoi_CMH"
    if nested_dir.exists():
        nested = sorted([
            d.name for d in nested_dir.iterdir()
            if d.is_dir() and d.name.startswith("Scenario-")
        ])
        existing.extend(nested)
    
    print(f"Already extracted: {len(existing)} scenarios")
    
    # Open the zip and list contents
    print("Opening Hanoi_CMH.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get all scenario directories in the zip
        all_names = zf.namelist()
        scenario_dirs = set()
        for name in all_names:
            parts = name.split('/')
            for p in parts:
                if p.startswith('Scenario-'):
                    scenario_dirs.add(p)
                    break
        
        print(f"Scenarios in ZIP: {len(scenario_dirs)}")
        print(f"Total files in ZIP: {len(all_names)}")
        print()
        print("Starting extraction (this may take several minutes)...")
        
        total = len(all_names)
        for i, member in enumerate(all_names):
            zf.extract(member, extract_to)
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"  Progress: {i+1}/{total} files ({pct:.1f}%)")
    
    # Verify
    print("\nVerifying extraction...")
    
    # Count scenarios in all possible locations
    all_scenarios = set()
    for d in extract_to.rglob("Scenario-*"):
        if d.is_dir() and (d / "Labels.csv").exists():
            all_scenarios.add(d.name)
    
    print(f"Total scenarios with Labels.csv: {len(all_scenarios)}")
    print("DONE!")


if __name__ == "__main__":
    extract_hanoi_zip()
