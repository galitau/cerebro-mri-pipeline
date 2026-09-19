#!/usr/bin/env python3
"""
Download sample T1-weighted brain MRI data for testing the cerebro-mri-pipeline.
This script uses nilearn to fetch open-access, anonymized brain MRI scans.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import datasets


def download_oasis_sample(output_dir="data"):
    """
    Download sample T1-weighted brain MRI from the OASIS dataset.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the downloaded data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample T1-weighted brain MRI from OASIS dataset...")
    print("This may take a few minutes on first download...")
    
    # Fetch OASIS VBM nifti data (small subset for testing)
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=2, data_dir=str(output_dir / "nilearn_data"))
    
    # Copy the first subject's T1 image to the main data directory
    # OASIS VBM provides gray matter maps, but we can use the raw data
    # Let's use MNI152 template as a fallback sample
    print("Downloading MNI152 T1 template as sample data...")
    mni152 = datasets.fetch_icbm152_2009(data_dir=str(output_dir / "nilearn_data"))
    
    # Copy the T1-weighted image
    t1_path = mni152.t1
    output_path = output_dir / "sample_t1.nii.gz"
    
    shutil.copy(t1_path, output_path)
    print(f"Sample T1-weighted MRI saved to: {output_path}")
    
    # Verify the file
    img = nib.load(output_path)
    data = img.get_fdata()
    print(f"Image shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{data.min():.2f}, {data.max():.2f}]")
    
    return output_path


def download_penn_sample(output_dir="data"):
    """
    Download sample data from the Penn Adrc dataset as alternative.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading sample T1-weighted brain MRI from Penn ADRC dataset...")
    
    # Fetch a single subject from the Penn ADRC dataset
    penn_dataset = datasets.fetch_development_fmri(n_subjects=1, data_dir=str(output_dir / "nilearn_data"))
    
    # Use the anatomical data if available
    if hasattr(penn_dataset, 'anat'):
        anat_path = penn_dataset.anat[0]
        output_path = output_dir / "sample_t1.nii.gz"
        shutil.copy(anat_path, output_path)
        print(f"Sample T1-weighted MRI saved to: {output_path}")
        return output_path
    else:
        print("Penn dataset anatomical data not available, using MNI152 template instead...")
        return download_oasis_sample(output_dir)


def main():
    """Main function to download sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample brain MRI data for testing")
    parser.add_argument("--output-dir", default="data", help="Output directory for downloaded data")
    parser.add_argument("--dataset", choices=["oasis", "penn"], default="oasis",
                        help="Dataset to download from")
    
    args = parser.parse_args()
    
    if args.dataset == "oasis":
        download_oasis_sample(args.output_dir)
    else:
        download_penn_sample(args.output_dir)
    
    print("\nSample data download complete!")
    print(f"Data saved to: {args.output_dir}/")
    print("You can now run the pipeline with:")
    print(f"  python pipeline.py --input {args.output_dir}/sample_t1.nii.gz --output-dir outputs/")


if __name__ == "__main__":
    main()
