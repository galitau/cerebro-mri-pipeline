#!/usr/bin/env python3
"""
Cerebro MRI Pipeline - End-to-end neuroimaging processing pipeline for cerebrovascular disease research.

This pipeline processes structural brain MRI scans (NIfTI format), performs automated skull-stripping
and tissue segmentation, computes volumetric metrics, and generates quality-control reports.

Designed for pediatric neuroimaging research at the Hospital for Sick Children / University of Toronto.
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import filters, morphology
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from nilearn import plotting, masking
import pandas as pd


class MRIPipeline:
    """Main MRI processing pipeline class."""
    
    def __init__(self, input_path, output_dir="outputs"):
        """
        Initialize the MRI pipeline.
        
        Parameters
        ----------
        input_path : str
            Path to input NIfTI file (.nii or .nii.gz)
        output_dir : str
            Directory for output files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the image
        self.img = self.load_nifti()
        self.data = self.img.get_fdata()
        self.affine = self.img.affine
        self.header = self.img.header
        
        # Voxel dimensions
        self.voxel_size = self.header.get_zooms()[:3]
        self.voxel_volume_mm3 = np.prod(self.voxel_size)
        
        # Storage for intermediate results
        self.brain_mask = None
        self.segmentation = None
        self.metrics = {}

        
        print(f"Pipeline initialized for: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Image shape: {self.data.shape}")
        print(f"Voxel size: {self.voxel_size} mm")
        print(f"Voxel volume: {self.voxel_volume_mm3:.3f} mm³")
    
    def load_nifti(self):
        """
        Load NIfTI volume and verify headers, dimensions, and affine matrix.
        
        Returns
        -------
        nibabel.Nifti1Image
            Loaded NIfTI image object
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        print(f"Loading NIfTI file: {self.input_path}")
        img = nib.load(self.input_path)
        
        # Verify header
        print(f"Data type: {img.header.get_data_dtype()}")
        print(f"Dimensions: {img.shape}")
        print(f"Affine matrix:\n{img.affine}")
        
        # Check if affine is valid (non-singular)
        if np.linalg.det(img.affine) == 0:
            raise ValueError("Invalid affine matrix (singular)")
        
        return img
    
    def skull_strip(self, method="morphological"):
        """
        Perform brain extraction (skull-stripping).
        
        Parameters
        ----------
        method : str
            Method to use: 'morphological' (default) or 'nilearn'
        
        Returns
        -------
        numpy.ndarray
            Binary brain mask
        """
        print(f"\nPerforming skull-stripping using {method} method...")
        
        if method == "nilearn":
            # Use nilearn's brain mask computation
            self.brain_mask = masking.compute_brain_mask(self.img, threshold=0.5)
        else:
            # Use morphological operations (pure Python)
            self.brain_mask = self._morphological_skull_strip()
        
        # Apply mask to data
        brain_data = self.data * self.brain_mask
        
        print(f"Brain mask shape: {self.brain_mask.shape}")
        print(f"Brain voxels: {np.sum(self.brain_mask)}")
        print(f"Brain volume: {np.sum(self.brain_mask) * self.voxel_volume_mm3 / 1000:.2f} cm³")
        
        return self.brain_mask
    
    def _morphological_skull_strip(self):
        """
        Perform skull-stripping using intensity thresholding and morphological operations.
        
        Returns
        -------
        numpy.ndarray
            Binary brain mask
        """
        data = self.data
        
        # Normalize intensity
        data_norm = (data - data.min()) / (data.max() - data.min())
        
        # Initial threshold (Otsu)
        threshold = filters.threshold_otsu(data_norm)
        binary = data_norm > threshold
        
        # 3D morphological operations
        # Remove small objects
        binary = morphology.remove_small_objects(binary, max_size=1000)
        
        # Fill holes
        binary = ndimage.binary_fill_holes(binary)
        
        # Morphological closing to smooth
        selem = morphology.ball(3)
        binary = morphology.closing(binary, selem)
        
        binary = morphology.opening(binary, morphology.ball(2))
        
        # Find largest connected component (the brain)
        labeled, num_features = ndimage.label(binary)
        if num_features > 0:
            sizes = ndimage.sum(binary, labeled, range(num_features + 1))
            max_label = sizes.argmax()
            binary = labeled == max_label
        
        return binary.astype(np.float64)
    
    def tissue_segmentation(self, method="gmm"):
        """
        Segment brain volume into Gray Matter (GM), White Matter (WM), and CSF.
        
        Parameters
        ----------
        method : str
            Segmentation method: 'gmm' (Gaussian Mixture Model) or 'otsu'
        
        Returns
        -------
        numpy.ndarray
            Segmentation map (0=background, 1=CSF, 2=GM, 3=WM)
        """
        print(f"\nPerforming tissue segmentation using {method} method...")
        
        if self.brain_mask is None:
            raise RuntimeError("Skull-stripping must be performed before segmentation")
        
        # Extract brain data
        brain_data = self.data[self.brain_mask > 0]
        
        if method == "gmm":
            self.segmentation = self._gmm_segmentation(brain_data)
        else:
            self.segmentation = self._otsu_segmentation(brain_data)
        
        print(f"Segmentation complete. Shape: {self.segmentation.shape}")
        
        return self.segmentation
    
    def _gmm_segmentation(self, brain_data):
        """
        Segment using Gaussian Mixture Model.
        
        Parameters
        ----------
        brain_data : numpy.ndarray
            Flattened brain voxel intensities
        
        Returns
        -------
        numpy.ndarray
            Full 3D segmentation map
        """
        # Reshape for GMM
        data_reshaped = brain_data.reshape(-1, 1)
        
        # Fit 3-component GMM (CSF, GM, WM)
        gmm = GaussianMixture(n_components=3, random_state=42, max_iter=200)
        labels = gmm.fit_predict(data_reshaped)
        
        # Order components by mean intensity (CSF < GM < WM)
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        
        # Relabel: 0=CSF, 1=GM, 2=WM
        relabeled = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indices):
            relabeled[labels == idx] = i + 1  # 1=CSF, 2=GM, 3=WM
        
        # Create full 3D segmentation
        segmentation = np.zeros(self.data.shape, dtype=np.int32)
        brain_indices = np.where(self.brain_mask > 0)
        segmentation[brain_indices] = relabeled
        
        return segmentation
    
    def _otsu_segmentation(self, brain_data):
        """
        Segment using multi-Otsu thresholding.
        
        Parameters
        ----------
        brain_data : numpy.ndarray
            Flattened brain voxel intensities
        
        Returns
        -------
        numpy.ndarray
            Full 3D segmentation map
        """
        # Normalize
        brain_data_norm = (brain_data - brain_data.min()) / (brain_data.max() - brain_data.min())
        
        # Multi-Otsu thresholding (3 classes)
        thresholds = filters.threshold_multiotsu(brain_data_norm, classes=3)
        
        # Create segmentation
        segmentation = np.zeros(self.data.shape, dtype=np.int32)
        brain_indices = np.where(self.brain_mask > 0)
        
        # Assign labels based on thresholds
        brain_voxels = brain_data_norm
        labels = np.zeros_like(brain_voxels, dtype=np.int32)
        labels[brain_voxels < thresholds[0]] = 1  # CSF
        labels[(brain_voxels >= thresholds[0]) & (brain_voxels < thresholds[1])] = 2  # GM
        labels[brain_voxels >= thresholds[1]] = 3  # WM
        
        segmentation[brain_indices] = labels
        
        return segmentation
    
    def compute_metrics(self):
        """
        Compute quantitative volumetric metrics.
        
        Returns
        -------
        dict
            Dictionary containing all computed metrics
        """
        print("\nComputing volumetric metrics...")
        
        if self.segmentation is None:
            raise RuntimeError("Segmentation must be performed before computing metrics")
        
        # Count voxels for each tissue type
        csf_voxels = np.sum(self.segmentation == 1)
        gm_voxels = np.sum(self.segmentation == 2)
        wm_voxels = np.sum(self.segmentation == 3)
        total_brain_voxels = np.sum(self.brain_mask)
        
        # Convert to volumes (cm³ = mm³ / 1000)
        voxel_volume_cm3 = self.voxel_volume_mm3 / 1000
        
        csf_volume = csf_voxels * voxel_volume_cm3
        gm_volume = gm_voxels * voxel_volume_cm3
        wm_volume = wm_voxels * voxel_volume_cm3
        total_brain_volume = total_brain_voxels * voxel_volume_cm3
        
        # Compute ratios
        gm_wm_ratio = gm_volume / wm_volume if wm_volume > 0 else 0
        gm_csf_ratio = gm_volume / csf_volume if csf_volume > 0 else 0
        
        # Store metrics
        self.metrics = {
            "subject_id": self.input_path.stem,
            "total_brain_volume_cm3": round(total_brain_volume, 3),
            "csf_volume_cm3": round(csf_volume, 3),
            "gm_volume_cm3": round(gm_volume, 3),
            "wm_volume_cm3": round(wm_volume, 3),
            "csf_voxels": int(csf_voxels),
            "gm_voxels": int(gm_voxels),
            "wm_voxels": int(wm_voxels),
            "total_brain_voxels": int(total_brain_voxels),
            "gm_wm_ratio": round(gm_wm_ratio, 4),
            "gm_csf_ratio": round(gm_csf_ratio, 4),
            "voxel_size_mm": [float(v) for v in self.voxel_size],
            "image_shape": list(self.data.shape)
        }
        
        # Print metrics
        print("\n--- Volumetric Metrics ---")
        print(f"Total Brain Volume: {total_brain_volume:.2f} cm³")
        print(f"CSF Volume: {csf_volume:.2f} cm³ ({csf_voxels} voxels)")
        print(f"Gray Matter Volume: {gm_volume:.2f} cm³ ({gm_voxels} voxels)")
        print(f"White Matter Volume: {wm_volume:.2f} cm³ ({wm_voxels} voxels)")
        print(f"GM/WM Ratio: {gm_wm_ratio:.4f}")
        print(f"GM/CSF Ratio: {gm_csf_ratio:.4f}")
        
        return self.metrics
    
    def generate_qc_plots(self):
        """
        Generate quality control plots showing orthographic views with segmentation overlays.
        """
        print("\nGenerating QC plots...")
        
        if self.segmentation is None:
            raise RuntimeError("Segmentation must be performed before generating QC plots")
        
        # Create figure with 3 views
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get middle slices
        x_mid = self.data.shape[0] // 2
        y_mid = self.data.shape[1] // 2
        z_mid = self.data.shape[2] // 2
        
        # Create segmentation overlay
        segmentation_rgba = np.zeros((*self.segmentation.shape, 4))
        segmentation_rgba[self.segmentation == 1] = [0, 0, 1, 0.3]  # CSF: blue
        segmentation_rgba[self.segmentation == 2] = [1, 0, 0, 0.3]  # GM: red
        segmentation_rgba[self.segmentation == 3] = [0, 1, 0, 0.3]  # WM: green
        
        # Sagittal view
        axes[0].imshow(self.data[x_mid, :, :], cmap='gray', origin='lower')
        axes[0].imshow(segmentation_rgba[x_mid, :, :], origin='lower')
        axes[0].set_title('Sagittal (x={})'.format(x_mid))
        axes[0].axis('off')
        
        # Coronal view
        axes[1].imshow(self.data[:, y_mid, :], cmap='gray', origin='lower')
        axes[1].imshow(segmentation_rgba[:, y_mid, :], origin='lower')
        axes[1].set_title('Coronal (y={})'.format(y_mid))
        axes[1].axis('off')
        
        # Axial view
        axes[2].imshow(self.data[:, :, z_mid], cmap='gray', origin='lower')
        axes[2].imshow(segmentation_rgba[:, :, z_mid], origin='lower')
        axes[2].set_title('Axial (z={})'.format(z_mid))
        axes[2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.5, label='CSF'),
            Patch(facecolor='red', alpha=0.5, label='Gray Matter'),
            Patch(facecolor='green', alpha=0.5, label='White Matter')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
        
        plt.suptitle(f'QC Report: {self.input_path.stem}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        qc_path = self.output_dir / f"{self.input_path.stem}_qc.png"
        plt.savefig(qc_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"QC plot saved to: {qc_path}")
        
        return qc_path
    
    def save_outputs(self):
        """
        Save metrics to CSV and JSON files.
        """
        print("\nSaving outputs...")
        
        if not self.metrics:
            raise RuntimeError("Metrics must be computed before saving outputs")
        
        # Save to CSV
        csv_path = self.output_dir / "metrics.csv"
        df = pd.DataFrame([self.metrics])
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to CSV: {csv_path}")
        
        # Save to JSON
        json_path = self.output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to JSON: {json_path}")
        
        return csv_path, json_path
    
    def run(self, skull_strip_method="morphological", segmentation_method="gmm"):
        """
        Run the complete pipeline.
        
        Parameters
        ----------
        skull_strip_method : str
            Method for skull-stripping
        segmentation_method : str
            Method for tissue segmentation
        """
        print("=" * 60)
        print("CEREBRO MRI PIPELINE - Processing Start")
        print("=" * 60)
        
        # Step 1: Skull-stripping
        self.skull_strip(method=skull_strip_method)
        
        # Step 2: Tissue segmentation
        self.tissue_segmentation(method=segmentation_method)
        
        # Step 3: Compute metrics
        self.compute_metrics()
        
        # Step 4: Generate QC plots
        self.generate_qc_plots()
        
        # Step 5: Save outputs
        self.save_outputs()
        
        print("\n" + "=" * 60)
        print("CEREBRO MRI PIPELINE - Processing Complete")
        print("=" * 60)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Cerebro MRI Pipeline - End-to-end neuroimaging processing for cerebrovascular disease research"
    )
    parser.add_argument("--input", required=True, help="Path to input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--skull-strip-method", choices=["morphological", "nilearn"], default="morphological",
                        help="Method for skull-stripping")
    parser.add_argument("--segmentation-method", choices=["gmm", "otsu"], default="gmm",
                        help="Method for tissue segmentation")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = MRIPipeline(args.input, args.output_dir)
    pipeline.run(
        skull_strip_method=args.skull_strip_method,
        segmentation_method=args.segmentation_method
    )


if __name__ == "__main__":
    main()
