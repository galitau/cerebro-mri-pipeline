# Cerebro MRI Pipeline

An end-to-end Python neuroimaging processing pipeline for cerebrovascular disease research.

This pipeline processes structural brain MRI scans (NIfTI format), performs automated skull-stripping and tissue segmentation, computes volumetric metrics, and generates quality-control reports for studying cerebrovascular diseases such as sickle cell disease and stroke in pediatric populations.

## Scientific Background

Cerebrovascular diseases, including sickle cell disease and pediatric stroke, require precise quantitative analysis of brain structure to understand disease progression and treatment efficacy. This pipeline provides automated, reproducible analysis of T1-weighted MRI scans to:

- **Extract brain tissue** through automated skull-stripping
- **Segment tissue classes** into Gray Matter (GM), White Matter (WM), and Cerebrospinal Fluid (CSF)
- **Compute volumetric metrics** including absolute volumes and tissue ratios
- **Generate quality control** visualizations to verify processing accuracy

These metrics are critical for:
- Monitoring brain development in pediatric patients
- Assessing tissue loss due to cerebrovascular events
- Comparing patient populations against healthy controls
- Longitudinal tracking of disease progression

## Architecture and Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     CEREBRO MRI PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. INPUT LOADING                                               │
│     - Load NIfTI file (.nii / .nii.gz)                          │
│     - Verify headers, dimensions, affine matrix                 │
│     - Extract voxel dimensions for volume calculations          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. SKULL-STRIPPING (Brain Extraction)                          │
│     - Method: Morphological operations (default) or Nilearn     │
│     - Intensity thresholding + 3D morphological ops             │
│     - Largest connected component extraction                    │
│     - Output: Binary brain mask                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. TISSUE SEGMENTATION                                         │
│     - Method: Gaussian Mixture Model (default) or Otsu          │
│     - Segment into: CSF, Gray Matter, White Matter              │
│     - 3-class classification based on intensity distribution    │
│     - Output: Segmentation map (0=BG, 1=CSF, 2=GM, 3=WM)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. QUANTITATIVE METRICS                                        │
│     - Voxel counts per tissue class                             │
│     - Absolute volumes (cm³) using voxel dimensions             │
│     - GM/WM and GM/CSF ratios                                   │
│     - Total brain volume                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. QUALITY CONTROL PLOTS                                       │
│     - Orthographic 3-plane views (axial, sagittal, coronal)     │
│     - Segmentation overlays with color coding                   │
│     - Visual verification of processing accuracy                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. OUTPUT GENERATION                                           │
│     - metrics.csv: Tabular format for analysis                  │
│     - metrics.json: Structured format for integration           │
│     - QC plot: PNG image for visual inspection                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cerebro-mri-pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `nibabel` - NIfTI file I/O
- `nilearn` - Neuroimaging data processing and plotting
- `numpy` - Numerical computations
- `scipy` - Scientific computing utilities
- `matplotlib` - Plotting and visualization
- `scikit-image` - Image processing algorithms
- `pandas` - Data manipulation and export
- `scikit-learn` - Machine learning (Gaussian Mixture Models)

## Quickstart

### 1. Download Sample Data

Download sample T1-weighted brain MRI scans for testing:

```bash
python scripts/download_sample_data.py
```

This downloads the MNI152 T1 template (1mm isotropic) to `data/sample_t1.nii.gz`.

**Options:**
```bash
python scripts/download_sample_data.py --output-dir data --dataset oasis
```

### 2. Run the Pipeline

Process a NIfTI file with the pipeline:

```bash
python pipeline.py --input data/sample_t1.nii.gz --output-dir outputs
```

**Options:**
```bash
python pipeline.py --input <path-to-nifti> --output-dir <output-directory> \
    --skull-strip-method morphological \
    --segmentation-method gmm
```

**Available methods:**
- `--skull-strip-method`: `morphological` (default) or `nilearn`
- `--segmentation-method`: `gmm` (default) or `otsu`

### 3. Review Outputs

The pipeline generates three outputs in the specified directory:

- `metrics.csv` - Tabular metrics
- `metrics.json` - Structured JSON metrics
- `<subject_id>_qc.png` - Quality control visualization

## Example Terminal Output

```
Loading NIfTI file: data\sample_t1.nii.gz
Data type: int16
Dimensions: (197, 233, 189)
Affine matrix:
[[   1.    0.    0.  -98.]
 [   0.    1.    0. -134.]
 [   0.    0.    1.  -72.]
 [   0.    0.    0.    1.]]
Pipeline initialized for: data\sample_t1.nii.gz
Output directory: outputs
Image shape: (197, 233, 189)
Voxel size: (1.0, 1.0, 1.0) mm
Voxel volume: 1.000 mm³
============================================================
CEREBRO MRI PIPELINE - Processing Start
============================================================

Performing skull-stripping using morphological method...
Brain mask shape: (197, 233, 189)
Brain voxels: 1651889
Brain volume: 1651.89 cm³

Performing tissue segmentation using gmm method...
Segmentation complete. Shape: (197, 233, 189)

Computing volumetric metrics...

--- Volumetric Metrics ---
Total Brain Volume: 1651.89 cm³
CSF Volume: 150.45 cm³ (150449 voxels)
Gray Matter Volume: 926.69 cm³ (926689 voxels)
White Matter Volume: 574.75 cm³ (574751 voxels)
GM/WM Ratio: 1.6123
GM/CSF Ratio: 6.1595

Generating QC plots...
QC plot saved to: outputs\sample_t1.nii_qc.png

Saving outputs...
Metrics saved to CSV: outputs\metrics.csv
Metrics saved to JSON: outputs\metrics.json

============================================================
CEREBRO MRI PIPELINE - Processing Complete
============================================================
```

## Output Metrics Explained

### Volumetric Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| `total_brain_volume_cm3` | Total brain tissue volume | cm³ |
| `csf_volume_cm3` | Cerebrospinal fluid volume | cm³ |
| `gm_volume_cm3` | Gray matter volume | cm³ |
| `wm_volume_cm3` | White matter volume | cm³ |
| `csf_voxels` | Number of CSF voxels | count |
| `gm_voxels` | Number of gray matter voxels | count |
| `wm_voxels` | Number of white matter voxels | count |
| `total_brain_voxels` | Total brain voxels | count |

### Ratio Metrics

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| `gm_wm_ratio` | Gray matter to white matter ratio | Indicates myelination and maturation; altered in neurodevelopmental disorders |
| `gm_csf_ratio` | Gray matter to CSF ratio | Reflects cortical atrophy; decreased in neurodegeneration and stroke |

### Metadata

- `voxel_size_mm`: Voxel dimensions in millimeters [x, y, z]
- `image_shape`: 3D image dimensions [x, y, z]
- `subject_id`: Identifier derived from input filename

## Project Structure

```
cerebro-mri-pipeline/
├── data/                      # Input NIfTI files
│   └── sample_t1.nii.gz      # Sample data (downloaded)
├── scripts/
│   └── download_sample_data.py # Data download script
├── outputs/                   # Pipeline outputs
│   ├── metrics.csv           # Tabular metrics
│   ├── metrics.json          # JSON metrics
│   └── sample_t1.nii_qc.png  # QC visualization
├── pipeline.py                # Main pipeline script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Pipeline Methods

### Skull-Stripping Methods

**Morphological (default):**
- Pure Python implementation using intensity thresholding
- 3D morphological operations (closing, opening)
- Largest connected component extraction
- No external dependencies beyond Python packages

**Nilearn:**
- Uses `nilearn.masking.compute_brain_mask()`
- Intensity-based brain extraction
- May provide more robust results for certain datasets

### Segmentation Methods

**Gaussian Mixture Model (default):**
- Fits 3-component GMM to brain voxel intensities
- Automatically orders components by mean intensity
- More flexible for non-normal intensity distributions
- Computationally intensive but accurate

**Multi-Otsu:**
- Multi-threshold Otsu segmentation
- Faster computation
- Assumes roughly equal class proportions
- May be less accurate for pathological cases

## Clinical Applications

This pipeline is designed for research in:

- **Sickle Cell Disease:** Monitor cerebrovascular complications and silent cerebral infarcts
- **Pediatric Stroke:** Quantify tissue loss and track recovery
- **Neurodevelopment:** Study brain maturation in at-risk populations
- **Treatment Monitoring:** Assess efficacy of interventions over time

## Limitations and Considerations

- The pipeline is optimized for T1-weighted structural MRI
- Skull-stripping accuracy may vary with scan quality and pathology
- Segmentation assumes standard 3-class tissue distribution
- Pediatric brains may require age-specific templates for optimal results
- Always review QC plots before using metrics in analysis

## Future Enhancements

Potential improvements for future versions:

- Support for T2-weighted and FLAIR sequences
- Age-specific segmentation templates for pediatric populations
- Longitudinal processing pipelines
- Integration with FSL/ANTs for advanced registration
- Batch processing for multi-subject studies
- Statistical analysis tools for group comparisons

## Citation

If you use this pipeline in your research, please cite:

```
Cerebro MRI Pipeline: An end-to-end neuroimaging processing pipeline
for cerebrovascular disease research at the Hospital for Sick Children
/ University of Toronto.
```

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contact

For questions or issues related to this pipeline, please contact:
[Your contact information]

## Acknowledgments

- Hospital for Sick Children (SickKids)
- University of Toronto
- Nilearn and Nibabel development communities
- Open neuroimaging data initiatives (OASIS, ICBM)
