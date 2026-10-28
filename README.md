# Soybean Segmentation Using Unsupervised Clustering and BioCLIP

This repository contains an experimental computer vision pipeline for identifying and segmenting soybean plants in field imagery. The project explores unsupervised clustering, image processing, and pretrained vision models to distinguish soybean vegetation from surrounding weeds, grass, and soil.

The work focuses on soybean quadrat images collected under field conditions, where overlapping vegetation, variable lighting, and similar plant colors make accurate segmentation challenging.

## Project Overview

The project includes two approaches to soybean segmentation:

**1. Unsupervised Color Clustering**

The initial approach uses MiniBatchKMeans to group pixels based on their color characteristics. Images are converted into CIELAB color space, and normalized excess green (ExG) is used to help distinguish vegetation from background material.

The pipeline includes:

- Manual selection of the quadrat boundaries
- Perspective correction and image preprocessing
- Feature extraction using color and greenness
- Unsupervised pixel clustering
- Cluster visualization and evaluation
- Optional assignment of clusters to soybean and non-soybean classes

Additional experiments explore shape-based filtering to improve separation between broadleaf soybean plants and narrow-leaf vegetation.

**2. BioCLIP-Based Segmentation**

The second approach incorporates a pretrained BioCLIP vision model to evaluate candidate soybean regions using learned visual features rather than relying entirely on pixel color.

This pipeline focuses on:

- Identifying candidate vegetation regions
- Evaluating candidate regions using BioCLIP
- Filtering and refining soybean predictions
- Generating segmentation masks and image overlays
- Saving region-level scores for further analysis

The objective is to investigate whether pretrained visual representations can improve soybean identification in images containing mixed vegetation.

## Repository Structure

```text
Bioclip_segmentation/
├── README.md
├── .gitignore
└── unsupervised_clustering/
    ├── soybean_clustering.py
    ├── soybean_clustering_instructions.md
    ├── bioclip_inference.py
    ├── bioclip_localize.py
    └── bioclip_segment.py
```

| File | Description |
|---|---|
| `soybean_clustering.py` | Unsupervised clustering and vegetation analysis |
| `bioclip_inference.py` | BioCLIP inference experiments |
| `bioclip_localize.py` | Localization of potential soybean regions |
| `bioclip_segment.py` | BioCLIP-based soybean segmentation pipeline |
| `soybean_clustering_instructions.md` | Detailed instructions for running and reviewing clustering results |

## Getting Started

### Requirements

The project is developed in Python and uses libraries for numerical computing, image processing, clustering, and deep learning.

For the unsupervised clustering pipeline, install the required packages:

```powershell
python -m pip install numpy pillow scipy scikit-learn matplotlib joblib
```

The BioCLIP pipeline additionally requires PyTorch and the dependencies for loading the pretrained model. A complete pinned dependency list has not yet been added.

### Unsupervised Clustering

Navigate to the project directory:

```powershell
cd unsupervised_clustering
```

Place the input images in a folder named `soybean_images`, then run:

```powershell
python soybean_clustering.py --input soybean_images --output soybean_results
```

For new images, the program prompts the user to select the four corners inside the PVC quadrat. The selected region is corrected for perspective and processed using the clustering pipeline.

Results are saved in `soybean_results`, including cluster visualizations, vegetation masks, and summary statistics.

Additional options for model reuse, cluster assignment, and parameter adjustment are documented in `soybean_clustering_instructions.md`.

### BioCLIP Segmentation

To process images using the BioCLIP segmentation pipeline:

```powershell
python bioclip_segment.py --input soybean_images --output bioclip_segmentation
```

The program processes each input image and saves the corresponding results in a separate folder.

Generated outputs include:

- `soybean_mask.png` — Predicted soybean segmentation mask
- `soybean_overlay.jpg` — Segmentation results overlaid on the original image
- `vegetation_mask.png` — Candidate vegetation mask
- `component_scores.csv` — Region-level scoring information

Additional intermediate masks and visualizations are saved for reviewing individual processing stages.

## Current Results and Limitations

Initial experiments show that unsupervised color clustering can identify vegetation regions, including soybean leaves. However, grass and other weeds frequently appear in the same clusters because of similarities in color and lighting.

The BioCLIP approach incorporates pretrained visual features to help distinguish candidate soybean regions. Challenges remain in images with overlapping leaves, dense weeds, and visually similar plant species.

Current limitations include:

- Difficulty separating soybean from nearby vegetation
- Sensitivity to differences in lighting and plant appearance
- Incomplete segmentation of overlapping soybean leaves
- Limited quantitative validation against manually annotated masks

The pipeline remains experimental. A systematic evaluation using labeled images is needed before reporting segmentation accuracy or comparing the approaches quantitatively.

## Future Work

Planned improvements include integrating additional spatial and shape information, refining candidate-region selection, and testing segmentation across a larger range of field conditions.

Future evaluation will focus on comparing predicted masks with manually annotated images using precision, recall, and intersection over union (IoU).

## References

- [BioCLIP — Imageomics](https://github.com/Imageomics/bioclip)
- [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn — MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
