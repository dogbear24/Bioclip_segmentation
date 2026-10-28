# Soybean Segmentation Using Unsupervised Clustering and BioCLIP

This repository contains code I've been working on for identifying soybean plants in field images. The main goal is to separate soybean from surrounding weeds, grass, and soil using image processing and pretrained vision models.

I started with unsupervised color clustering and later experimented with shape-based filtering and BioCLIP to improve how soybean regions are identified.

## Approach

There are two main approaches in the project.

**Unsupervised clustering**

The initial method uses K-means clustering to group pixels based on color. Additional filtering is used to identify vegetation and distinguish likely soybean regions from grass and other plants based on shape.

**BioCLIP segmentation**

The second method uses a pretrained BioCLIP model to help identify soybean regions. It combines image processing with model-based scoring to generate segmentation masks.

Neither approach requires training a segmentation model from scratch.

## Files

- `soybean_clustering.py` - Unsupervised clustering and shape-based filtering
- `bioclip_inference.py` - BioCLIP inference experiments
- `bioclip_localize.py` - Locating potential soybean regions
- `bioclip_segment.py` - Generating soybean segmentation masks
- `soybean_clustering_instructions.md` - Additional instructions for the clustering script

## Running the Code

The scripts use Python. The BioCLIP approach requires the model and its dependencies to be installed. Dependencies are not pinned in this repository yet.

To run BioCLIP segmentation, navigate to the project folder:

    cd unsupervised_clustering

Place the input images in a folder named `soybean_images`, then run:

    python bioclip_segment.py --input soybean_images --output bioclip_segmentation

The script processes the input images and creates a separate output folder for each one.

Outputs include:

- `soybean_mask.png` - Binary segmentation mask
- `soybean_overlay.jpg` - Predicted soybean regions overlaid on the original image
- `component_scores.csv` - Scores for candidate regions

The original field images and generated outputs are not included in this repository.

## Current Limitations

The biggest challenge is distinguishing soybean leaves from nearby weeds, especially when plants overlap or have similar colors.

Color clustering alone can confuse soybean with other vegetation. BioCLIP helps incorporate additional visual information, but segmentation is still imperfect, particularly in crowded areas with grass.

This is an experimental research project, and the resulting masks have not been validated as ground-truth segmentation.

## References

BioCLIP: https://github.com/Imageomics/bioclip
