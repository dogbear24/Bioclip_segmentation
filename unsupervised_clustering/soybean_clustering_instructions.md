# Clustering soybean quadrat images

The supplied photo was processed inside the PVC square only. This is a runnable
baseline for **soybean present / no soybean** research. It fits an unsupervised
color model; it does not yet reliably distinguish soybean from grass or other
green plants.

## What the first run showed

The model finds the broadleaf row on the right, but grass at the left and bottom
also enters the plant clusters. Consequently, the highlighted output is labeled
**vegetation candidates**, not confirmed soybean. No accuracy estimate is
available because no ground-truth soybean mask was supplied.

The model uses six intermediate color clusters rather than forcing two groups.
Different soil, debris, leaf colors and shadows can form separate groups. These
groups can later be interpreted as two final classes, **only if visual review
shows the groups separate soybean from everything else**. Naming a cluster is
human interpretation after clustering; it is not supervised model fitting.

| Final class | Meaning |
| --- | --- |
| 0 | No soybean: soil, dead material, grass, other weeds |
| 1 | Soybean pixels |
| 255 | Outside the selected quadrat; ignore in original-size masks |

## Run it on Windows

1. Save `soybean_clustering.py` in your **Downloads** folder.
2. Create a folder named `soybean_images` in Downloads. Put your image files in it.
   Start with a few representative photos. Keep filenames unique.
3. Open PowerShell and run these commands **one at a time**:

```powershell
cd "$env:USERPROFILE\Downloads"
python -m pip install numpy pillow scipy scikit-learn matplotlib joblib
python .\soybean_clustering.py --input .\soybean_images --output .\soybean_results
```

If you already use a Python environment for your research, run the commands in
that environment. No GPU is needed.

For each photo, a window opens. Click four points just **inside** the pipe in
this order:

1. Top-left.
2. Top-right.
3. Bottom-right.
4. Bottom-left.

Exclude the white PVC, including any fitting that protrudes inward. The crop is
straightened to an 800 x 800 square before feature extraction. Coordinates are
saved per image, so rerunning the same photos does not require clicking again.
This is manual quadrat selection, not automatic PVC detection. Select new
corners for each new camera view.

If the plot window does not open, run from a normal local PowerShell terminal
using a Python installation with a desktop plotting backend. A headless server
needs a `--corners-json` file instead of mouse input.

## Inspect the results before assigning species

Open `soybean_results\images`. Each input filename has its own results folder.

| Output | What it contains |
| --- | --- |
| `selected_roi.png` | Original photo with the chosen interior boundary outlined |
| `roi.png` | Straightened quadrat crop; outside areas are excluded |
| `summary.png` | Crop, all clusters, and highlighted vegetation candidates |
| `cluster_review.png` | One panel per cluster, showing its original image pixels |
| `cluster_ids.png` | Raw cluster IDs 0 through K-1, not semantic class IDs |
| `vegetation_candidates.png` | 255 for candidate vegetation, 0 otherwise; can include weeds |
| `cluster_stats.csv` | Size and mean color/greenness of each cluster |
| `tiles.csv` | Candidate percentages in an 8 x 8 grid; row/column numbering starts at 0 |
| `soybean_results\results.csv` | One summary row per image |

Look at every plant-related cluster across multiple images. Ask: **Does this
cluster include both soybean leaves and grass or other weeds?** If yes, it
cannot be assigned a pure soybean label. That mixing occurred in the supplied
photo. Its green cluster IDs are not a validated soybean mapping.

The green mask uses an adjustable-in-code heuristic: cluster-center normalized
excess green >= 0.10. This heuristic interprets green color, not soybean species.
It can miss yellow or injured leaves and include green non-soybean plants.

## Produce the final two-class masks after review

If the clusters genuinely separate soybean from background and weeds, provide
the IDs that contain soybean. The following is an **example command only**;
`2 4` are placeholders, not recommended labels for the supplied photo:

```powershell
python .\soybean_clustering.py --input .\soybean_images --output .\soybean_results --soybean-clusters 2 4
```

Every selected cluster is assigned class 1. Every other cluster becomes class 0.
The interpretation is saved with the model and reused on subsequent images.
Review its behavior on new conditions; a saved mapping does not make it correct.

This adds:

| Output | Meaning |
| --- | --- |
| `class_mask.png` | Rectified crop: 0 = no soybean, 1 = user-mapped soybean |
| `class_mask_original.png` | Original image size: 0/1 inside, 255 outside the quadrat |
| Additional `tiles.csv` fields | Soybean candidate percentage and a present/absent flag for each tile |
| Additional `results.csv` fields | Soybean candidate percentage and a present/absent flag for the full quadrat |

Presence defaults to at least **1% user-mapped soybean pixels** in the tile or
quadrat. This is an unvalidated starting threshold, not a biological detection
standard. Adjust it for the smallest plant you need to detect and test it:

```powershell
python .\soybean_clustering.py --input .\soybean_images --output .\soybean_results --presence-percent 0.5
```

Masks containing raw 0/1 class IDs may look almost black in a photo viewer.
Use `summary.png` for a visible overlay. Pixel fractions estimate projected
canopy cover in the selected crop, not true leaf area or plant counts.

## Reuse one model across images

The first run pools an equal number of random pixels per image, from up to 100
images, to fit one shared model. Later runs with the **same output folder**
reuse `kmeans_model.joblib`; this keeps cluster IDs consistent.

Adding new photos to the input folder and rerunning the original command uses
the existing model. Only new photos need their corners selected. Load only model
files you created yourself.

To fit a new model on a broader image set, increase the training sample and
explicitly refit:

```powershell
python .\soybean_clustering.py --input .\soybean_images --output .\soybean_results --refit --clusters 8 --samples-per-image 10000
```

Refitting can change cluster IDs. It clears the prior species mapping; review
the new clusters before assigning soybean IDs again. Cluster counts such as
6 or 8 are starting settings, not an optimal number established from this photo.

## If soybean and weeds remain mixed

Changing cluster names cannot separate pixels already in the same cluster.
Changing the cluster count alone may not solve the problem either. This baseline
has **color and greenness only**, so it has no explicit leaf-shape knowledge.

The next unsupervised experiment should cluster **image-patch features that
include shape and context**, rather than individual pixel colors. Learned
self-supervised patch embeddings are one option. Their clusters still need
semantic review and validation. If a reliable species detector is the main
goal, a small labeled dataset for soybean / non-soybean segmentation is another
option, with predictions kept strictly inside the quadrat.

Validate on separate photos covering weeds, small soybean plants, bare ground,
shadows and damaged leaves. Hand-label representative areas and compare against
the output masks using precision, recall and IoU. Keep validation photos out of
the initial fit, then process them with the saved model. For whole-quadrat
present/absent outputs, also count false positives on photos with no soybean and
false negatives on photos with soybean. Unsupervised fitting does not eliminate
the need for labels to measure species detection accuracy.

## Method and references

For each crop pixel the script extracts Lab L/a/b color plus normalized excess
green, `2g - r - b`, where r/g/b are RGB divided by R+G+B. Features are standardized
using the pooled training sample, weighted `[0.30, 1.00, 0.65, 1.20]`, then
clustered with MiniBatchKMeans (`random_state=42`, `n_init=10`). Coordinates are
not features, so the model does not treat the right side of an image as soybean
merely because the sample row happens to be there.

- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
- [MiniBatchKMeans reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [Pillow image transformations](https://pillow.readthedocs.io/en/stable/reference/Image.html)

The published example was visually inspected for crop boundaries and cluster
mixing. Model reuse and the mask encoding/projection were checked programmatically.
These are software checks, not a measurement of soybean detection accuracy.
