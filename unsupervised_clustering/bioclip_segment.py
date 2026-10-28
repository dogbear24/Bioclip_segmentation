"""Hybrid soybean mask: MiniBatchKMeans vegetation + shape filtering + BioCLIP 2.

BioCLIP predicts a label for each proposed region, not a pixel-wise segmentation.
The mask boundaries come from vegetation clustering and morphology.
"""

from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import hashlib
import os
import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

MODEL_NAME = "hf-hub:imageomics/bioclip-2"
LABELS = [
    "a soybean plant",
    "a grass weed",
    "a broadleaf weed",
    "a small seedling",
    "bare soil without plants",
]


def image_folder_names(images, input_path):
    """Name one result folder per image without overwriting duplicate stems."""
    root = input_path if input_path.is_dir() else input_path.parent
    counts = Counter(image.stem.casefold() for image in images)
    names = []
    for image in images:
        name = image.stem
        if counts[name.casefold()] > 1 or len(name) > 120:
            relative = image.relative_to(root).as_posix()
            suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
            name = f"{name[:100]}_{suffix}"
        names.append(name)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Image names produce duplicate output folders")
    return names


def load_bioclip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()
    tokens = tokenizer([f"a photo of {label}" for label in LABELS]).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return model, preprocess, text_features


def compute_exg(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    return 2 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]


def make_vegetation_mask(rgb, n_clusters=6, sample_limit=50000):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    exg = compute_exg(rgb)
    h, w = rgb.shape[:2]
    features = np.dstack([
        lab[:, :, 0] / 255.0,
        (lab[:, :, 1] - 128.0) / 128.0,
        (lab[:, :, 2] - 128.0) / 128.0,
        exg,
    ]).reshape(-1, 4)

    if len(features) > sample_limit:
        rng = np.random.default_rng(42)
        sample = features[rng.choice(len(features), sample_limit, replace=False)]
    else:
        sample = features

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, n_init=10, random_state=42, batch_size=4096
    )
    kmeans.fit(sample)
    cluster_labels = kmeans.predict(features).reshape(h, w)
    vegetation_mask = np.zeros((h, w), dtype=np.uint8)

    for cluster_id in range(n_clusters):
        pixels = cluster_labels == cluster_id
        if not np.any(pixels):
            continue
        mean_exg = float(exg[pixels].mean())
        mean_a = float(lab[:, :, 1][pixels].mean())
        if mean_exg > 0.05 and mean_a < 135:
            vegetation_mask[pixels] = 255

    # K-Means alone can call tan/brown soil a vegetation cluster.
    # Require each retained pixel to be visibly green as well.
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    green_pixel = (
        (g > r + 3) & (g > b + 3) &
        (exg > 0.08) & (lab[:, :, 1] < 125)
    )
    vegetation_mask[~green_pixel] = 0

    # Opening removes isolated noise. Do NOT close with 5x5 here:
    # closing tends to join soybean leaves to neighboring grass.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)


def split_vegetation_regions(vegetation_mask, opening_size=7,
                             min_distance=12):
    """Return watershed labels and the actually filtered candidate mask.

    Crucially, the watershed runs on the opened mask, not the original mask;
    otherwise removed grass blades get added back during region growth.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    candidate_mask = cv2.morphologyEx(
        vegetation_mask, cv2.MORPH_OPEN, kernel
    )
    binary = candidate_mask > 0
    if not binary.any():
        return np.zeros(vegetation_mask.shape, dtype=np.int32), candidate_mask

    distance = distance_transform_edt(binary)
    peaks = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=2,
        labels=binary.astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros(binary.shape, dtype=np.int32)
    for marker_id, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = marker_id

    # Every isolated vegetation component must get at least one marker.
    n, connected = cv2.connectedComponents(candidate_mask, connectivity=8)
    next_marker = len(peaks) + 1
    for component_id in range(1, n):
        region = connected == component_id
        if np.any(markers[region]):
            continue
        masked_distance = np.where(region, distance, -1)
        y, x = np.unravel_index(masked_distance.argmax(), distance.shape)
        markers[y, x] = next_marker
        next_marker += 1

    labels = watershed(
        -distance, markers, mask=binary, watershed_line=True
    )
    return labels.astype(np.int32), candidate_mask


def classify_crop(crop_rgb, model, preprocess, text_features, device):
    image_tensor = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        similarities = image_features @ text_features.T
        scores = (100 * similarities).softmax(dim=-1)
    return scores.squeeze(0).cpu().numpy()


def save_image(output_path, image, params=None):
    """Raise an explicit error if OpenCV cannot write an image."""
    ok = cv2.imwrite(str(output_path), image, params or [])
    if not ok:
        raise OSError(f"OpenCV failed to save image: {output_path}")


def save_mask_overlay(rgb, mask, output_path, color_bgr, opacity=0.40):
    """Tint only selected pixels while retaining the original RGB photograph."""
    original_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = original_bgr.copy()
    selected = mask > 0
    if np.any(selected):
        color = np.array(color_bgr, dtype=np.float32)
        result[selected] = np.clip(
            (1.0 - opacity) * original_bgr[selected].astype(np.float32)
            + opacity * color,
            0, 255
        ).astype(np.uint8)
        # Show pixel-mask boundaries without covering the underlying image.
        contours, _ = cv2.findContours(
            (selected.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color_bgr, 1)
    save_image(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])


def refine_accepted_region(component_mask, min_leaf_half_width=6.0,
                           max_thin_extension=18.0, min_broad_fraction=0.10):
    """Remove long, thin extensions from BioCLIP-accepted regions.

    K-Means/connected components may join grass to a soybean. BioCLIP
    classifies the *whole* crop, so we additionally require a broad-leaf
    interior and limit how far slender pixels can extend from that interior.

    Returns (refined_bool_mask, broad_fraction). These are heuristics, not
    validated soybean/grass labels.
    """
    ys, xs = np.where(component_mask)
    if len(xs) == 0:
        return np.zeros_like(component_mask, dtype=bool), 0.0

    # Compute locally; adding a zero border keeps distance estimates correct
    # even when a component touches its bounding-box edge.
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = component_mask[y0:y1, x0:x1].astype(np.uint8)
    padded = cv2.copyMakeBorder(patch, 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=0)
    thickness = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    core = (thickness >= min_leaf_half_width) & (patch > 0)
    broad_fraction = float(core.sum() / max(int(patch.sum()), 1))

    refined = np.zeros_like(component_mask, dtype=bool)
    if not core.any() or broad_fraction < min_broad_fraction:
        return refined, broad_fraction

    # Recover edges and tips near thick leaf interiors, but do not regrow
    # a long grass blade that is attached by a narrow bridge.
    distance_to_core = cv2.distanceTransform(
        (~core).astype(np.uint8), cv2.DIST_L2, 5
    )
    kept = (patch > 0) & (distance_to_core <= max_thin_extension)
    refined[y0:y1, x0:x1] = kept
    return refined, broad_fraction


def segment_image(image_path, model, preprocess, text_features, device,
                  output_dir, min_area=120, padding=12,
                  soybean_threshold=0.60, margin=0.12,
                  opening_size=5, min_width=5.0, min_distance=12,
                  min_leaf_half_width=6.0, max_thin_extension=18.0,
                  min_broad_fraction=0.10):
    rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = rgb.shape[:2]
    vegetation_mask = make_vegetation_mask(rgb)
    labels, candidate_mask = split_vegetation_regions(
        vegetation_mask, opening_size=opening_size, min_distance=min_distance
    )
    soybean_mask = np.zeros((h, w), dtype=np.uint8)
    soybean_before_trim = np.zeros((h, w), dtype=np.uint8)
    rows = []

    for comp_id in np.unique(labels):
        if comp_id == 0:
            continue
        component_mask = labels == comp_id
        ys, xs = np.where(component_mask)
        area = len(xs)
        if area < min_area:
            continue

        x = int(xs.min())
        y = int(ys.min())
        bw = int(xs.max() - x + 1)
        bh = int(ys.max() - y + 1)
        aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
        fill_ratio = area / (bw * bh)

        # Perimeter-based proxy for local leaf/region thickness.
        local = component_mask[y:y + bh, x:x + bw].astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        mean_width = 2.0 * area / max(perimeter, 1.0)

        row = {
            "image": image_path.name,
            "component_id": int(comp_id),
            "x1": x, "y1": y, "x2": x + bw, "y2": y + bh,
            "area": int(area),
            "aspect_ratio": round(aspect_ratio, 3),
            "fill_ratio": round(fill_ratio, 3),
            "mean_width": round(mean_width, 3),
            "broad_core_fraction": None,
            "retained_fraction": None,
            "soybean_score": None,
            "grass_score": None,
            "broadleaf_score": None,
            "seedling_score": None,
            "soil_score": None,
            "keep_as_soybean": False,
            "rejection_reason": "",
        }
        if aspect_ratio > 3.0 or fill_ratio < 0.18 or mean_width < min_width:
            row["rejection_reason"] = "shape_filter"
            rows.append(row)
            continue

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        crop_rgb = rgb[y1:y2, x1:x2].copy()
        local_mask = component_mask[y1:y2, x1:x2]
        background = np.full_like(crop_rgb, 128)
        crop_rgb = np.where(local_mask[:, :, None], crop_rgb, background)
        probs = classify_crop(
            crop_rgb, model, preprocess, text_features, device
        )
        soy, grass, broadleaf, seedling, soil = [float(v) for v in probs]
        best_other = max(grass, broadleaf, seedling, soil)
        keep = (soy >= soybean_threshold and soy - best_other >= margin)

        row.update({
            "soybean_score": round(soy, 4),
            "grass_score": round(grass, 4),
            "broadleaf_score": round(broadleaf, 4),
            "seedling_score": round(seedling, 4),
            "soil_score": round(soil, 4),
            "keep_as_soybean": False,
            "rejection_reason": "bioclip_filter" if not keep else "",
        })
        if keep:
            soybean_before_trim[component_mask] = 255
            refined, core_fraction = refine_accepted_region(
                component_mask,
                min_leaf_half_width=min_leaf_half_width,
                max_thin_extension=max_thin_extension,
                min_broad_fraction=min_broad_fraction,
            )
            retained_fraction = float(refined.sum() / max(area, 1))
            row["broad_core_fraction"] = round(core_fraction, 4)
            row["retained_fraction"] = round(retained_fraction, 4)
            if refined.sum() >= min_area:
                soybean_mask[refined] = 255
                row["keep_as_soybean"] = True
                # Boxes should reflect the *refined* mask, not an old box
                # that may still surround long grass extensions.
                kept_y, kept_x = np.where(refined)
                row["x1"], row["x2"] = int(kept_x.min()), int(kept_x.max()) + 1
                row["y1"], row["y2"] = int(kept_y.min()), int(kept_y.max()) + 1
            else:
                row["rejection_reason"] = "too_thin_after_refinement"
        rows.append(row)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    soybean_mask = cv2.morphologyEx(
        soybean_mask, cv2.MORPH_OPEN, kernel3
    )

    save_image(output_dir / "vegetation_mask.png", vegetation_mask)
    save_image(output_dir / "candidate_mask.png", candidate_mask)
    save_image(output_dir / "soybean_mask.png", soybean_mask)
    save_image(output_dir / "soybean_before_trim_mask.png", soybean_before_trim)

    # Each segmentation mask shown directly on the full-resolution original.
    # BGR: cyan = initial vegetation; yellow = shape-filtered proposals;
    # green = final BioCLIP-accepted soybean mask.
    save_mask_overlay(
        rgb, vegetation_mask, output_dir / "vegetation_overlay.jpg",
        color_bgr=(255, 255, 0),
    )
    save_mask_overlay(
        rgb, candidate_mask, output_dir / "candidate_overlay.jpg",
        color_bgr=(0, 255, 255),
    )
    save_mask_overlay(
        rgb, soybean_mask, output_dir / "soybean_overlay.jpg",
        color_bgr=(0, 255, 0),
    )
    save_mask_overlay(
        rgb, soybean_before_trim,
        output_dir / "soybean_before_trim_overlay.jpg",
        color_bgr=(0, 0, 255),
    )

    overlay_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tinted = overlay_bgr.copy()
    tinted[soybean_mask > 0] = (0, 255, 0)
    overlay_bgr = cv2.addWeighted(overlay_bgr, 0.65, tinted, 0.35, 0)
    for row in rows:
        if not row["keep_as_soybean"]:
            continue
        cv2.rectangle(overlay_bgr, (row["x1"], row["y1"]),
                      (row["x2"], row["y2"]), (0, 255, 0), 2)
        cv2.putText(
            overlay_bgr,
            f'{row["component_id"]}:{row["soybean_score"]:.2f}',
            (row["x1"], max(15, row["y1"] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )
    save_image(output_dir / "overlay.jpg", overlay_bgr)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="soybean_images")
    parser.add_argument("--output", default="bioclip_segmentation")
    parser.add_argument(
        "--output-mode", choices=["replace", "timestamp"], default="replace",
        help="update each image's result folder (default) or save a new timestamped folder"
    )
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--soybean-threshold", type=float, default=0.60)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--opening-size", type=int, default=5)
    parser.add_argument("--min-width", type=float, default=5.0)
    parser.add_argument("--min-distance", type=int, default=12)
    parser.add_argument("--min-leaf-half-width", type=float, default=6.0,
                        help="minimum half-width in pixels for broad-leaf interior")
    parser.add_argument("--max-thin-extension", type=float, default=18.0,
                        help="max length in pixels of narrow extensions from broad interior")
    parser.add_argument("--min-broad-fraction", type=float, default=0.10,
                        help="fraction of a region that must be broad leaf interior")
    args = parser.parse_args()
    if args.opening_size < 1 or args.opening_size % 2 != 1:
        parser.error("--opening-size must be a positive odd number")
    if args.min_width < 0 or args.min_distance < 1 or args.min_area < 1:
        parser.error("min-width, min-distance, and min-area must be positive")
    if not 0 <= args.soybean_threshold <= 1 or not 0 <= args.margin <= 1:
        parser.error("threshold and margin must be between 0 and 1")
    if args.min_leaf_half_width < 0 or args.max_thin_extension < 0:
        parser.error("min-leaf-half-width and max-thin-extension must be nonnegative")
    if not 0 <= args.min_broad_fraction <= 1:
        parser.error("min-broad-fraction must be between 0 and 1")

    input_path = Path(args.input)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(p for p in input_path.rglob("*")
                        if p.is_file() and p.suffix.lower() in extensions)
    else:
        parser.error(f"Input not found: {input_path.resolve()}")
    if not images:
        parser.error(f"No images found in {input_path.resolve()}")
    folder_names = image_folder_names(images, input_path)

    base_output_dir = Path(args.output)
    if args.output_mode == "timestamp":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = base_output_dir.with_name(f"{base_output_dir.name}_{timestamp}")
    else:
        output_dir = base_output_dir

    # Do not write generated overlays into the input image folder.
    if (output_dir.resolve() == input_path.resolve() or
            (input_path.is_dir() and output_dir.resolve().is_relative_to(input_path.resolve()))):
        parser.error("Output directory must be separate from the input image directory")
    if output_dir.resolve() == Path.cwd().resolve():
        parser.error("Choose a dedicated output folder, not the working directory")

    print(f"Input: {input_path.resolve()} ({len(images)} image(s))", flush=True)
    print(f"Output: {output_dir.resolve()} [mode={args.output_mode}]", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BioCLIP 2 on {device}...", flush=True)
    model, preprocess, text_features = load_bioclip(device)

    # Generate into a temporary directory first. Any failed image leaves previous
    # output intact, instead of deleting working masks before writing new ones.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="bioclip_staging_", dir=str(output_dir.parent)
    ) as staging_path:
        staging_dir = Path(staging_path)
        failures = []
        for image_path, folder_name in tqdm(
            zip(images, folder_names), total=len(images), desc="Segmenting"
        ):
            try:
                image_output_dir = staging_dir / folder_name
                image_output_dir.mkdir()
                rows = segment_image(
                    image_path, model, preprocess, text_features,
                    device, image_output_dir, min_area=args.min_area,
                    padding=args.padding,
                    soybean_threshold=args.soybean_threshold,
                    margin=args.margin,
                    opening_size=args.opening_size,
                    min_width=args.min_width,
                    min_distance=args.min_distance,
                    min_leaf_half_width=args.min_leaf_half_width,
                    max_thin_extension=args.max_thin_extension,
                    min_broad_fraction=args.min_broad_fraction,
                )
                pd.DataFrame(rows).to_csv(
                    image_output_dir / "component_scores.csv", index=False
                )
                print(f"{image_path.name}: "
                      f"{sum(r['keep_as_soybean'] for r in rows)} soybean regions")
            except Exception:
                failures.append(str(image_path))
                traceback.print_exc()

        if failures:
            raise RuntimeError(
                f"Failed on {len(failures)} image(s); old results were kept. "
                "See traceback above for details."
            )

        # Publish only after every image succeeds. Update generated files inside
        # each image's folder while preserving notes and other unrelated files.
        # Legacy flat outputs are left untouched; new runs use image folders.
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_results in staging_dir.iterdir():
            destination = output_dir / image_results.name
            destination.mkdir(parents=True, exist_ok=True)
            for result in image_results.iterdir():
                os.replace(result, destination / result.name)

    print(f"Saved results for {len(images)} image(s): {output_dir.resolve()}", flush=True)
    print("Each image has its own folder with soybean_mask.png, "
          "soybean_overlay.jpg, other previews, and component_scores.csv")


if __name__ == "__main__":
    main()
