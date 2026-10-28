"""Cluster pixels inside PVC quadrats; review clusters before naming species.

Install: python -m pip install numpy pillow scipy scikit-learn scikit-image matplotlib joblib
Run:     python soybean_clustering.py --input images --output results

No species labels are used to fit the color model. Shape filtering then keeps
broadleaf candidates using width, elongation, solidity and circularity rules.
These are unvalidated broadleaf candidates, not a trained soybean classifier.
Only --soybean-clusters creates a tentative soybean/no-soybean interpretation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageOps
from scipy import ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from threadpoolctl import threadpool_limits


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
WEIGHTS = np.array([0.30, 1.00, 0.65, 1.20], dtype=np.float32)
SRGB, LAB = ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")


def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def check_points(points, image_size):
    """Require an ordered, convex quadrilateral: TL, TR, BR, BL."""
    p = np.asarray(points, dtype=float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("Supply four finite x,y points: TL, TR, BR, BL.")
    width, height = image_size
    if (p < 0).any() or (p[:, 0] >= width).any() or (p[:, 1] >= height).any():
        raise ValueError("A corner is outside the image.")
    edges = np.roll(p, -1, axis=0) - p
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not ((crosses > 0).all() or (crosses < 0).all()):
        raise ValueError("Corners must form a convex quadrilateral, in order.")
    area = abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
               - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2
    if area < 100:
        raise ValueError("Selected quadrat is too small.")
    return p


def click_points(im, title):
    import matplotlib.pyplot as plt
    if "agg" == plt.get_backend().lower():
        raise RuntimeError("No interactive display. Run locally, or use --corners-json.")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(im)
    ax.set_title(title + "\nClick just INSIDE the pipe: TL, TR, BR, BL. Close to cancel.")
    fig.tight_layout()
    points = plt.ginput(4, timeout=0, show_clicks=True)
    plt.close(fig)
    return check_points(points, im.size)


def rectify(im, points, size):
    """Map a quadrilateral to a square; return output-to-input homography."""
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    a, b = [], []
    for (u, v), (x, y) in zip(target, points):
        a.extend([[u, v, 1, 0, 0, 0, -x * u, -x * v],
                  [0, 0, 0, u, v, 1, -y * u, -y * v]])
        b.extend([x, y])
    coefficients = np.linalg.solve(a, b)
    h = np.append(coefficients, 1).reshape(3, 3)
    crop = im.transform((size, size), Image.Transform.PERSPECTIVE,
                        tuple(coefficients), Image.Resampling.BILINEAR)
    return crop, h


def features(crop):
    """Color features only; no x/y position, so a row location is not a shortcut."""
    rgb = np.asarray(crop, dtype=np.float32)
    lab = np.asarray(ImageCms.profileToProfile(crop, SRGB, LAB, outputMode="LAB"),
                     dtype=np.float32)
    chromaticity = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 1)
    excess_green = 2 * chromaticity[..., 1] - chromaticity[..., 0] - chromaticity[..., 2]
    return np.dstack([lab, excess_green]).reshape(-1, 4), excess_green


def filter_leaf_shapes(mask, min_area=80, min_width=18, max_elongation=3.0,
                       min_solidity=0.75, min_circularity=0.43):
    """Geometry rules AFTER color clustering; dimensions refer to an 800px ROI.

    Watershed regions are approximations, not counted or confirmed leaves.
    Output is restricted to the supplied candidate pixels. Narrow appendages
    are trimmed even when a blade is connected to an otherwise broad region.
    """
    raw = np.asarray(mask, dtype=bool)
    if raw.ndim != 2:
        raise ValueError("Shape filtering expects a two-dimensional mask.")
    if not raw.any():
        return raw.copy(), []
    scale = min(raw.shape) / 800.0
    width_limit = min_width * scale
    area_limit = min_area * scale * scale
    # Repair only small holes for measurements; output never adds these pixels.
    cleaned = ndi.binary_closing(raw, structure=np.ones((3, 3))) | raw
    distance = ndi.distance_transform_edt(np.pad(cleaned, 1))[1:-1, 1:-1]
    smooth = ndi.gaussian_filter(distance, sigma=max(0.5, scale))
    # Suppress shallow peaks so one grass blade is not split into round chunks.
    maxima = h_maxima(smooth, 2.0 * scale) & cleaned
    markers, count = ndi.label(maxima)
    if count == 0:
        return np.zeros_like(raw), []
    regions = watershed(-distance, markers, mask=cleaned)
    keep = np.zeros(count + 1, dtype=bool)
    stats = []
    for region in regionprops(regions):
        width = float(region.axis_minor_length)
        elongation = float(region.axis_major_length / max(width, 1e-6))
        circularity = float(min(1.0, 4 * np.pi * region.area /
                                max(region.perimeter ** 2, 1e-6)))
        reasons = []
        if region.area < area_limit:
            reasons.append("small")
        if width < width_limit:
            reasons.append("narrow")
        if elongation > max_elongation:
            reasons.append("elongated")
        if region.solidity < min_solidity:
            reasons.append("low_solidity")
        if circularity < min_circularity:
            reasons.append("low_circularity")
        keep[region.label] = not reasons
        y0, x0, y1, x1 = region.bbox
        stats.append({"region_id": int(region.label), "area_pixels": int(region.area),
                      "minor_width_at_800px": round(width / scale, 3),
                      "elongation": round(elongation, 3),
                      "solidity": round(float(region.solidity), 3),
                      "circularity": round(circularity, 3),
                      "kept_region": not reasons, "rejection_reason": ";".join(reasons),
                      "touches_roi_border": y0 == 0 or x0 == 0 or
                                            y1 == raw.shape[0] or x1 == raw.shape[1]})
    # Limited restoration from wide cores avoids restoring an attached thin blade.
    radius = max(1, int(round(width_limit / 2)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    wide = ndi.binary_dilation(distance >= radius, structure=disk) & cleaned
    result = keep[regions] & wide & raw
    retained = np.bincount(regions[result], minlength=count + 1)
    for row in stats:
        row["retained_pixels"] = int(retained[row["region_id"]])
    return result, stats


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_figure(fig, path, dpi):
    """Finish and decode the PNG before replacing the user-visible preview."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    buffer.seek(0)
    with Image.open(buffer) as image:
        rgb = image.convert("RGB")
    temporary = path.with_name(path.name + ".tmp")
    rgb.save(temporary, format="PNG")
    temporary.replace(path)


def save_panels(crop, labels, candidate, positive, k, summary_path, review_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    rgb = np.asarray(crop)
    colors = plt.get_cmap("tab20")(np.arange(k))[:, :3]
    colored = colors[labels]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4))
    axes[0].imshow(rgb)
    axes[0].set_title("Inside PVC square only")
    axes[1].imshow(colored)
    axes[1].set_title(f"{k} unsupervised color clusters")
    axes[1].legend(handles=[Patch(color=colors[i], label=str(i)) for i in range(k)],
                   ncol=min(k, 6), loc="upper center", bbox_to_anchor=(0.5, -0.015),
                   frameon=False, handlelength=1)
    mask = candidate
    overlay = rgb.astype(float).copy()
    overlay[mask] = 0.42 * overlay[mask] + 0.58 * np.array([250, 199, 45])
    axes[2].imshow(overlay.astype(np.uint8))
    if positive is None:
        axes[2].set_title("Color + shape candidates\nBroad leaves; not verified soybean")
    else:
        axes[2].set_title("User-mapped soybean + shape\nTentative species interpretation")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, summary_path, dpi=150)
    plt.close(fig)

    rows = math.ceil(k / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < k:
            selection = labels == i
            visible = (rgb * 0.16).astype(np.uint8)
            visible[selection] = rgb[selection]
            ax.imshow(visible)
            ax.set_title(f"Cluster {i} | {100 * selection.mean():.1f}% of crop")
        ax.axis("off")
    fig.suptitle("Review every cluster: plant colors can include soybean AND weeds", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, review_path, dpi=130)
    plt.close(fig)


def save_shape_comparison(crop, color_mask, kept_mask, path):
    import matplotlib.pyplot as plt
    rgb = np.asarray(crop)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0))
    axes[0].imshow(rgb)
    axes[0].set_title("Inside PVC square")
    for ax, mask, title in zip(axes[1:], [color_mask, kept_mask],
                              ["Color candidates", "Color + shape candidates"]):
        overlay = rgb.astype(float).copy()
        overlay[mask] = 0.42 * overlay[mask] + 0.58 * np.array([250, 199, 45])
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Yellow marks candidate pixels; species identification is unvalidated.", fontsize=11)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)
    plt.close(fig)


def process_one(item, model, grid, presence_percent, shape_options=None):
    key, crop_path, original_path, points, homography = item
    folder = crop_path.parent
    crop = load_rgb(crop_path)
    raw, exg = features(crop)
    with threadpool_limits(limits=2):
        labels = model["kmeans"].predict(model["scaler"].transform(raw) * WEIGHTS)
    labels = labels.reshape(crop.height, crop.width).astype(np.uint8)
    Image.fromarray(labels).save(folder / "cluster_ids.png")
    color_candidate = np.isin(labels, model["green_cluster_ids"])
    Image.fromarray(color_candidate.astype(np.uint8) * 255).save(folder / "vegetation_candidates.png")
    positive = model.get("soybean_cluster_ids")
    # Reviewed mappings can include injured/non-green leaves; filter their shape directly.
    shape_source = color_candidate if positive is None else np.isin(labels, positive)
    candidate, shape_rows = filter_leaf_shapes(shape_source, **(shape_options or {}))
    Image.fromarray(candidate.astype(np.uint8) * 255).save(folder / "broadleaf_candidates.png")
    Image.fromarray((shape_source & ~candidate).astype(np.uint8) * 255).save(folder / "shape_rejected.png")
    shape_csv = folder / "shape_regions.csv"
    shape_csv.unlink(missing_ok=True)
    if shape_rows:
        write_csv(shape_csv, shape_rows)
    else:
        shape_csv.write_text("region_id,area_pixels,minor_width_at_800px,elongation,solidity,circularity,kept_region,rejection_reason,touches_roi_border,retained_pixels\n",
                             encoding="utf-8")
    save_shape_comparison(crop, shape_source, candidate, folder / "shape_comparison.png")
    if positive is not None:
        # A post-hoc interpretation, not species knowledge learned by K-means.
        semantic = candidate.astype(np.uint8)
        Image.fromarray(semantic).save(folder / "class_mask.png")
        inverse = np.linalg.inv(homography)
        inverse /= inverse[2, 2]
        original = load_rgb(original_path)
        restored = Image.fromarray(semantic).transform(
            original.size, Image.Transform.PERSPECTIVE, tuple(inverse.ravel()[:8]),
            Image.Resampling.NEAREST, fillcolor=255)
        restored.save(folder / "class_mask_original.png")
    rows = []
    rgb = np.asarray(crop)
    for i in range(model["kmeans"].n_clusters):
        m = labels == i
        avg_rgb = rgb[m].mean(axis=0) if m.any() else [np.nan] * 3
        rows.append({"cluster_id": i, "crop_percent": round(100 * float(m.mean()), 4),
                     "mean_R": round(float(avg_rgb[0]), 2),
                     "mean_G": round(float(avg_rgb[1]), 2),
                     "mean_B": round(float(avg_rgb[2]), 2),
                     "mean_excess_green": round(float(exg[m].mean()), 4) if m.any() else ""})
    write_csv(folder / "cluster_stats.csv", rows)
    tile_rows = []
    for r in range(grid):
        for c in range(grid):
            y0, y1 = r * crop.height // grid, (r + 1) * crop.height // grid
            x0, x1 = c * crop.width // grid, (c + 1) * crop.width // grid
            row = {"tile_row": r, "tile_col": c,
                   "vegetation_candidate_percent": round(100 * float(color_candidate[y0:y1, x0:x1].mean()), 3),
                   "broadleaf_candidate_percent": round(100 * float(candidate[y0:y1, x0:x1].mean()), 3)}
            if positive is not None:
                row["user_mapped_soybean_percent"] = round(100 * float(semantic[y0:y1, x0:x1].mean()), 3)
                row["user_mapped_soybean_present"] = bool(
                    100 * float(semantic[y0:y1, x0:x1].mean()) >= presence_percent)
            tile_rows.append(row)
    write_csv(folder / "tiles.csv", tile_rows)
    save_panels(crop, labels, candidate, positive, model["kmeans"].n_clusters,
                folder / "summary.png", folder / "cluster_review.png")
    summary = {"image": key, "vegetation_candidate_percent": round(100 * float(color_candidate.mean()), 3),
               "broadleaf_candidate_percent": round(100 * float(candidate.mean()), 3),
               "soybean_mapping_applied": positive is not None}
    if positive is not None:
        summary["user_mapped_soybean_percent"] = round(100 * float(semantic.mean()), 3)
        summary["user_mapped_soybean_present"] = bool(100 * float(semantic.mean()) >= presence_percent)
    else:
        # Final cleanup: a refit or unmapped run must not retain old species masks.
        for name in ("class_mask.png", "class_mask_original.png"):
            (folder / name).unlink(missing_ok=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="One image, or a folder of images (recursive).")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--corners-json", type=Path, help="Saved per-image corners; otherwise click four points.")
    parser.add_argument("--clusters", type=int, default=6, help="Intermediate color clusters, not semantic classes.")
    parser.add_argument("--size", type=int, default=800, help="Rectified square side in pixels.")
    parser.add_argument("--samples-per-image", type=int, default=5000)
    parser.add_argument("--fit-images", type=int, default=100, help="Maximum images sampled to fit one shared model.")
    parser.add_argument("--grid", type=int, default=8, help="Export per-tile candidate percentages.")
    parser.add_argument("--presence-percent", type=float, default=1.0,
                        help="After class mapping: minimum %% soybean candidate pixels for present (default 1; unvalidated).")
    parser.add_argument("--refit", action="store_true", help="Fit a new model; cluster numbers may change.")
    parser.add_argument("--min-leaf-width", type=float, default=18,
                        help="Minimum region minor width, normalized to an 800px ROI (default 18).")
    parser.add_argument("--max-leaf-elongation", type=float, default=3.0,
                        help="Maximum major/minor axis ratio (default 3).")
    parser.add_argument("--min-leaf-circularity", type=float, default=0.43,
                        help="Minimum region compactness, 4*pi*area/perimeter^2 (default 0.43).")
    parser.add_argument("--soybean-clusters", nargs="+", type=int,
                        help="After visual review: IDs interpreted as soybean; all other IDs become no-soybean.")
    args = parser.parse_args()
    if not 2 <= args.clusters <= 20:
        parser.error("--clusters must be between 2 and 20.")
    if args.size < 32 or not 1 <= args.grid <= args.size:
        parser.error("--size must be >=32 and --grid between 1 and --size.")
    if args.samples_per_image < args.clusters or args.fit_images < 1:
        parser.error("Use at least --clusters samples per image and one fit image.")
    if not 0 < args.presence_percent <= 100:
        parser.error("--presence-percent must be greater than 0 and at most 100.")
    if args.min_leaf_width <= 0 or args.max_leaf_elongation < 1:
        parser.error("Leaf width must be positive and maximum elongation must be at least 1.")
    if not 0 <= args.min_leaf_circularity <= 1:
        parser.error("--min-leaf-circularity must be between 0 and 1.")
    shape_options = {"min_width": args.min_leaf_width,
                     "max_elongation": args.max_leaf_elongation,
                     "min_circularity": args.min_leaf_circularity}
    if args.input.is_file():
        paths = [args.input.resolve()]
        base = args.input.resolve().parent
    elif args.input.is_dir():
        base = args.input.resolve()
        out = args.output.resolve()
        paths = sorted(p for p in base.rglob("*") if p.is_file()
                       and p.suffix.lower() in EXTENSIONS and not p.resolve().is_relative_to(out))
    else:
        parser.error("--input does not exist.")
    if not paths:
        parser.error("No supported images found.")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "kmeans_model.joblib"
    model = joblib.load(model_path) if model_path.exists() and not args.refit else None
    if model is not None:
        args.size = model["size"]
        print("Reusing existing model and its crop size; --clusters only applies when fitting.")
    corners_path = args.corners_json or args.output / "roi_corners.json"
    corners = json.loads(corners_path.read_text(encoding="utf-8")) if corners_path.exists() else {"images": {}}
    # A per-image key prevents accidentally applying one photo's quadrat to all photos.
    prepared = []
    for path in paths:
        key = path.relative_to(base).as_posix()
        im = load_rgb(path)
        entry = corners["images"].get(key)
        if entry is None:
            points = click_points(im, key)
            corners["images"][key] = {"image_size": list(im.size), "corners": points.tolist()}
        else:
            if entry["image_size"] != list(im.size):
                raise ValueError(f"{key}: saved corners have a different image size. Reselect its corners.")
            points = check_points(entry["corners"], im.size)
        (args.output / "roi_corners.json").write_text(json.dumps(corners, indent=2), encoding="utf-8")
        # Include the extension in the output key so sample.jpg and sample.png do not collide.
        folder = args.output / "images" / key
        folder.mkdir(parents=True, exist_ok=True)
        crop, h = rectify(im, points, args.size)
        crop_path = folder / "roi.png"
        crop.save(crop_path)
        outline = im.copy()
        ImageDraw.Draw(outline).line([tuple(p) for p in points] + [tuple(points[0])],
                                    fill=(255, 190, 0), width=max(2, im.width // 350))
        outline.thumbnail((1200, 1200))
        outline.save(folder / "selected_roi.png")
        prepared.append((key, crop_path, path, points, h))

    if model is None:
        rng = np.random.default_rng(42)
        selected = np.sort(rng.choice(len(prepared), min(args.fit_images, len(prepared)), replace=False))
        samples = []
        for i in selected:
            raw, _ = features(load_rgb(prepared[i][1]))
            take = rng.choice(len(raw), min(args.samples_per_image, len(raw)), replace=False)
            samples.append(raw[take])
        sample = np.concatenate(samples)
        scaler = StandardScaler().fit(sample)
        with threadpool_limits(limits=2):
            kmeans = MiniBatchKMeans(n_clusters=args.clusters, n_init=10,
                                    random_state=42, batch_size=4096)
            kmeans.fit(scaler.transform(sample) * WEIGHTS)
        centers_raw = scaler.inverse_transform(kmeans.cluster_centers_ / WEIGHTS)
        # A heuristic interpretation of green COLOR, not soybean species.
        green_ids = np.flatnonzero(centers_raw[:, 3] >= 0.10).tolist()
        model = {"scaler": scaler, "kmeans": kmeans, "size": args.size,
                 "green_cluster_ids": green_ids, "soybean_cluster_ids": None,
                 "feature_names": ["Lab L", "Lab a", "Lab b", "normalized excess green"],
                 "training_images": [prepared[i][0] for i in selected]}
        print(f"Fitted {args.clusters} clusters on {len(sample):,} unlabeled pixel samples.")
    if args.soybean_clusters is not None:
        if any(i < 0 or i >= model["kmeans"].n_clusters for i in args.soybean_clusters):
            parser.error("A requested soybean cluster ID is outside this model's range.")
        model["soybean_cluster_ids"] = sorted(set(args.soybean_clusters))
    joblib.dump(model, model_path)
    print("Green candidate cluster IDs:", model["green_cluster_ids"])
    print("A green candidate mask includes grass/weeds. It is not a verified soybean mask.")
    print("Adding width, elongation, solidity and circularity rules; broadleaf candidates remain unvalidated.")
    if model["soybean_cluster_ids"] is not None:
        print("Applying user soybean mapping:", model["soybean_cluster_ids"],
              "(all other clusters interpreted as no-soybean).")
    summaries = []
    for item in prepared:
        summaries.append(process_one(item, model, args.grid, args.presence_percent, shape_options))
        print("Processed", item[0])
    write_csv(args.output / "results.csv", summaries)
    (args.output / "run_info.json").write_text(json.dumps({
        "method": "Unsupervised color MiniBatchKMeans followed by rule-based broadleaf shape filtering",
        "shape_filter": {**shape_options, "min_area_at_800px": 80, "min_solidity": 0.75,
                         "algorithm": "distance peaks, watershed regions, shape rules and width-core trimming",
                         "thresholds_validated": False},
        "green_cluster_ids": model["green_cluster_ids"],
        "soybean_cluster_ids": model["soybean_cluster_ids"],
        "classes_if_user_mapped": {"0": "no soybean", "1": "soybean", "255": "outside ROI (original-size mask only)"},
        "warning": "Species interpretation is unvalidated. Broadleaf weeds can pass, and small, overlapping or damaged soybean leaves may be rejected.",
        "coverage": "Fraction of rectified image pixels; projected canopy cover, not physical leaf area.",
        "presence_percent_threshold": args.presence_percent,
        "presence_threshold_warning": "Default 1 percent is an adjustable heuristic, not a validated detection threshold.",
        "accuracy": None,
    }, indent=2), encoding="utf-8")
    print("Done. Open summary.png and cluster_review.png for each image under", args.output / "images")


if __name__ == "__main__":
    main()
