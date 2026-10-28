
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


MODEL = "hf-hub:imageomics/bioclip-2"

LABELS = [
    "a soybean plant",
    "a grass weed",
    "a broadleaf weed",
    "a small seedling",
    "bare soil without plants",
]


def get_positions(length, patch, stride):
    positions = list(range(0, max(1, length - patch + 1), stride))
    last = max(0, length - patch)
    if positions[-1] != last:
        positions.append(last)
    return positions


def process_image(path, model, preprocess, text_features,
                  device, output_dir, patch, stride,
                  threshold, min_area, max_side):

    original = Image.open(path).convert("RGB")
    ow, oh = original.size

    scale = min(1.0, max_side / max(ow, oh))
    w = max(patch, round(ow * scale))
    h = max(patch, round(oh * scale))

    image = original.resize((w, h))
    rgb = np.asarray(image)

    heat_sum = np.zeros((h, w), np.float32)
    weight_sum = np.zeros((h, w), np.float32)

    xs = get_positions(w, patch, stride)
    ys = get_positions(h, patch, stride)
    locations = [(x, y) for y in ys for x in xs]

    # Smooth weighting reduces patch-edge artifacts
    a = np.hanning(patch + 2)[1:-1]
    weights = np.outer(a, a).astype(np.float32)
    weights = np.maximum(weights, 0.02)

    batch_size = 8 if device == "cuda" else 2

    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start:start + batch_size]

        crops = [
            preprocess(image.crop((x, y, x + patch, y + patch)))
            for x, y in batch_locations
        ]

        batch = torch.stack(crops).to(device)

        with torch.inference_mode():
            features = model.encode_image(batch)
            features = torch.nn.functional.normalize(
                features, dim=-1
            )

            similarities = features @ text_features.T

            # Relative probabilities across supplied labels
            probs = (100 * similarities).softmax(dim=-1)
            soybean_scores = probs[:, 0].cpu().numpy()

        for (x, y), score in zip(batch_locations, soybean_scores):
            heat_sum[y:y+patch, x:x+patch] += score * weights
            weight_sum[y:y+patch, x:x+patch] += weights

    heatmap = heat_sum / np.maximum(weight_sum, 1e-6)

    # Convert patch scores into candidate pixel regions
    mask = (heatmap >= threshold).astype(np.uint8) * 255

    # Remove small noisy regions
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask, connectivity=8)
    )

    filtered = np.zeros_like(mask)
    detections = []

    for i in range(1, count):
        x, y, bw, bh, area = stats[i]

        if area < min_area:
            continue

        component = labels == i
        filtered[component] = 255

        mean_score = float(heatmap[component].mean())

        # Coordinates mapped back to the original image
        x1 = round(x * ow / w)
        y1 = round(y * oh / h)
        x2 = round((x + bw) * ow / w)
        y2 = round((y + bh) * oh / h)

        detections.append({
            "image": path.name,
            "region_id": len(detections) + 1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area_pixels_processed": int(area),
            "mean_soybean_score": round(mean_score, 4),
        })

    stem = path.stem

    # Heatmap: red indicates higher soybean similarity
    heat_uint8 = np.uint8(np.clip(heatmap * 255, 0, 255))
    colored_heat = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    heat_overlay = cv2.addWeighted(
        rgb_bgr, 0.55, colored_heat, 0.45, 0
    )

    # Annotated regions
    annotated = rgb_bgr.copy()
    green = np.zeros_like(annotated)
    green[:, :] = (0, 255, 0)

    overlay = cv2.addWeighted(annotated, 0.65, green, 0.35, 0)
    annotated[filtered > 0] = overlay[filtered > 0]

    for detection in detections:
        x1 = round(detection["x1"] * w / ow)
        y1 = round(detection["y1"] * h / oh)
        x2 = round(detection["x2"] * w / ow)
        y2 = round(detection["y2"] * h / oh)

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.putText(
            annotated,
            str(detection["region_id"]),
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )

    cv2.imwrite(str(output_dir / f"{stem}_heatmap.jpg"), heat_overlay)
    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), filtered)
    cv2.imwrite(str(output_dir / f"{stem}_detections.jpg"), annotated)

    # Save the raw score map for future threshold experiments
    np.save(output_dir / f"{stem}_scores.npy", heatmap)

    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="soybean_images")
    parser.add_argument("--output", default="bioclip_localization")
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()

    if args.patch < 2 or args.stride < 1 or not 0 <= args.threshold <= 1:
        parser.error("Invalid patch, stride, or threshold")
    if args.min_area < 1 or args.max_side < args.patch:
        parser.error("Invalid min-area or max-side")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading BioCLIP 2...")

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to(device).eval()

    prompts = [f"a photo of {label}" for label in LABELS]

    with torch.inference_mode():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(
            text_features, dim=-1
        )

    input_path = Path(args.input)
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        )
    else:
        parser.error(f"Input not found: {input_path}")

    all_detections = []

    for image_path in tqdm(images, desc="Localizing"):
        try:
            detections = process_image(
                image_path, model, preprocess, text_features,
                device, output_dir, args.patch, args.stride,
                args.threshold, args.min_area, args.max_side
            )
            all_detections.extend(detections)
            print(f"{image_path.name}: {len(detections)} regions")
        except Exception as exc:
            print(f"Failed {image_path}: {exc}")

    columns = [
        "image", "region_id", "x1", "y1", "x2", "y2",
        "area_pixels_processed", "mean_soybean_score"
    ]
    pd.DataFrame(all_detections, columns=columns).to_csv(
        output_dir / "detections.csv", index=False
    )
    print(f"Saved {len(all_detections)} regions to {output_dir}")


if __name__ == "__main__":
    main()
