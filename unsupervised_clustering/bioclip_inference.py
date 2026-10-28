
from pathlib import Path
import argparse

import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Categories to compare against your weed images
LABELS = [
    "a soybean plant",
    "a grass weed",
    "a broadleaf weed",
    "a small seedling",
    "bare soil without plants",
]

MODEL_NAME = "hf-hub:imageomics/bioclip-2"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="soybean_images")
    parser.add_argument("--output", default="bioclip_results")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder missing: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading BioCLIP 2...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(device).eval()

    prompts = [f"a photo of {label}" for label in LABELS]
    text_tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = torch.nn.functional.normalize(
            text_features, dim=-1
        )

    images = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images")

    results = []

    for path in tqdm(images):
        try:
            with Image.open(path) as img:
                image = preprocess(img.convert("RGB"))
            image = image.unsqueeze(0).to(device)

            with torch.inference_mode():
                image_features = model.encode_image(image)
                image_features = torch.nn.functional.normalize(
                    image_features, dim=-1
                )

                similarities = (
                    image_features @ text_features.T
                ).squeeze(0)

                probabilities = (
                    100 * similarities
                ).softmax(dim=-1).cpu().tolist()

            top_index = max(
                range(len(probabilities)),
                key=lambda i: probabilities[i]
            )

            row = {
                "filename": str(path.relative_to(input_dir)),
                "prediction": LABELS[top_index],
                "score": round(probabilities[top_index], 4),
            }

            for label, score in zip(LABELS, probabilities):
                row[label] = round(score, 4)

            results.append(row)

        except Exception as error:
            print(f"Error processing {path}: {error}")

    output_file = output_dir / "predictions.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)

    print(f"\nProcessed {len(results)} images")
    print(f"Results saved to: {output_file}")

    print("\nPrediction counts:")
    if results:
        print(pd.DataFrame(results)["prediction"].value_counts())


if __name__ == "__main__":
    main()
