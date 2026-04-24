"""
prepare.py — Download and preprocess the TACO litter dataset.

This script is NOT modified by the agent. It downloads the TACO dataset from
HuggingFace, reads the COCO annotations JSON from the included ZIP file,
converts polygon segmentation annotations to binary pixel masks
(litter vs. background), and saves everything to the repo-local data/ folder
ready for auto-research/train.py.

Dataset: https://huggingface.co/datasets/Zesky665/TACO
Format:  COCO_format.zip inside the HF snapshot contains:
           data/annotations.json   — COCO JSON with segmentation polygons
           data/batch_*/           — image files

Output layout:
    data/
        images/       *.jpg  (resized to IMAGE_SIZE x IMAGE_SIZE)
        masks/        *.png  (binary uint8: 0=background, 255=litter)
        train.txt     list of stem names for training split
        val.txt       list of stem names for validation split
        meta.json     dataset statistics
"""

import io
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[1]
IMAGE_SIZE    = 512          # spatial resolution stored on disk
VAL_FRACTION  = 0.15
RANDOM_SEED   = 42
DATA_DIR      = REPO_ROOT / "data"
IMAGES_DIR    = DATA_DIR / "images"
MASKS_DIR     = DATA_DIR / "masks"
HF_REPO       = "Zesky665/TACO"
ZIP_INNER     = "COCO_format.zip"
ANNOTATIONS   = "data/annotations.json"
# ─────────────────────────────────────────────────────────────────────────────


def find_zip(snapshot_dir: str) -> str:
    for root, _, files in os.walk(snapshot_dir):
        for f in files:
            if f == ZIP_INNER:
                return os.path.join(root, f)
    raise FileNotFoundError(f"{ZIP_INNER} not found under {snapshot_dir}")


def polygon_to_mask(segmentation: list, width: int, height: int) -> np.ndarray:
    """Convert a COCO flat-polygon list [[x1,y1,x2,y2,...], ...] to a uint8 mask."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        xy = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(xy, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def main():
    random.seed(RANDOM_SEED)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download / locate the dataset ─────────────────────────────────────
    print(f"Downloading {HF_REPO} snapshot …")
    snapshot_dir = snapshot_download(repo_id=HF_REPO, repo_type="dataset")
    print(f"  Snapshot at: {snapshot_dir}")

    zip_path = find_zip(snapshot_dir)
    print(f"  ZIP found: {zip_path}")

    # ── Parse COCO annotations ────────────────────────────────────────────
    print("Reading COCO annotations …")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ANNOTATIONS) as f:
            coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    image_ids = list(images_by_id.keys())
    print(f"  Images: {len(image_ids)}   Annotations: {len(coco['annotations'])}")

    # ── Split ─────────────────────────────────────────────────────────────
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * VAL_FRACTION))
    splits = {
        "val":   image_ids[:n_val],
        "train": image_ids[n_val:],
    }

    # ── Process ───────────────────────────────────────────────────────────
    stems_by_split: dict[str, list[str]] = {}

    with zipfile.ZipFile(zip_path) as zf:
        # Build a case-insensitive lookup of zip entries (some files are .JPG)
        name_map = {}
        for entry in zf.namelist():
            name_map[entry.lower()] = entry

        for split_name, ids in splits.items():
            stems = []
            skipped = 0
            for img_id in tqdm(ids, desc=f"Processing {split_name}"):
                meta = images_by_id[img_id]
                # file_name is like "batch_1/000006.jpg"
                inner_path = f"data/{meta['file_name']}"
                inner_key  = inner_path.lower()

                if inner_key not in name_map:
                    skipped += 1
                    continue

                try:
                    with zf.open(name_map[inner_key]) as img_f:
                        img_bytes = img_f.read()
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    tqdm.write(f"  Skipping {inner_path}: {e}")
                    skipped += 1
                    continue

                orig_w, orig_h = img.size
                img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

                # ── Build binary mask from all annotations ─────────────────
                combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for ann in anns_by_image.get(img_id, []):
                    seg = ann.get("segmentation", [])
                    if not seg or isinstance(seg, dict):   # skip RLE
                        continue
                    m = polygon_to_mask(seg, orig_w, orig_h)
                    combined_mask = np.maximum(combined_mask, m)

                mask_pil     = Image.fromarray(combined_mask * 255, mode="L")
                mask_resized = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

                stem = f"{img_id:06d}"
                img_resized.save(IMAGES_DIR / f"{stem}.jpg", quality=92)
                mask_resized.save(MASKS_DIR  / f"{stem}.png")
                stems.append(stem)

            if skipped:
                print(f"  Skipped {skipped} entries in {split_name}")
            stems_by_split[split_name] = stems

    (DATA_DIR / "train.txt").write_text("\n".join(stems_by_split["train"]) + "\n")
    (DATA_DIR / "val.txt").write_text(  "\n".join(stems_by_split["val"])   + "\n")

    # ── Statistics ────────────────────────────────────────────────────────
    all_stems = stems_by_split["train"] + stems_by_split["val"]
    litter_pixels = 0
    total_pixels  = 0
    for stem in all_stems[:200]:
        m = np.array(Image.open(MASKS_DIR / f"{stem}.png"))
        litter_pixels += int((m > 127).sum())
        total_pixels  += m.size

    pos_frac = litter_pixels / max(total_pixels, 1)
    meta_out = {
        "image_size":                    IMAGE_SIZE,
        "train_count":                   len(stems_by_split["train"]),
        "val_count":                     len(stems_by_split["val"]),
        "litter_pixel_fraction_sample":  round(pos_frac, 4),
        "pos_weight_suggestion":         round((1 - pos_frac) / max(pos_frac, 1e-6), 2),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta_out, indent=2))

    print("\nDone.")
    print(f"  Train: {meta_out['train_count']}   Val: {meta_out['val_count']}")
    print(f"  Litter pixel fraction (200-sample): {pos_frac:.2%}")
    print(f"  Suggested BCEWithLogitsLoss pos_weight: {meta_out['pos_weight_suggestion']}")
    print(f"  Metadata → {DATA_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
