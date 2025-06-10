import os, glob
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
SRC_ROOT    = 'data'      # your original data
OUT_ROOT    = 'data2'     # where to dump overlays
CLASSES     = ['Angry','Anxious','Excitement','Sadness']
THRESHOLD   = 127         # binarization
TARGET_SIZE = (128, 128)  # <-- pick a uniform size that works for you

# --- STEP 1: build per-class templates ---
templates = {}
for cls in CLASSES:
    train_paths = glob.glob(os.path.join(SRC_ROOT, 'train', cls, '*.jpg'))
    if not train_paths:
        raise RuntimeError(f"No train images found for class {cls}")

    print(f"Building template for class '{cls}' from {len(train_paths)} images...")
    acc = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)
    for p in train_paths:
        # skip any training images with "aug" in the filename
        if 'aug' in os.path.basename(p).lower():
            continue

        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        _, bw = cv2.threshold(im, THRESHOLD, 255, cv2.THRESH_BINARY)
        acc += (bw.astype(np.float32) / 255.0)

    mean_mask = (acc / len([p for p in train_paths if 'aug' not in os.path.basename(p).lower()])) > 0.5
    templates[cls] = (mean_mask.astype(np.uint8) * 255)
print("Templates built!\n")

# --- STEP 2: process both splits and save overlays ---
for split in ['train', 'val']:
    for cls in CLASSES:
        in_dir  = os.path.join(SRC_ROOT, split, cls)
        out_dir = os.path.join(OUT_ROOT, split, cls)
        os.makedirs(out_dir, exist_ok=True)

        all_files = glob.glob(os.path.join(in_dir, '*.jpg'))
        # filter out any filename containing "aug"
        file_list = [p for p in all_files if 'aug' not in os.path.basename(p).lower()]
        skipped = len(all_files) - len(file_list)

        print(f"[{split}/{cls}] Found {len(all_files)} files in '{in_dir}'.")
        print(f"[{split}/{cls}] Skipping {skipped} files containing 'aug'; processing {len(file_list)} files.")
        print(f"[{split}/{cls}] Saving overlays to '{out_dir}'.\n")

        tpl = templates[cls]
        for src_path in tqdm(file_list, desc=f"{split}/{cls}", unit="img"):
            tqdm.write(f"  → Processing file: {src_path}")

            # load, resize, binarize
            im = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            _, bw = cv2.threshold(im, THRESHOLD, 255, cv2.THRESH_BINARY)

            # compute diff & overlay in red
            diff = cv2.absdiff(bw, tpl)
            bgr  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            bgr[diff == 255] = (0, 0, 255)

            # save
            fname = os.path.basename(src_path)
            cv2.imwrite(os.path.join(out_dir, fname), bgr)
        print()  # blank line between classes
print("All done!")