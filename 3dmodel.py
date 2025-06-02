import os
import cv2
import numpy as np
import pandas as pd

# --- Config ---
MASK_DIR = "masks_binary"  # Folder with binary edge maps
FRAME_PREFIX = "frame_"       # Assuming frame_0001.jpg etc.

results = []


files = [f for f in os.listdir("masks_binary") if f.startswith("frame_")]
print(f"Total predicted masks: {len(files)}")


# --- Process each mask ---
for fname in sorted(os.listdir(MASK_DIR)):
    if not fname.endswith((".png", ".jpg")):
        continue

    path = os.path.join(MASK_DIR, fname)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None or np.count_nonzero(mask) == 0:
        continue

    height, width = mask.shape
    diameters = []

    for y in range(height):
        row = mask[y, :]
        edges = np.where(row > 0)[0]
        if len(edges) >= 2:
            diam = edges[-1] - edges[0]
            diameters.append(diam)

    if not diameters:
        continue

    frame_id = int(fname.replace("frame_", "").split(".")[0])
    avg_d = np.mean(diameters)
    max_d = np.max(diameters)
    min_d = np.min(diameters)
    area = np.sum(mask > 0)

    results.append({
        "frame": frame_id,
        "avg_diameter_px": avg_d,
        "max_diameter_px": max_d,
        "min_diameter_px": min_d,
        "area_px": area
    })

# --- Output to DataFrame ---
df = pd.DataFrame(results)
df = df.sort_values("frame")
df.to_csv("vein_parameters.csv", index=False)

print("✅ Extracted parameters saved to 'vein_parameters.csv'")
print(df.head())
