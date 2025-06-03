import os
import cv2
import numpy as np

# --- Folder paths ---
input_folder = 'images2'         # Input folder with green-annotated images
output_folder = 'masks_binary2'  # Output folder for binary masks

os.makedirs(output_folder, exist_ok=True)

# --- Define HSV range for green ---
# This range detects bright green; adjust if your annotation is a different shade
LOWER_GREEN = np.array([40, 40, 40])     # H, S, V
UPPER_GREEN = np.array([90, 255, 255])

# --- Process each annotated mask ---
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img_bgr = cv2.imread(img_path)

        # Convert to HSV for better color detection
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Mask the green regions
        green_mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)

        # Save result to output folder
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, green_mask)

        print(f"✅ Converted: {filename}")

print("\n🎉 All green-annotated images converted to binary masks.")
