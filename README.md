# ASAIO2025 — Ultrasound Vein Segmentation + STL Generation

End-to-end scripts for turning ultrasound video into annotated edge masks, training a
U-Net edge detector, visualizing predictions, and generating a 3D STL vein model.

## What’s here

- Extract frames from an ultrasound video.
- Manually annotate edges and convert them to binary masks.
- Train an edge-detection model (`UNetEdgeDetector`).
- Overlay predictions on video or preview masks.
- Generate STL geometry from binary masks or from extracted parameters.

## Project layout (key folders)

- `images/` — extracted frames (input for training)
- `masks_annotated/` — manual annotations (white/green lines)
- `masks_binary/` — binary edge masks (training labels)
- `masks_binary2/` — alternate binary masks (green threshold)
- `stls/` — generated STL outputs

## Setup

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```
pip install numpy pandas pillow matplotlib opencv-python torch torchvision
pip install scikit-image scipy numpy-stl trimesh
```

If you need a CUDA build of PyTorch, install from `https://pytorch.org/` instead of
`pip install torch torchvision`.

## Typical workflow

1. **Extract frames from video**
```
python extractframes.py
```
Edit `video_path` and `output_folder` at the top of the script if needed.

2. **Annotate frames**
Use your preferred tool to draw edges in `masks_annotated/`.

3. **Convert annotations to binary masks**
```
python binarymasks.py
```
Use `binarymasks2.py` if you annotated in green.

4. **Train the edge detector**
```
python trainmodel.py
```
Outputs `trained_edge_model.pth`.

5. **Preview predictions**
```
python predicted_view.py
```
Or generate an overlayed video:
```
python annotatevideo.py
```

6. **Generate STL**
Option A: volumetric STL from masks:
```
python merge.py
```
Option B: parameterized STL from CSV profile:
```
python extract.py
python makemodel.py
```

## GUI launcher

There is a simple Tkinter GUI in `mastergui.py` for selecting a video and output path.
It currently calls a placeholder script name for STL generation; update it to your
actual pipeline script (for example `merge.py`).

## Batch script

For a minimal pipeline run:
```
master.bat
```

## Notes

- Most scripts have editable configuration at the top (paths, thresholds, sizes).
- Large data files (videos, images) are included in this repo; adjust paths as needed.
