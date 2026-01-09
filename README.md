Block Blast RL Automation (iPhone Mirroring)

This folder contains a simple Python utility to calibrate the Block Blast tray and grid
positions from macOS iPhone Mirroring, then script piece placements.

Requirements
- macOS Accessibility permissions for Terminal/Python.
- Conda environment: blockblast (Python 3.12.2)

Install
conda create -y -n blockblast -c conda-forge python=3.12.2
conda activate blockblast
python -m pip install pynput pyautogui

Calibrate
conda activate blockblast
  python blockblast_calibration.py calibrate

You will be prompted to click:
1) Top-left of the 8x8 grid
2) Bottom-right of the 8x8 grid
3) Center of tray pieces 0, 1, 2
4) A safe focus point inside the iPhone Mirroring window
5) For each tray piece: click and hold, press Space to capture pickup cursor, then drag to (3,7) and press Space, then to (8,2) and press Space

Shape Class Calibration
- Edit the class list in `blockblast_rl.py` (CLASSES) to match your needs.
- Calibrate one class at a time (includes tray pickup/scale for that tray):
  python blockblast_calibration.py calibrate-class <class_name> <tray_index 0-2>
- See which classes still need calibration:
  python blockblast_calibration.py status
- Reset calibration data:
  python blockblast_calibration.py reset calibration
  python blockblast_calibration.py reset pair --class 3x3 --tray 0

Calibration Summary
- Calibration is two layers:
  1) Tray transform: how cursor motion maps to piece motion for a tray slot.
  2) Class offset: shape-specific pickup jump (constant offset) after the tray transform.
- `calibrate-class` does both for a single (class, tray) in one flow.
- `place` uses the tray transform + optional class offset (via `--class`).
  - Class offsets are stored per tray; calibrate the same class on each tray where it appears.

Place a Piece
conda activate blockblast
python blockblast_calibration.py place <tray_index 0-2> <row 0-7> <col 0-7> [--class <class_name>]

Example
python blockblast_calibration.py place 0 3 4

Status Processing (blockblast_status.py)
- Shared scaling: coordinates from calibration.json (pyautogui-space) are scaled to the mss capture size.
- Tray classification:
  - Capture each tray box from `tray_boxes` with mss.
  - Convert to grayscale and threshold vs background to get a mask.
  - Compute the bounding box of foreground pixels.
  - Sample each cell center inside that box to build a binary block grid (1 = filled).
- Combo detection:
  - Sample `background_pixel` and `combo_pixel`.
  - If the background is stable and the combo pixel differs from background, combo is active.
- Score OCR:
  - Capture `score_box` with mss.
  - Convert to PIL and run pytesseract with `--psm 7` + digit whitelist.
- Board state:
  - Use `board_box` to compute 8x8 cell centers.
  - Compare center pixel vs a lower pixel in the same cell to mark occupancy.

Notes
- The script uses drag-and-drop from the tray center to the target cell center.
- Failsafe: move the cursor to a screen corner to abort (pyautogui default).
