#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    import pyautogui
except ImportError:
    pyautogui = None


CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


def require_deps():
    missing = []
    if keyboard is None:
        missing.append("pynput")
    if pyautogui is None:
        missing.append("pyautogui")
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Install with: python -m pip install " + " ".join(missing))
        sys.exit(1)


def load_calibration():
    if not CALIBRATION_PATH.exists():
        print("Missing calibration.json. Run blockblast_calibration.py calibrate first.")
        sys.exit(1)
    return json.loads(CALIBRATION_PATH.read_text())


def wait_for_space(prompt):
    print(prompt)

    def on_press(k):
        if k == keyboard.Key.space or getattr(k, "char", None) == " ":
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    pos = pyautogui.position()
    return float(pos.x), float(pos.y)


def cell_center(cal, row, col):
    tl = cal["grid"]["tl"]
    br = cal["grid"]["br"]
    size = cal["grid_size"]
    cell_w = (br["x"] - tl["x"]) / size
    cell_h = (br["y"] - tl["y"]) / size
    cx = tl["x"] + (col + 0.5) * cell_w
    cy = tl["y"] + (row + 0.5) * cell_h
    return cx, cy


def fit_and_report(name, feats, targets):
    x = np.array(feats, dtype=float)
    y = np.array(targets, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeffs
    err = pred - y
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    total_rmse = np.sqrt(np.mean(err ** 2))
    print(f"{name} fit:")
    print(f"  rmse_x={rmse[0]:.3f} px, rmse_y={rmse[1]:.3f} px, rmse_total={total_rmse:.3f} px")


def cursor_for_target(cal, tray_index, class_name, row, col):
    transforms = cal.get("drag_transform", [])
    if tray_index >= len(transforms):
        raise ValueError(f"Missing drag_transform for tray {tray_index}.")
    transform = transforms[tray_index]
    ax = transform.get("ax", 0.0)
    bx = transform.get("bx", 1.0) or 1.0
    ay = transform.get("ay", 0.0)
    by = transform.get("by", 1.0) or 1.0

    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    tray_key = str(tray_index)
    offset = offsets_by_tray.get(tray_key, {}).get(class_name)
    if offset is None:
        raise ValueError(f"Missing class offset for tray {tray_index}, class '{class_name}'.")

    tx, ty = cell_center(cal, row, col)
    tx -= offset.get("x", 0.0)
    ty -= offset.get("y", 0.0)
    cx = (tx - ax) / bx
    cy = (ty - ay) / by
    return cx, cy


def fit_tray_mapping(cal, tray_a, tray_b):
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    shared = set(offsets_by_tray.get(str(tray_a), {})) & set(
        offsets_by_tray.get(str(tray_b), {})
    )
    if not shared:
        print(f"No shared class offsets between trays {tray_a} and {tray_b}.")
        return

    targets = [
        (0, 0),
        (0, 7),
        (7, 0),
        (7, 7),
        (0, 3),
        (3, 0),
        (7, 3),
        (3, 7),
        (3, 3),
    ]

    points_a = []
    points_b = []
    for class_name in sorted(shared):
        for row, col in targets:
            ca = cursor_for_target(cal, tray_a, class_name, row, col)
            cb = cursor_for_target(cal, tray_b, class_name, row, col)
            points_a.append(ca)
            points_b.append(cb)

    feats_affine = [[x, y, 1.0] for x, y in points_a]
    feats_quad = [[x, y, x * x, y * y, x * y, 1.0] for x, y in points_a]

    print(f"\nTray mapping {tray_a} -> {tray_b} using {len(shared)} shared class(es):")
    fit_and_report("Affine", feats_affine, points_b)
    fit_and_report("Quadratic", feats_quad, points_b)


def main():
    require_deps()
    cal = load_calibration()

    mode = input("Mode: (1) fit tray drag, (2) fit tray-to-tray mapping [1/2]: ").strip() or "1"
    if mode == "2":
        try:
            tray_a = int(input("Source tray (0-2): ").strip())
            tray_b = int(input("Target tray (0-2): ").strip())
        except ValueError:
            print("Invalid tray index.")
            sys.exit(1)
        fit_tray_mapping(cal, tray_a, tray_b)
        return

    try:
        tray_index = int(input("Tray index (0-2): ").strip())
    except ValueError:
        print("Invalid tray index.")
        sys.exit(1)
    if tray_index not in (0, 1, 2):
        print("Tray index must be 0, 1, or 2.")
        sys.exit(1)

    class_name = input("Block class label (e.g., 3x3) [optional]: ").strip()
    if not class_name:
        class_name = "unknown"

    targets = [
        (0, 0),
        (0, 7),
        (7, 0),
        (7, 7),
        (0, 3),
        (3, 0),
        (7, 3),
        (3, 7),
        (3, 3),
    ]

    print(
        "\nHold a block from the tray, then for each target cell move it so the piece is centered\n"
        "on that grid cell and press Space to capture the cursor position."
    )

    cursor_points = []
    target_points = []
    for row, col in targets:
        cx, cy = cell_center(cal, row, col)
        prompt = f"Target cell ({row},{col}) -> press Space when centered."
        mx, my = wait_for_space(prompt)
        cursor_points.append((mx, my))
        target_points.append((cx, cy))

    feats_affine = [[x, y, 1.0] for x, y in cursor_points]
    feats_quad = [
        [x, y, x * x, y * y, x * y, 1.0] for x, y in cursor_points
    ]

    print("\nResults for tray", tray_index, "class", class_name)
    fit_and_report("Affine", feats_affine, target_points)
    fit_and_report("Quadratic", feats_quad, target_points)


if __name__ == "__main__":
    main()
