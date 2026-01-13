#!/usr/bin/env python3
import argparse
import atexit
import json
import sys
import time
import threading
from pathlib import Path

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")
INFERENCE_CACHE_PATH = CALIBRATION_PATH.with_name("calibration_inference.json")
CLASSES = [
    "1x1",
    "1x2",
    "2x1",
    "2x2",
    "2x3",
    "3x2",
    "4x1",
    "1x4",
    "1x5",
    "5x1",
    "3x3",
    "3x1",
    "1x3",
]

try:
    from pynput import mouse
except ImportError:
    mouse = None

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    import pyautogui
except ImportError:
    pyautogui = None
try:
    import numpy as np
except ImportError:
    np = None


def require_deps():
    missing = []
    if mouse is None:
        missing.append("pynput")
    if keyboard is None:
        missing.append("pynput")
    if pyautogui is None:
        missing.append("pyautogui")
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)


def _safe_mouse_up():
    if pyautogui is None:
        return
    try:
        pyautogui.mouseUp()
    except Exception:
        pass


atexit.register(_safe_mouse_up)


def _start_live_cursor(label):
    stop_event = threading.Event()
    pos = {"x": 0, "y": 0}

    def on_move(x, y):
        pos["x"] = x
        pos["y"] = y

    listener = mouse.Listener(on_move=on_move)
    listener.start()

    def printer():
        while not stop_event.is_set():
            sys.stdout.write(
                f"\r[{label}] cursor=({pos['x']:.0f}, {pos['y']:.0f})"
            )
            sys.stdout.flush()
            time.sleep(0.05)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    thread = threading.Thread(target=printer, daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        listener.stop()

    return stop


def wait_for_click(prompt, live_label=None):
    print(prompt)
    pos = {}
    stop_live = _start_live_cursor(live_label) if live_label else None

    def on_click(x, y, button, pressed):
        if pressed:
            pos["x"] = x
            pos["y"] = y
            return False

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    if stop_live:
        stop_live()
        print()
    return pos["x"], pos["y"]


def wait_for_keypress(prompt, key=None, live_label=None):
    print(prompt)
    if key is None:
        key = keyboard.Key.space
    stop_live = _start_live_cursor(live_label) if live_label else None

    def on_press(k):
        if k == key or getattr(k, "char", None) == " ":
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    pos = pyautogui.position()
    if stop_live:
        stop_live()
        print()
    return pos.x, pos.y


def calibrate_tray_piece(tray_index, tl, br):
    targets = [(2, 6), (7, 1)]
    grid_tmp = {"grid": {"tl": {"x": tl[0], "y": tl[1]}, "br": {"x": br[0], "y": br[1]}}, "grid_size": 8}
    print(
        f"Drag scale calibration for tray piece {tray_index}: "
        "click and HOLD the piece, then press Space to capture the pickup cursor."
    )
    px, py = wait_for_keypress(
        "Press Space to capture the pickup cursor position.",
        live_label=f"tray {tray_index} pickup",
    )
    print(
        f"Now move it so it is centered on grid cell "
        f"({targets[0][0]},{targets[0][1]}), then press Space (keep holding the mouse)."
    )
    cx1, cy1 = wait_for_keypress(
        "Press Space to capture the cursor position.",
        live_label=f"tray {tray_index} target 1",
    )
    print(
        f"Now move the SAME piece so it is centered on grid cell "
        f"({targets[1][0]},{targets[1][1]}), then press Space."
    )
    cx2, cy2 = wait_for_keypress(
        "Press Space to capture the cursor position.",
        live_label=f"tray {tray_index} target 2",
    )
    start = (px, py)
    t1 = cell_center(grid_tmp, targets[0][0], targets[0][1])
    t2 = cell_center(grid_tmp, targets[1][0], targets[1][1])

    def axis_transform(t1_v, t2_v, c1_v, c2_v):
        denom = c2_v - c1_v
        if abs(denom) < 1e-3:
            return 0.0, 1.0
        b = (t2_v - t1_v) / denom
        a = t1_v - b * c1_v
        return a, b

    ax, bx = axis_transform(t1[0], t2[0], cx1, cx2)
    ay, by = axis_transform(t1[1], t2[1], cy1, cy2)
    return {"x": px, "y": py}, {"ax": ax, "bx": bx, "ay": ay, "by": by}


def calibrate():
    require_deps()
    print("Calibration starting in 2 seconds. Focus the iPhone Mirroring window.")
    time.sleep(2)

    tl = wait_for_click("Click the TOP-LEFT corner of the 8x8 grid.", "grid tl")
    br = wait_for_click("Click the BOTTOM-RIGHT corner of the 8x8 grid.", "grid br")

    tray = []
    for i in range(3):
        tray.append(wait_for_click(f"Click the CENTER of tray piece {i}.", f"tray {i} center"))

    focus = wait_for_click("Click a SAFE spot inside the iPhone Mirroring window to focus it.", "focus")

    drag_transform = []
    pickups = []
    for i in range(3):
        pickup, transform = calibrate_tray_piece(i, tl, br)
        pickups.append(pickup)
        drag_transform.append(transform)

    data = {
        "grid": {"tl": {"x": tl[0], "y": tl[1]}, "br": {"x": br[0], "y": br[1]}},
        "tray": [{"x": x, "y": y} for x, y in tray],
        "grid_size": 8,
        "focus": {"x": focus[0], "y": focus[1]},
        "pickups": pickups,
        "drag_transform": drag_transform,
        "drag_scale_targets": [{"row": r, "col": c} for r, c in [(2, 6), (7, 1)]],
    }

    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))
    print(f"Saved calibration to {CALIBRATION_PATH}")


def load_calibration():
    if not CALIBRATION_PATH.exists():
        print("No calibration found. Run: python blockblast_rl.py calibrate")
        sys.exit(1)
    return json.loads(CALIBRATION_PATH.read_text())


def load_inference_cache():
    if not INFERENCE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(INFERENCE_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_inference_cache(cache):
    def _to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    INFERENCE_CACHE_PATH.write_text(json.dumps(_to_jsonable(cache), indent=2))


def cell_center(cal, row, col):
    tl = cal["grid"]["tl"]
    br = cal["grid"]["br"]
    size = cal["grid_size"]
    cell_w = (br["x"] - tl["x"]) / size
    cell_h = (br["y"] - tl["y"]) / size
    cx = tl["x"] + (col + 0.5) * cell_w
    cy = tl["y"] + (row + 0.5) * cell_h
    return cx, cy


TRAY_MAP_TARGETS = [
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


def _parse_class_dims(class_name):
    w, h = class_name.split("x")
    return int(w), int(h)


def _cursor_for_target(cal, tray_index, class_name, row, col):
    transforms = cal.get("drag_transform", [])
    if tray_index >= len(transforms):
        return None
    transform = transforms[tray_index]
    ax = transform.get("ax", 0.0)
    bx = transform.get("bx", 1.0) or 1.0
    ay = transform.get("ay", 0.0)
    by = transform.get("by", 1.0) or 1.0

    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
    tray_key = str(tray_index)
    offset = offsets_by_tray.get(tray_key, {}).get(class_name)
    if offset is None:
        offset = inferred_by_tray.get(tray_key, {}).get(class_name)
    if offset is None:
        return None

    tx, ty = cell_center(cal, row, col)
    tx -= offset.get("x", 0.0)
    ty -= offset.get("y", 0.0)
    cx = (tx - ax) / bx
    cy = (ty - ay) / by
    return cx, cy


def _fit_affine(points_src, points_dst):
    if np is None:
        return None
    a = np.array([[x, y, 1.0] for x, y in points_src], dtype=float)
    b = np.array(points_dst, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return coeffs


def _apply_affine(coeffs, x, y):
    if coeffs is None:
        return None
    vec = np.array([x, y, 1.0], dtype=float)
    out = vec @ coeffs
    return float(out[0]), float(out[1])


def _build_tray_mapping(cal, tray_a, tray_b, cache=None):
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
    shared = set(offsets_by_tray.get(str(tray_a), {})) & set(
        offsets_by_tray.get(str(tray_b), {})
    )
    if not shared:
        shared = set(inferred_by_tray.get(str(tray_a), {})) & set(
            inferred_by_tray.get(str(tray_b), {})
        )
    if not shared and cache:
        key = f"{tray_a}->{tray_b}"
        cached = cache.get("tray_mappings", {}).get(key)
        if cached:
            return cached
    if not shared:
        return None

    points_a = []
    points_b = []
    for class_name in sorted(shared):
        for row, col in TRAY_MAP_TARGETS:
            ca = _cursor_for_target(cal, tray_a, class_name, row, col)
            cb = _cursor_for_target(cal, tray_b, class_name, row, col)
            if ca is None or cb is None:
                continue
            points_a.append(ca)
            points_b.append(cb)

    if len(points_a) < 3:
        return None
    return _fit_affine(points_a, points_b)


def _build_size_offset_models(cal):
    if np is None:
        return {}
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
    models = {}
    for tray_index in [0, 1, 2]:
        tray_key = str(tray_index)
        data = {}
        data.update(offsets_by_tray.get(tray_key, {}))
        data.update(inferred_by_tray.get(tray_key, {}))
        if len(data) < 3:
            continue
        feats = []
        targets = []
        for class_name, offset in data.items():
            w, h = _parse_class_dims(class_name)
            feats.append([w, h, 1.0])
            targets.append([offset.get("x", 0.0), offset.get("y", 0.0)])
        x = np.array(feats, dtype=float)
        y = np.array(targets, dtype=float)
        coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        models[tray_key] = {
            "ax": float(coeffs[0, 0]),
            "bx": float(coeffs[1, 0]),
            "cx": float(coeffs[2, 0]),
            "ay": float(coeffs[0, 1]),
            "by": float(coeffs[1, 1]),
            "cy": float(coeffs[2, 1]),
        }
    return models


def _infer_offset_from_size_model(tray_index, class_name, size_models):
    tray_key = str(tray_index)
    model = size_models.get(tray_key)
    if not model:
        return None
    w, h = _parse_class_dims(class_name)
    x = model["ax"] * w + model["bx"] * h + model["cx"]
    y = model["ay"] * w + model["by"] * h + model["cy"]
    return {"x": x, "y": y, "inferred_from": "size_fit"}


def build_inference_cache(cal):
    cache = {"tray_mappings": {}, "size_offset_models": {}, "generated_at": time.time()}
    for tray_a in [0, 1, 2]:
        for tray_b in [0, 1, 2]:
            if tray_a == tray_b:
                continue
            mapping = _build_tray_mapping(cal, tray_a, tray_b)
            if mapping is not None:
                cache["tray_mappings"][f"{tray_a}->{tray_b}"] = mapping
    cache["size_offset_models"] = _build_size_offset_models(cal)
    return cache


def _infer_offset_from_other_tray(cal, tray_index, class_name, cache=None):
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
    candidates = [
        int(k) for k, v in offsets_by_tray.items() if class_name in v
    ]
    if not candidates:
        candidates = [
            int(k) for k, v in inferred_by_tray.items() if class_name in v
        ]
    for source_tray in candidates:
        if source_tray == tray_index:
            continue
        mapping = _build_tray_mapping(cal, source_tray, tray_index, cache=cache)
        if mapping is None:
            continue
        source_cursor = _cursor_for_target(
            cal, source_tray, class_name, *TRAY_MAP_TARGETS[-1]
        )
        if source_cursor is None:
            continue
        dest_cursor = _apply_affine(mapping, *source_cursor)
        if dest_cursor is None:
            continue
        transforms = cal.get("drag_transform", [])
        if tray_index >= len(transforms):
            continue
        transform = transforms[tray_index]
        ax = transform.get("ax", 0.0)
        bx = transform.get("bx", 1.0) or 1.0
        ay = transform.get("ay", 0.0)
        by = transform.get("by", 1.0) or 1.0
        tx, ty = cell_center(cal, *TRAY_MAP_TARGETS[-1])
        achieved_x = ax + bx * dest_cursor[0]
        achieved_y = ay + by * dest_cursor[1]
        return {
            "x": tx - achieved_x,
            "y": ty - achieved_y,
            "inferred_from": source_tray,
        }
    return None


def infer_missing_offsets(cal, classes=None, trays=None):
    if np is None:
        print("numpy is required for inference. Install with: python -m pip install numpy")
        return cal, 0
    classes = classes or CLASSES
    trays = trays if trays is not None else [0, 1, 2]
    cache = load_inference_cache()
    size_models = cache.get("size_offset_models") or _build_size_offset_models(cal)
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
    updated = 0
    for tray_index in trays:
        tray_key = str(tray_index)
        if tray_key not in offsets_by_tray:
            offsets_by_tray[tray_key] = {}
        if tray_key not in inferred_by_tray:
            inferred_by_tray[tray_key] = {}
        for class_name in classes:
            if class_name in offsets_by_tray[tray_key]:
                continue
            inferred = _infer_offset_from_other_tray(cal, tray_index, class_name, cache=cache)
            if inferred is None:
                inferred = _infer_offset_from_size_model(tray_index, class_name, size_models)
            if inferred is None:
                continue
            inferred_by_tray[tray_key][class_name] = inferred
            updated += 1
    cal["class_offsets_by_tray"] = offsets_by_tray
    cal["class_offsets_by_tray_inferred"] = inferred_by_tray
    save_inference_cache(build_inference_cache(cal))
    return cal, updated


def drag_piece(tray_index, row, col, duration=0.1, class_name=None, debug=False):
    require_deps()
    cal = load_calibration()

    tray = cal["tray"][tray_index]
    pickups = cal.get("pickups", [])
    pickup = pickups[tray_index] if tray_index < len(pickups) else tray
    dest = cell_center(cal, row, col)
    start_x, start_y = pickup["x"], pickup["y"]
    base_dest = dest
    transforms = cal.get("drag_transform", [])
    transform = transforms[tray_index] if tray_index < len(transforms) else None
    drag_scales = cal.get("drag_scales", [])
    scale = drag_scales[tray_index] if tray_index < len(drag_scales) else None
    sx = None
    sy = None
    if transform:
        ax = transform.get("ax", 0.0)
        bx = transform.get("bx", 1.0) or 1.0
        ay = transform.get("ay", 0.0)
        by = transform.get("by", 1.0) or 1.0
    elif scale:
        ax = 0.0
        ay = 0.0
        bx = 1.0
        by = 1.0
        sx = scale.get("x", 1.0) or 1.0
        sy = scale.get("y", 1.0) or 1.0
    else:
        ax = 0.0
        ay = 0.0
        bx = 1.0
        by = 1.0
    offset = None
    if class_name:
        offsets_by_tray = cal.get("class_offsets_by_tray", {})
        inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
        tray_key = str(tray_index)
        if tray_key in offsets_by_tray and class_name in offsets_by_tray[tray_key]:
            offset = offsets_by_tray[tray_key][class_name]
        elif tray_key in inferred_by_tray and class_name in inferred_by_tray[tray_key]:
            offset = inferred_by_tray[tray_key][class_name]
        else:
            cache = load_inference_cache()
            offset = _infer_offset_from_other_tray(cal, tray_index, class_name, cache=cache)
            if offset is None:
                size_models = cache.get("size_offset_models") or _build_size_offset_models(cal)
                offset = _infer_offset_from_size_model(tray_index, class_name, size_models)
    if offset:
        dest = (dest[0] - offset.get("x", 0), dest[1] - offset.get("y", 0))
    if transform:
        dest = ((dest[0] - ax) / bx, (dest[1] - ay) / by)
    elif scale:
        dest = (start_x + (dest[0] - start_x) / sx, start_y + (dest[1] - start_y) / sy)
    else:
        dest = (start_x + (dest[0] - start_x), start_y + (dest[1] - start_y))
    focus = cal.get("focus")

    if debug:
        print("Debug placement:")
        print(f"- tray_index: {tray_index}")
        print(f"- class: {class_name}")
        print(f"- pickup: ({start_x:.2f}, {start_y:.2f})")
        print(f"- target cell center: ({base_dest[0]:.2f}, {base_dest[1]:.2f})")
        if transform:
            print(f"- transform: ax={ax:.2f}, bx={bx:.4f}, ay={ay:.2f}, by={by:.4f}")
        elif scale:
            print(f"- scale: x={sx:.4f}, y={sy:.4f}")
        if offset:
            print(f"- class offset: ({offset.get('x', 0):.2f}, {offset.get('y', 0):.2f})")
        print(f"- final cursor dest: ({dest[0]:.2f}, {dest[1]:.2f})")

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02
    if focus:
        pyautogui.click(focus["x"], focus["y"])
        time.sleep(0.05)
    pyautogui.moveTo(pickup["x"], pickup["y"])
    pyautogui.dragTo(dest[0], dest[1], duration=duration, button="left")


def calibrate_class_offset(class_name, tray_index):
    require_deps()
    if class_name not in CLASSES:
        print(f"Unknown class '{class_name}'. Add it to CLASSES in blockblast_rl.py.")
        sys.exit(1)
    cal = load_calibration()
    tl = (cal["grid"]["tl"]["x"], cal["grid"]["tl"]["y"])
    br = (cal["grid"]["br"]["x"], cal["grid"]["br"]["y"])
    pickup, transform = calibrate_tray_piece(tray_index, tl, br)
    pickups = cal.get("pickups", [])
    while len(pickups) < 3:
        pickups.append({"x": 0, "y": 0})
    pickups[tray_index] = pickup
    transforms = cal.get("drag_transform", [])
    while len(transforms) < 3:
        transforms.append({"ax": 0.0, "bx": 1.0, "ay": 0.0, "by": 1.0})
    transforms[tray_index] = transform
    cal["pickups"] = pickups
    cal["drag_transform"] = transforms
    cal["drag_scale_targets"] = [{"row": r, "col": c} for r, c in [(2, 6), (7, 1)]]
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    print(f"Updated tray {tray_index} calibration in {CALIBRATION_PATH}")

    targets = (3, 3)
    t = cell_center({"grid": {"tl": {"x": tl[0], "y": tl[1]}, "br": {"x": br[0], "y": br[1]}}, "grid_size": 8}, *targets)
    transform = cal.get("drag_transform", [])[tray_index]
    ax = transform.get("ax", 0.0)
    bx = transform.get("bx", 1.0) or 1.0
    ay = transform.get("ay", 0.0)
    by = transform.get("by", 1.0) or 1.0

    print(
        f"Class offset calibration for '{class_name}' (tray {tray_index}): "
        "keep holding the same piece, then move it to the target cell."
    )
    print(
        f"Now move it so it is centered on grid cell "
        f"({targets[0]},{targets[1]}), then press Space."
    )
    cx, cy = wait_for_keypress(
        "Press Space to capture the cursor position.",
        live_label=f"class {class_name} target",
    )
    px, py = pickup["x"], pickup["y"]
    achieved_x = ax + bx * cx
    achieved_y = ay + by * cy
    offset = {"x": t[0] - achieved_x, "y": t[1] - achieved_y}
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    tray_key = str(tray_index)
    if tray_key not in offsets_by_tray:
        offsets_by_tray[tray_key] = {}
    offsets_by_tray[tray_key][class_name] = offset
    cal["class_offsets_by_tray"] = offsets_by_tray
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    print(f"Saved class offset for '{class_name}' in {CALIBRATION_PATH}")


def status():
    cal = load_calibration()
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
    pickups = cal.get("pickups", [])
    transforms = cal.get("drag_transform", [])

    def tray_ok(idx):
        has_pickup = idx < len(pickups) and pickups[idx].get("x", 0) and pickups[idx].get("y", 0)
        has_transform = idx < len(transforms) and any(
            transforms[idx].get(k, 0) for k in ("ax", "bx", "ay", "by")
        )
        return bool(has_pickup and has_transform)
    any_tray = set()
    for tray_classes in offsets_by_tray.values():
        any_tray.update(tray_classes.keys())
    for tray_classes in inferred_by_tray.values():
        any_tray.update(tray_classes.keys())
    missing_any = [c for c in CLASSES if c not in any_tray]

    print("Class calibration status (any tray):")
    print(f"- Total classes: {len(CLASSES)}")
    print(f"- Calibrated: {len(CLASSES) - len(missing_any)}")
    if any_tray:
        print("- Calibrated list:")
        for c in sorted(any_tray):
            print(f"  - {c}")
    if missing_any:
        print("- Missing:")
        for c in missing_any:
            print(f"  - {c}")
    else:
        print("- All classes calibrated.")

    print("Class calibration status (by tray):")
    for i in range(3):
        tray_key = str(i)
        tray_classes = offsets_by_tray.get(tray_key, {})
        inferred_classes = inferred_by_tray.get(tray_key, {})
        missing_tray = [c for c in CLASSES if c not in tray_classes]
        print(
            f"- Tray {i}: {len(CLASSES) - len(missing_tray)}/{len(CLASSES)} calibrated"
        )
        if tray_classes:
            print("  Calibrated:")
            for c in sorted(tray_classes.keys()):
                print(f"    - {c}")
        if inferred_classes:
            print("  Inferred:")
            for c in sorted(inferred_classes.keys()):
                print(f"    - {c}")
        if missing_tray:
            print("  Missing:")
            for c in missing_tray:
                print(f"    - {c}")


def reset_calibration(scope, class_name=None, tray_index=None):
    cal = load_calibration()
    if scope == "calibration":
        cal["class_offsets_by_tray"] = {}
        cal["class_offsets_by_tray_inferred"] = {}
        cal["pickups"] = []
        cal["drag_scales"] = []
        cal["drag_transform"] = []
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
        print(f"Cleared class + tray calibration in {CALIBRATION_PATH}")
        return
    if scope == "pair":
        if not class_name or tray_index is None:
            print("Missing class name or tray index for reset pair.")
            return
        offsets_by_tray = cal.get("class_offsets_by_tray", {})
        inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
        tray_key = str(tray_index)
        if tray_key in offsets_by_tray and class_name in offsets_by_tray[tray_key]:
            del offsets_by_tray[tray_key][class_name]
            cal["class_offsets_by_tray"] = offsets_by_tray
        if tray_key in inferred_by_tray and class_name in inferred_by_tray[tray_key]:
            del inferred_by_tray[tray_key][class_name]
            cal["class_offsets_by_tray_inferred"] = inferred_by_tray
        pickups = cal.get("pickups", [])
        scales = cal.get("drag_scales", [])
        transforms = cal.get("drag_transform", [])
        if tray_index < len(pickups):
            pickups[tray_index] = {"x": 0, "y": 0}
        if tray_index < len(scales):
            scales[tray_index] = {"x": 1.0, "y": 1.0}
        if tray_index < len(transforms):
            transforms[tray_index] = {"ax": 0.0, "bx": 1.0, "ay": 0.0, "by": 1.0}
        cal["pickups"] = pickups
        cal["drag_scales"] = scales
        cal["drag_transform"] = transforms
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
        print(
            f"Cleared class '{class_name}' and tray {tray_index} in {CALIBRATION_PATH}"
        )
        return
    if scope == "class":
        if not class_name or tray_index is None:
            print("Missing class name or tray index for class reset.")
            return
        offsets_by_tray = cal.get("class_offsets_by_tray", {})
        inferred_by_tray = cal.get("class_offsets_by_tray_inferred", {})
        tray_key = str(tray_index)
        if tray_key in offsets_by_tray and class_name in offsets_by_tray[tray_key]:
            del offsets_by_tray[tray_key][class_name]
            cal["class_offsets_by_tray"] = offsets_by_tray
        if tray_key in inferred_by_tray and class_name in inferred_by_tray[tray_key]:
            del inferred_by_tray[tray_key][class_name]
            cal["class_offsets_by_tray_inferred"] = inferred_by_tray
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
        print(
            f"Cleared class '{class_name}' for tray {tray_index} in {CALIBRATION_PATH}"
        )
        return
    print("Unknown reset scope. Use: calibration, pair.")


def main():
    parser = argparse.ArgumentParser(description="Block Blast RL automation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("calibrate", help="Full calibration (grid, tray, focus, scales).")

    class_parser = sub.add_parser("calibrate-class", help="Calibrate class offset.")
    class_parser.add_argument("class_name", type=str, help="Class name from CLASSES.")
    class_parser.add_argument("tray_index", type=int, choices=[0, 1, 2], help="Tray index 0-2.")

    place_parser = sub.add_parser("place", help="Place a piece.")
    place_parser.add_argument("tray_index", type=int, choices=[0, 1, 2], help="Tray index 0-2.")
    place_parser.add_argument("row", type=int, help="Row 0-7.")
    place_parser.add_argument("col", type=int, help="Col 0-7.")
    place_parser.add_argument("--class", dest="class_name", type=str, help="Optional class name.")
    place_parser.add_argument("--debug", action="store_true", help="Print computed placement info.")

    sub.add_parser("status", help="Show class calibration status.")
    reset_parser = sub.add_parser("reset", help="Reset calibration data.")
    reset_parser.add_argument(
        "scope",
        choices=["calibration", "pair", "class"],
        help="What to reset.",
    )
    reset_parser.add_argument("--class", dest="class_name", type=str, help="Class name to reset.")
    reset_parser.add_argument("--tray", dest="tray_index", type=int, choices=[0, 1, 2], help="Tray index to reset.")

    sub.add_parser("infer", help="Infer missing class offsets from other trays.")
    sub.add_parser(
        "cache",
        help="Save tray mappings and size-fit models for future inference.",
    )

    args = parser.parse_args()

    if args.cmd == "calibrate":
        calibrate()
        return
    if args.cmd == "calibrate-class":
        calibrate_class_offset(args.class_name, args.tray_index)
        return
    if args.cmd == "place":
        if not (0 <= args.row <= 7 and 0 <= args.col <= 7):
            print("row/col must be between 0 and 7.")
            sys.exit(1)
        drag_piece(
            args.tray_index,
            args.row,
            args.col,
            class_name=args.class_name,
            debug=args.debug,
        )
        return
    if args.cmd == "status":
        status()
        return
    if args.cmd == "reset":
        tray_index = args.tray_index if args.tray_index is not None else None
        reset_calibration(args.scope, class_name=args.class_name, tray_index=tray_index)
        return
    if args.cmd == "infer":
        cal = load_calibration()
        cal, updated = infer_missing_offsets(cal)
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
        print(f"Inferred {updated} missing offset(s) into {CALIBRATION_PATH}")
        return
    if args.cmd == "cache":
        cal = load_calibration()
        cache = build_inference_cache(cal)
        save_inference_cache(cache)
        print(f"Saved inference cache to {INFERENCE_CACHE_PATH}")
        return


if __name__ == "__main__":
    main()
