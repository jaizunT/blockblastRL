import json
import mss
import pytesseract
import time
import pyautogui
from PIL import Image
import numpy as np

"""Utilities for reading BlockBlast screen status."""

# Load calibration data.
with open("calibration.json") as f:
    CALIBRATION = json.load(f)

# Match pixel_locate scaling (mss monitor vs pyautogui coords).
sct = mss.mss()
scale_x = 1.0
scale_y = 1.0
try:
    mon = sct.monitors[1]
    screen_w = mon["width"]
    screen_h = mon["height"]
    mouse_w, mouse_h = pyautogui.size()
    if mouse_w and mouse_h:
        scale_x = screen_w / mouse_w
        scale_y = screen_h / mouse_h
except Exception:
    pass

_SNAPSHOT = None


def screenshot():
    """Capture the full screen once and cache it for reuse."""
    global _SNAPSHOT
    mon = sct.monitors[1]
    shot = sct.grab(mon)
    img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
        mon["height"], mon["width"], 3
    )
    _SNAPSHOT = {
        "img": img,
        "left": mon["left"],
        "top": mon["top"],
        "width": mon["width"],
        "height": mon["height"],
    }
    return _SNAPSHOT


def _get_snapshot():
    return _SNAPSHOT if _SNAPSHOT is not None else screenshot()


def _scaled_point(x, y):
    return int(x * scale_x), int(y * scale_y)


def sample_pixel(x, y, snapshot=None):
    snapshot = _get_snapshot() if snapshot is None else snapshot
    left, top = _scaled_point(x, y)
    rel_x = left - snapshot["left"]
    rel_y = top - snapshot["top"]
    if (
        rel_x < 0
        or rel_y < 0
        or rel_x >= snapshot["width"]
        or rel_y >= snapshot["height"]
    ):
        return (0, 0, 0)
    r, g, b = snapshot["img"][rel_y, rel_x]
    return int(r), int(g), int(b)

# ---------- Classification of blocks ----------

# Screenshot trays
tray0_tl = CALIBRATION["tray_boxes"]["0"]["tl"]
tray0_br = CALIBRATION["tray_boxes"]["0"]["br"]
tray1_tl = CALIBRATION["tray_boxes"]["1"]["tl"]
tray1_br = CALIBRATION["tray_boxes"]["1"]["br"]
tray2_tl = CALIBRATION["tray_boxes"]["2"]["tl"]
tray2_br = CALIBRATION["tray_boxes"]["2"]["br"]

def sample_tray(tray_tl, tray_br, snapshot=None):
    snapshot = _get_snapshot() if snapshot is None else snapshot
    left, top = tray_tl['x'], tray_tl['y']
    right, bottom = tray_br['x'], tray_br['y']

    # Adjust for scaling
    left, top = _scaled_point(left, top)
    right, bottom = _scaled_point(right, bottom)

    width = right - left
    height = bottom - top
    rel_x = left - snapshot["left"]
    rel_y = top - snapshot["top"]
    tray_img = snapshot["img"][rel_y:rel_y + height, rel_x:rel_x + width]

    # Convert to gray scale for easier processing
    img_array = (
        0.299 * tray_img[..., 0]
        + 0.587 * tray_img[..., 1]
        + 0.114 * tray_img[..., 2]
    ).astype(np.uint8)

    # Get max and min for bounding box
    bg_pixel = img_array[0,0]
    bg = int(bg_pixel)
    mask = np.abs(img_array.astype(np.int16) - bg) > 10

    cols_with_fg = np.where(mask.any(axis=0))[0]
    rows_with_fg = np.where(mask.any(axis=1))[0]
    if len(cols_with_fg) == 0 or len(rows_with_fg) == 0:
        return None

    x0, x1 = cols_with_fg[0], cols_with_fg[-1]
    y0, y1 = rows_with_fg[0], rows_with_fg[-1] 
    cropped = img_array[y0:y1+1, x0:x1+1]

    # Classify blocks into rectangular classes
    h, w = cropped.shape
    CELL_LENGTH = 16

    block_h = int(np.round(h / CELL_LENGTH))
    block_w = int(np.round(w / CELL_LENGTH))

    block = np.zeros((block_h, block_w), dtype=int)

    # Fill in block shape using pixel at center of each cell using thresholding
    for i in range(block_h):
        for j in range(block_w):
            sample = int(cropped[i * CELL_LENGTH + CELL_LENGTH // 2, j * CELL_LENGTH + CELL_LENGTH // 2])
            if abs(sample - bg) > 10:
                block[i][j] = 1
    
    return block

def classify_tray(tray_index, snapshot=None):
    if tray_index == 0:
        tray_tl, tray_br = tray0_tl, tray0_br
    elif tray_index == 1:
        tray_tl, tray_br = tray1_tl, tray1_br
    elif tray_index == 2:
        tray_tl, tray_br = tray2_tl, tray2_br
    else:
        raise ValueError("Invalid tray index")
    
    block = sample_tray(tray_tl, tray_br, snapshot=snapshot)
    return block

def classify_all_trays(snapshot=None):
    snapshot = _get_snapshot() if snapshot is None else snapshot
    blocks = []
    for tray_index in range(3):
        block = classify_tray(tray_index, snapshot=snapshot)
        blocks.append(block)
    return blocks

def print_block(block):
    for row in block:
        print("".join(['#' if cell else '.' for cell in row]))
    print()



# ---------- Combo status ----------

# Check pixel value at background area
bg_pixel = CALIBRATION["background_pixel"]
# Check pixel value at combo area
combo_pixel = CALIBRATION["combo_pixel"]

def get_bg_pixel():
    return sample_pixel(bg_pixel['x'], bg_pixel['y'])
def background_stable():
    snap1 = screenshot()
    sampled_pixel = sample_pixel(bg_pixel['x'], bg_pixel['y'], snapshot=snap1)
    time.sleep(0.02)
    # If background pixel matches same value over multiple samples, it's stable
    snap2 = screenshot()
    return sampled_pixel == sample_pixel(bg_pixel['x'], bg_pixel['y'], snapshot=snap2)

def is_in_combo(snapshot=None):
    snapshot = _get_snapshot() if snapshot is None else snapshot
    sampled_pixel = sample_pixel(combo_pixel['x'], combo_pixel['y'], snapshot=snapshot)
    background_sampled = sample_pixel(bg_pixel['x'], bg_pixel['y'], snapshot=snapshot)
    # If combo pixel differs from background pixel, we are in combo
    return sampled_pixel != background_sampled

# ---------- Score status ----------

# Screenshot score area
score_tl = CALIBRATION["score_box"]["tl"]
score_br = CALIBRATION["score_box"]["br"]

# OCR to read score value
def get_score(snapshot=None):
    snapshot = _get_snapshot() if snapshot is None else snapshot
    left, top = score_tl['x'],score_tl['y']
    right, bottom = score_br['x'], score_br['y']
    # Adjust for scaling
    left, top = _scaled_point(left, top)
    right, bottom = _scaled_point(right, bottom)

    width = right - left
    height = bottom - top
    rel_x = left - snapshot["left"]
    rel_y = top - snapshot["top"]
    score_img = snapshot["img"][rel_y:rel_y + height, rel_x:rel_x + width]
    # OCR processing
    img = Image.fromarray(score_img, mode="RGB")
    text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    text = text.strip()
    try:
        score_value = int(text)
        return score_value
    except ValueError:
        return 0
    
def print_score():
    score = get_score()
    print(f"Current score: {score}")
# ---------- Board state ----------

# Read board area screenshot
board_tl = CALIBRATION["board_box"]["tl"]
board_br = CALIBRATION["board_box"]["br"]


# Grab pixel values in 8x8 grid
tl_x, tl_y = board_tl['x'], board_tl['y']
br_x, br_y = board_br['x'], board_br['y']

cell_w = (br_x - tl_x) // 8
cell_h = (br_y - tl_y) // 8

CELL_PIXEL_BUFFER = 10  # Pixels away from center to sample top/bottom
difference_threshold = 10

def get_board_state(snapshot=None):
    snapshot = _get_snapshot() if snapshot is None else snapshot
    board = np.zeros((8, 8), dtype=int)
    left, top = _scaled_point(tl_x, tl_y)
    right, bottom = _scaled_point(br_x, br_y)
    width = right - left
    height = bottom - top
    rel_x = left - snapshot["left"]
    rel_y = top - snapshot["top"]
    img = snapshot["img"][rel_y:rel_y + height, rel_x:rel_x + width]

    cell_w = width // 8
    cell_h = height // 8
    for row in range(8):
        for col in range(8):
            cx = int(col * cell_w + cell_w // 2)
            cy = int(row * cell_h + cell_h // 2)
            top = img[cy + CELL_PIXEL_BUFFER, cx]
            bot = img[cy - CELL_PIXEL_BUFFER, cx]
            if np.any(np.abs(top.astype(int) - bot.astype(int)) > difference_threshold):
                board[row, col] = 1

    return board

def print_board(board):
    for row in board:
        print("".join(['#' if cell else '.' for cell in row]))
    print()
# --------------------
