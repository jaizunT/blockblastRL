import json
import mss
import pytesseract
import time
import pyautogui
from PIL import ImageGrab
from PIL import Image
import numpy as np

"""Utilities for reading BlockBlast screen status and block templates."""

# Block classes you use for calibration.
BLOCK_CLASSES = [
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

# Load calibration data.
with open("calibration.json") as f:
    CALIBRATION = json.load(f)

img = ImageGrab.grab(all_screens=True)
screen_w, screen_h = pyautogui.size()
scale_x = img.width / screen_w if screen_w else 1.0
scale_y = img.height / screen_h if screen_h else 1.0

# Functions to generate block templates for each class.

def parse_class(class_name):
    w, h = class_name.split("x")
    return int(w), int(h)


def is_connected(mask, w, h, diagonal=True):
    cells = [(x, y) for y in range(h) for x in range(w) if mask & (1 << (y * w + x))]
    if not cells:
        return False
    stack = [cells[0]]
    seen = set([cells[0]])
    while stack:
        x, y = stack.pop()
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                if mask & (1 << (ny * w + nx)):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        if diagonal:
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                    if mask & (1 << (ny * w + nx)):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
    return len(seen) == len(cells)


def mask_to_array(mask, w, h):
    return [
        [1 if mask & (1 << (y * w + x)) else 0 for x in range(w)]
        for y in range(h)
    ]


def generate_class_masks(w, h, diagonal=True):
    total = w * h
    masks = []
    for mask in range(1, 1 << total):
        if not is_connected(mask, w, h, diagonal=diagonal):
            continue
        cells = [(x, y) for y in range(h) for x in range(w) if mask & (1 << (y * w + x))]
        min_x = min(x for x, _ in cells)
        max_x = max(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        max_y = max(y for _, y in cells)
        if (max_x - min_x + 1) != w or (max_y - min_y + 1) != h:
            continue
        masks.append(mask_to_array(mask, w, h))
    return masks


def build_block_library(classes=None, diagonal=True):
    classes = classes or BLOCK_CLASSES
    library = {}
    for cls in classes:
        w, h = parse_class(cls)
        library[cls] = generate_class_masks(w, h, diagonal=diagonal)
    return library

def print_formatted_mask(mask):
    for row in mask:
        print("".join(['#' if cell else '.' for cell in row]))
    print()
# ---------- Initialize MSS ----------
sct = mss.mss()
def sample_pixel(x, y):
    # Adjust for scaling
    left = int(x * scale_x)
    top = int(y * scale_y)

    shot = sct.grab({"left": left, "top": top, "width": 1, "height": 1})
    pixel = shot.pixel(0, 0)
    b, g, r = pixel
    return r, g, b

# ---------- Classification of blocks ----------

# Screenshot trays
tray0_tl = CALIBRATION["tray_boxes"]["0"]["tl"]
tray0_br = CALIBRATION["tray_boxes"]["0"]["br"]
tray1_tl = CALIBRATION["tray_boxes"]["1"]["tl"]
tray1_br = CALIBRATION["tray_boxes"]["1"]["br"]
tray2_tl = CALIBRATION["tray_boxes"]["2"]["tl"]
tray2_br = CALIBRATION["tray_boxes"]["2"]["br"]

def sample_tray(tray_tl, tray_br):
    left, top = tray_tl['x'], tray_tl['y']
    bottom, right = tray_br['y'], tray_br['x']

    # Adjust for scaling
    left, top = int(left*scale_x), int(top*scale_y)
    bottom, right = int(bottom*scale_x), int(right*scale_y)

    width = right - left
    height = bottom - top
    shot = sct.grab({"left": left, "top": top, "width": width, "height": height})

    # Convert to gray scale for easier processing
    img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb).convert("L")

    # Reduce size for faster processing
    img_array = np.array(img)

    # Get max and min for bounding box
    bg_pixel = img_array[0,0]
    bg = int(bg_pixel)
    mask = np.abs(img_array.astype(np.int16) - bg) > 10

    cols_with_fg = np.where(mask.any(axis=0))[0]
    rows_with_fg = np.where(mask.any(axis=1))[0]

    x0, x1 = cols_with_fg[0], cols_with_fg[-1]
    y0, y1 = rows_with_fg[0], rows_with_fg[-1] 
    cropped = img_array[y0:y1+1, x0:x1+1]

    # Classify blocks into rectangular classes
    h, w = cropped.shape
    CELL_LENGTH = 16

    block_h = int(np.round(h / CELL_LENGTH))
    block_w = int(np.round(w / CELL_LENGTH))

    block = [[0 for i in range(block_w)] for j in range(block_h)]

    # Fill in block shape using pixel at center of each cell using thresholding
    for i in range(block_h):
        for j in range(block_w):
            if np.abs(cropped[i*CELL_LENGTH + CELL_LENGTH//2, j*CELL_LENGTH + CELL_LENGTH//2] - bg) > 10:
                block[i][j] = 1
    
    return block

def classify_tray(tray_index):
    if tray_index == 0:
        tray_tl, tray_br = tray0_tl, tray0_br
    elif tray_index == 1:
        tray_tl, tray_br = tray1_tl, tray1_br
    elif tray_index == 2:
        tray_tl, tray_br = tray2_tl, tray2_br
    else:
        raise ValueError("Invalid tray index")
    
    block = sample_tray(tray_tl, tray_br)
    return block

def classify_all_trays():
    blocks = []
    for tray_index in range(3):
        block = classify_tray(tray_index)
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
    sampled_pixel = sample_pixel(bg_pixel['x'], bg_pixel['y'])
    time.sleep(0.02)
    # If background pixel matches same value over multiple samples, it's stable
    return sampled_pixel == sample_pixel(bg_pixel['x'], bg_pixel['y'])

def is_in_combo():
    sampled_pixel = sample_pixel(combo_pixel['x'], combo_pixel['y'])
    background_sampled = sample_pixel(bg_pixel['x'], bg_pixel['y'])
    # If combo pixel differs from background pixel, we are in combo
    return sampled_pixel != background_sampled

# ---------- Score status ----------

# Screenshot score area
score_tl = CALIBRATION["score_box"]["tl"]
score_br = CALIBRATION["score_box"]["br"]

# OCR to read score value
def get_score():
    left, top = score_tl['x'],score_tl['y']
    bottom, right = score_br['y'], score_br['x']
    # Adjust for scaling
    left, top = int(left*scale_x), int(top*scale_y)
    bottom, right = int(bottom*scale_x), int(right*scale_y)

    width = right - left
    height = bottom - top
    shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    # OCR processing
    img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
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

BOTTOM_CELL_PIXEL_BUFFER = 5

def get_board_state():
    board = [[False for _ in range(8)] for _ in range(8)]
    for row in range(8):
        for col in range(8):
            # Sample pixel at center of cell
            center_cell_x = tl_x + col * cell_w + cell_w // 2
            center_cell_y = tl_y + row * cell_h + cell_h // 2
            center_pixel = sample_pixel(center_cell_x, center_cell_y)
            # Sample pixel at bottom of cell
            bottom_cell_x = tl_x + col * cell_w + cell_w // 2
            bottom_cell_y = tl_y + (row + 1) * cell_h - BOTTOM_CELL_PIXEL_BUFFER
            bottom_pixel = sample_pixel(bottom_cell_x, bottom_cell_y)
            # Determine if block is present based on pixel comparison
            # If center of cell is different from bottom of cell, mark as occupied
            if center_pixel != bottom_pixel:
                board[row][col] = True
    return board

def print_board(board):
    for row in board:
        print("".join(['#' if cell else '.' for cell in row]))
    print()
# --------------------


# """Example usage of the block library generation and printing templates."""
# BLOCK_LIBRARY = build_block_library()
# print(f"Generated block library with {sum(len(v) for v in BLOCK_LIBRARY.values())} templates.")

# print(f"Example templates for class '3x3': {BLOCK_LIBRARY['3x3']}")
# for template in BLOCK_LIBRARY['3x3']:
#     print_formatted_mask(template)

