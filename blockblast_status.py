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

# ---------- Classification of blocks ----------

# Screenshot trays

# Binary mapping of block presence

# Get max dimensions

# Classify blocks into rectangular classes

# Classify blocks into template classes

# ---------- Combo status ----------

# Check pixel value

# ---------- Score status ----------

# Screenshot score area

# ---------- Board state ----------

# Read board area screenshot





# """Example usage of the block library generation and printing templates."""
# BLOCK_LIBRARY = build_block_library()
# print(f"Generated block library with {sum(len(v) for v in BLOCK_LIBRARY.values())} templates.")

# print(f"Example templates for class '3x3': {BLOCK_LIBRARY['3x3']}")
# for template in BLOCK_LIBRARY['3x3']:
#     print_formatted_mask(template)

if __name__ == "__main__":
    BLOCK_LIBRARY = build_block_library()