#!/usr/bin/env python3
import sys
import mss

try:
    import pyautogui
    from PIL import ImageGrab, ImageDraw
except ImportError:
    pyautogui = None
    ImageGrab = None
    ImageDraw = None


def main():
    if pyautogui is None or ImageGrab is None or ImageDraw is None:
        print("Missing dependency: pyautogui or pillow")
        print("Install with: python -m pip install pyautogui pillow")
        sys.exit(1)

    x, y = pyautogui.position()
    img = ImageGrab.grab(all_screens=True)
    screen_w, screen_h = pyautogui.size()
    scale_x = img.width / screen_w if screen_w else 1.0
    scale_y = img.height / screen_h if screen_h else 1.0
    x = int(x * scale_x)
    y = int(y * scale_y)
    draw = ImageDraw.Draw(img)

    size = 15
    draw.line((x - size, y, x + size, y), fill=(255, 0, 0), width=2)
    draw.line((x, y - size, x, y + size), fill=(255, 0, 0), width=2)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 0, 0), width=2)

    img.save("cursor_check.png")
    print(f"Saved cursor_check.png at ({x}, {y}) scale=({scale_x:.2f}, {scale_y:.2f})")


if __name__ == "__main__":
    main()
