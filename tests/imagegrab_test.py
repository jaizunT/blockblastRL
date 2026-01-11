#!/usr/bin/env python3
import sys

try:
    import pyautogui
    from PIL import ImageGrab
except ImportError:
    pyautogui = None
    ImageGrab = None


def main():
    if pyautogui is None or ImageGrab is None:
        print("Missing dependency: pyautogui or pillow")
        print("Install with: python -m pip install pyautogui pillow")
        sys.exit(1)

    mouse_w, mouse_h = pyautogui.size()
    img = ImageGrab.grab(all_screens=True)
    img_w, img_h = img.size
    scale_x = img_w / mouse_w if mouse_w else 1.0
    scale_y = img_h / mouse_h if mouse_h else 1.0

    print(f"pyautogui.size = {mouse_w}x{mouse_h}")
    print(f"ImageGrab.size  = {img_w}x{img_h}")
    print(f"scale = ({scale_x:.3f}, {scale_y:.3f})")

    x, y = pyautogui.position()
    px = int(x * scale_x)
    py = int(y * scale_y)
    if 0 <= px < img_w and 0 <= py < img_h:
        r, g, b, *_ = img.getpixel((px, py))
        print(f"Cursor at ({x}, {y}) -> pixel ({px}, {py}) RGB=({r},{g},{b})")
    else:
        print(f"Scaled cursor ({px}, {py}) is out of bounds.")


if __name__ == "__main__":
    main()
