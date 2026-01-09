from PIL import Image
import mss
import pytesseract
import pyautogui
from pynput import mouse
import sys
import time

sct = mss.mss()
scale_x = 1.0
scale_y = 1.0

if pyautogui is not None:
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

pos = {"x": 0, "y": 0}
ls = []

def on_move(x, y):
    pos["x"] = x
    pos["y"] = y

def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.left:
        ls.append((pos["x"], pos["y"]))

def screenshot():
    x1, y1 = ls.pop()
    x2, y2 = ls.pop()

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    # Adjust for scaling
    left, top = int(left*scale_x), int(top*scale_y)
    bottom, right = int(bottom*scale_x), int(right*scale_y)

    width = right - left
    height = bottom - top
    shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    # OCR processing
    img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)
    text = pytesseract.image_to_string(img, config='--psm 7')
    text = text.strip()
    print()
    print(f"OCR Result: {text}")
def screenshot_img():
    x1, y1 = ls.pop()
    x2, y2 = ls.pop()

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    # Adjust for scaling
    left, top = int(left*scale_x), int(top*scale_y)
    bottom, right = int(bottom*scale_x), int(right*scale_y)

    width = right - left
    height = bottom - top
    shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    # Save image
    img = Image.frombytes("RGB", (shot.width, shot.height), shot.rgb).convert("L")
    img.save("screenshot.png")
    print()
    print(f"Saved screenshot.png")

listener = mouse.Listener(on_move=on_move, on_click=on_click)
listener.start()

while True:
    if len(ls) >= 2:
        # screenshot()
        screenshot_img()
    line = f"Cursor: ({pos['x']:.0f}, {pos['y']:.0f})"
    sys.stdout.write("\r" + line.ljust(40))
    sys.stdout.flush()
    time.sleep(0.05)