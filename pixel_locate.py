#!/usr/bin/env python3
import sys
import time

try:
    from pynput import mouse
except ImportError:
    mouse = None

try:
    import mss
except ImportError:
    mss = None

try:
    import pyautogui
except ImportError:
    pyautogui = None


def main():
    if mouse is None:
        print("Missing dependency: pynput")
        print("Install with: python -m pip install pynput")
        sys.exit(1)
    if mss is None:
        print("Missing dependency: mss")
        print("Install with: python -m pip install mss")
        sys.exit(1)

    pos = {"x": 0, "y": 0}

    def on_move(x, y):
        pos["x"] = x
        pos["y"] = y

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

    def sample_rgb(x, y):
        left = int(x * scale_x)
        top = int(y * scale_y)
        shot = sct.grab({"left": left, "top": top, "width": 1, "height": 1})
        pixel = shot.pixel(0, 0)
        b, g, r = pixel
        return r, g, b

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left:
            r, g, b = sample_rgb(pos["x"], pos["y"])
            sys.stdout.write(
                f"\nClick: ({pos['x']:.0f}, {pos['y']:.0f}) RGB=({r}, {g}, {b})\n"
            )
            sys.stdout.flush()

    listener = mouse.Listener(on_move=on_move, on_click=on_click)
    listener.start()

    try:
        while True:
            r, g, b = sample_rgb(pos["x"], pos["y"])
            line = f"Cursor: ({pos['x']:.0f}, {pos['y']:.0f}) RGB=({r}, {g}, {b})"
            sys.stdout.write("\r" + line.ljust(40))
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
    finally:
        sct.close()
        listener.stop()


if __name__ == "__main__":
    main()
