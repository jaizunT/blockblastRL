import blockblast_calibration as calibration
import blockblast_status as status
import blockblast_play as play

# Auto-calibration script for BlockBlast game.

# Checks current tray to see current tray
# Checks calibration.json to see if any class is missing for the tray
# If missing, prompts user to calibrate that class
def auto_calibrate():
    snapshot = status.screenshot()
    blocks = play.get_blocks(snapshot)
    for tray_index, block in enumerate(blocks):
        if block is None:
            continue
        class_name = f"{block.shape[0]}x{block.shape[1]}"
        if not is_class_calibrated(class_name, tray_index):
            print(f"Class {class_name} not calibrated for tray {tray_index}.")
            # Prompt user to calibrate this class and calibrates if user agrees
            input("Press Enter to start calibration... Press Ctrl+C to skip.")
            calibration.calibrate_class_offset(class_name, tray_index)


def is_class_calibrated(class_name, tray_index):
    cal = calibration.load_calibration()
    offsets_by_tray = cal.get("class_offsets_by_tray", {})
    tray_key = str(tray_index)
    return class_name in offsets_by_tray.get(tray_key, {})

def main():
    while True:
        # Asks user if want to calibrate, restart, or exit program
        user_input = input("Type 'c' to auto-calibrate, 'r' to restart game, or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting program.")
            break
        elif user_input.lower() == 'r':
            play.click_settings_replay()
            print("Game restarted.")
        elif user_input.lower() == 'c':
            auto_calibrate()
            print("Auto-calibration complete.")
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
