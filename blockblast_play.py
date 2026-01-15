import time
import random
import blockblast_status as status
import blockblast_calibration as calibration
import numpy as np

# Defines methods for placing blocks in BlockBlast game.
    
def place_block(board, tray_index, grid_x, grid_y, block):

    lines_cleared = get_lines_cleared(board, grid_x, grid_y, block)
    calibration.drag_piece(
        tray_index, grid_y, grid_x, class_name=f"{block.shape[0]}x{block.shape[1]}"
    )
    return lines_cleared

    

def invalid_placement(grid_x, grid_y, block, board):
    if block is None:
        return True
    # Check if placing block at (grid_x, grid_y) would overlap existing blocks or go out of bounds
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            board_x = grid_x + j
            board_y = grid_y + i
            if board_x < 0 or board_y < 0 or board_x >= 8 or board_y >= 8:
                return True  # Out of bounds
            if board[board_y][board_x] != 0 and block[i][j] == 1:
                return True  # Overlaps existing block
    return False

def expected_board_after_placement(grid_x, grid_y, block, previous_board):
    board = np.array(previous_board, dtype=int, copy=True)
    block_h, block_w = block.shape

    for i in range(block_h):
        for j in range(block_w):
            if block[i][j] == 1:
                board[grid_y + i, grid_x + j] = 1

    full_rows = np.where(np.all(board != 0, axis=1))[0]
    full_cols = np.where(np.all(board != 0, axis=0))[0]

    if full_rows.size:
        board[full_rows, :] = 0
    if full_cols.size:
        board[:, full_cols] = 0

    return board


def placed_correctly(grid_x, grid_y, block, current_board, previous_board):
    expected = expected_board_after_placement(grid_x, grid_y, block, previous_board)
    return np.array_equal(np.array(current_board), expected)

def place_blocks(tray_indices, positions, blocks):
    for tray_index, (grid_x, grid_y), block in zip(tray_indices, positions, blocks):
        success = place_block(tray_index, grid_x, grid_y, block)
        if not success:
            print(f"Failed to place block from tray {tray_index} at ({grid_x}, {grid_y})")
            return False
    return True

def random_place():
    block0, block1, block2 = status.classify_all_trays()
    blocks = {0: block0, 1: block1, 2: block2}
    order = [0, 1, 2]
    random.shuffle(order)
    for tray_index in order:
        block = blocks[tray_index]
        h, w = block.shape
        print(f"Tray {tray_index} block shape: {h}x{w}")
        status.print_block(block)
        grid_x = random.randint(0, 8 - w)
        grid_y = random.randint(0, 8 - h)
        print(f"Placing tray {tray_index} block at ({grid_x}, {grid_y})")
        place_block(tray_index, grid_x, grid_y, block)

def get_blocks(snapshot=None):
    return status.classify_all_trays(snapshot=snapshot)

def refresh_status():
    while True:
        if status.background_stable():
            break
    board = status.get_board_state()
    score = status.get_score()
    in_combo = status.is_in_combo()
    return board, score, in_combo

# Get number of lines cleared since based on previous board state and current action
def get_lines_cleared(board, grid_x, grid_y, block):
    temp_board = board.copy()
    block_h, block_w = block.shape
    for i in range(block_h):
        for j in range(block_w):
            if block[i][j] == 1:
                temp_board[grid_y + i][grid_x + j] = 1

    rows_cleared = [r for r in range(8) if all(temp_board[r][c] != 0 for c in range(8))]
    cols_cleared = [c for c in range(8) if all(temp_board[r][c] != 0 for r in range(8))]
    return len(rows_cleared) + len(cols_cleared)

# Checks internal board state with currently placed piece for loss condition if rest of pieces can't fit anywhere.
def check_loss(board, block, grid_x, grid_y, blocks, debug=False):

    # If loss pixels different from bg pixels and loss pixels are similar and bg pixels are similar, then it's a loss
    loss_pixel1 = status.CALIBRATION["loss_pixel1"] 
    loss_pixel2 = status.CALIBRATION["loss_pixel2"]
    bg_pixel1 = status.CALIBRATION["bg_pixel1"]
    bg_pixel2 = status.CALIBRATION["bg_pixel2"]
    lr1, lg1, lb1 = status.sample_pixel(loss_pixel1['x'], loss_pixel1['y'])
    lr2, lg2, lb2 = status.sample_pixel(loss_pixel2['x'], loss_pixel2['y'])
    br1, bg1, bb1 = status.sample_pixel(bg_pixel1['x'], bg_pixel1['y'])
    br2, bg2, bb2 = status.sample_pixel(bg_pixel2['x'], bg_pixel2['y'])
    if (status.color_difference((lr1, lg1, lb1), (lr2, lg2, lb2)) < 15 and
        status.color_difference((br1, bg1, bb1), (br2, bg2, bb2)) < 15 and
        status.color_difference((lr1, lg1, lb1), (br1, bg1, bb1)) > 30):
        if debug:
            print("[loss] loss pixels indicate loss")
        return True
    return False


# Click on correct position to start game
def click_restart():
    restart_pixel = status.CALIBRATION["restart_pixel"]
    restart_pixel_value = status.CALIBRATION["restart_pixel_value"]


    while status.sample_pixel(restart_pixel['x'], restart_pixel['y'], snapshot=status.screenshot()) != tuple(restart_pixel_value):
        time.sleep(0.05)

    status.pyautogui.click(restart_pixel['x'], restart_pixel['y'])
    
    print("Restart Button pressed")
    time.sleep(2.0)  # Wait for game to start

def click_out_of_ad():
    video_pixel1 = status.CALIBRATION["video_pixel1"]
    video_pixel2 = status.CALIBRATION["video_pixel2"]
    video_pixel_value = status.CALIBRATION["video_pixel_value"]

    no_thanks_pixel = status.CALIBRATION["no_thanks_pixel"]

    while not (status.sample_pixel(video_pixel1['x'], video_pixel1['y'], snapshot=status.screenshot()) == tuple(video_pixel_value)
            and status.sample_pixel(video_pixel2['x'], video_pixel2['y'], snapshot=status.screenshot()) == tuple(video_pixel_value)):
            time.sleep(0.05)
    status.pyautogui.click(no_thanks_pixel['x'], no_thanks_pixel['y'])
    print("Dismissed video ad")

def video_ad_detected(snapshot=None):
    snapshot = status.screenshot() if snapshot is None else snapshot
    video_pixel1 = status.CALIBRATION["video_pixel1"]
    video_pixel2 = status.CALIBRATION["video_pixel2"]
    video_pixel_value = tuple(status.CALIBRATION["video_pixel_value"])
    return (
        status.sample_pixel(video_pixel1['x'], video_pixel1['y'], snapshot=snapshot) == video_pixel_value
        and status.sample_pixel(video_pixel2['x'], video_pixel2['y'], snapshot=snapshot) == video_pixel_value
    )
def click_settings_replay():
    focus_pixel = status.CALIBRATION["focus"]
    status.pyautogui.click(focus_pixel['x'], focus_pixel['y'])
    settings_pixel = status.CALIBRATION["settings_pixel"]

    status.pyautogui.click(settings_pixel['x'], settings_pixel['y'])

    replay_pixel = status.CALIBRATION["replay_pixel"]
    status.pyautogui.click(replay_pixel['x'], replay_pixel['y'])
    time.sleep(2.0)  # Wait for game to start
    
    
