import time
import random
import blockblast_status as status
import blockblast_calibration as calibration
import numpy as np

# Defines methods for placing blocks in BlockBlast game.
    
def place_block(board, tray_index, grid_x, grid_y, block):
    # if self.invalid_placement(grid_x, grid_y, block):
    #     print(f"Invalid placement for tray {tray_index} at ({grid_x}, {grid_y})")
    #     return False
    
    # previous_board = self.board

    # Get lines cleared from action
    lines_cleared = get_lines_cleared(board, grid_x, grid_y, block)
    calibration.drag_piece(
        tray_index, grid_y, grid_x, class_name=f"{block.shape[0]}x{block.shape[1]}"
    )
    return lines_cleared
    # if not self.refresh_status():
    #     print("Loss detected after placement.")
    #     return False

    # if not self.placed_correctly(grid_x, grid_y, block, previous_board):
    #     print(f"Block from tray {tray_index} not placed correctly at ({grid_x}, {grid_y})")
    #     return False
    
    # print(f"Placed block from tray {tray_index} at ({grid_x}, {grid_y}) successfully")
    # return True
    # return lines cleared
    

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
    # # Create a temporary board with the current block placed
    # temp_board = board.copy()
    # if block is None:
    #     return False
    # block_h, block_w = block.shape
    # for i in range(block_h):
    #     for j in range(block_w):
    #         if block[i][j] == 1:
    #             temp_board[grid_y + i][grid_x + j] = 1

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


    
    # Check if any of the remaining blocks can fit anywhere on the temp_board
    if debug:
        print("[loss] evaluating remaining blocks:")
        for i, b in enumerate(blocks):
            if b is None:
                print(f"[loss] tray {i}: None")
            else:
                print(f"[loss] tray {i}: {b.shape}")
    for b in blocks:
        if b is block or b is None:
            continue  # Skip the block that was just placed
        b_h, b_w = b.shape
        for y in range(8 - b_h + 1):
            for x in range(8 - b_w + 1):
                if not invalid_placement(x, y, b, temp_board):
                    if debug:
                        print(f"[loss] found valid placement for {b.shape} at ({y},{x})")
                    return False  # Found a valid placement, not a loss
    if debug:
        print("[loss] no valid placements found")
    return True  # No valid placements found, loss condition met

# Click on correct position to start game
def click_restart():
    restart_pixel = status.CALIBRATION["restart_pixel"]
    restart_pixel_value = status.CALIBRATION["restart_pixel_value"]
    # # Print all pixel values from (0, 0) to (1000, 1000) for debugging
    # for y in range(0, 1000, 50):
    #     for x in range(0, 1000, 50):
    #         pixel_value = status.sample_pixel(x, y)
    #         print(f"Pixel at ({x}, {y}): {pixel_value}")
    while status.sample_pixel(restart_pixel['x'], restart_pixel['y'], snapshot=status.screenshot()) != tuple(restart_pixel_value):
        time.sleep(0.05)
    status.pyautogui.click(restart_pixel['x'], restart_pixel['y'])
    print("Restart Button pressed")
    time.sleep(1.5)  # Wait for game to start

def click_settings_replay():
    focus_pixel = status.CALIBRATION["focus"]
    status.pyautogui.click(focus_pixel['x'], focus_pixel['y'])
    settings_pixel = status.CALIBRATION["settings_pixel"]

    status.pyautogui.click(settings_pixel['x'], settings_pixel['y'])

    replay_pixel = status.CALIBRATION["replay_pixel"]
    status.pyautogui.click(replay_pixel['x'], replay_pixel['y'])
    time.sleep(1.5)  # Wait for game to start
    
    
