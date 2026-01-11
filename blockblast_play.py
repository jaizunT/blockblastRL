import time
import random
import blockblast_status as status
import blockblast_calibration as calibration

class BlockBlastGame:
    # Stores game state and provides methods to interact with the game.
    def __init__(self):
        board = status.get_board_state()
        self.board = board
        self.score = status.get_score()
        self.in_combo = status.is_in_combo()
        self.loss = status.check_loss()
    
    def place_block(self, tray_index, grid_x, grid_y, block):
        if self.invalid_placement(grid_x, grid_y, block):
            print(f"Invalid placement for tray {tray_index} at ({grid_x}, {grid_y})")
            return False
        
        previous_board = self.board
        calibration.drag_piece(tray_index, grid_x, grid_y, class_name=f"{block.shape[1]}x{block.shape[0]}")
        
        if not self.refresh_status():
            print("Loss detected after placement.")
            return False

        if not self.placed_correctly(grid_x, grid_y, block, previous_board):
            print(f"Block from tray {tray_index} not placed correctly at ({grid_x}, {grid_y})")
            return False
        
        print(f"Placed block from tray {tray_index} at ({grid_x}, {grid_y}) successfully")
        return True

    def invalid_placement(self, grid_x, grid_y, block):
        # Check if placing block at (grid_x, grid_y) would overlap existing blocks or go out of bounds
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                board_x = grid_x + j
                board_y = grid_y + i
                if board_x < 0 or board_y < 0 or board_x >= 8 or board_y >= 8:
                    return True  # Out of bounds
                if self.board[board_y][board_x] != 0 and block[i][j] == 1:
                    return True  # Overlaps existing block
        return False
    
    def placed_correctly(self, grid_x, grid_y, block, previous_board):
        # Verify the block filled expected cells, accounting for line clears.
        board_h = len(self.board)
        board_w = len(self.board[0]) if board_h else 0
        if hasattr(block, "shape"):
            block_h, block_w = block.shape
        else:
            block_h = len(block)
            block_w = len(block[0]) if block_h else 0

        expected = [row[:] for row in previous_board]
        for i in range(block_h):
            for j in range(block_w):
                if block[i][j] == 1:
                    expected[grid_y + i][grid_x + j] = 1

        full_rows = [r for r in range(board_h) if all(expected[r][c] != 0 for c in range(board_w))]
        full_cols = [c for c in range(board_w) if all(expected[r][c] != 0 for r in range(board_h))]

        for r in full_rows:
            for c in range(board_w):
                expected[r][c] = 0
        for c in full_cols:
            for r in range(board_h):
                expected[r][c] = 0

        return self.board == expected

    def place_blocks(self, tray_indices, positions, blocks):
        for tray_index, (grid_x, grid_y), block in zip(tray_indices, positions, blocks):
            success = self.place_block(tray_index, grid_x, grid_y, block)
            if not success:
                print(f"Failed to place block from tray {tray_index} at ({grid_x}, {grid_y})")
                return False
        return True
    
    def random_place(self):
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
            self.place_block(tray_index, grid_x, grid_y, block)

    def get_blocks(self):
        return status.classify_all_trays()
    
    def refresh_status(self):
        while True:
            if status.background_stable():
                break
        self.board = status.get_board_state()
        self.score = status.get_score()
        self.in_combo = status.is_in_combo()
        # Return loss status
        return self.check_loss()

    # Checks pixels to determine if game is lost
    def check_loss(self):
        pass
    # Click on correct position to start game
    def start_game(self):
        pass
