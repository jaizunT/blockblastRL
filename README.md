Block Blast RL Automation (iPhone Mirroring)

This folder contains a simple Python utility to calibrate the Block Blast tray and grid
positions from macOS iPhone Mirroring, then script piece placements.

Requirements
- macOS Accessibility permissions for Terminal/Python.
- Conda environment: blockblast (Python 3.12.2)

Install
conda create -y -n blockblast -c conda-forge python=3.12.2
conda activate blockblast
python -m pip install pynput pyautogui

Calibrate
conda activate blockblast
  python blockblast_calibration.py calibrate

You will be prompted to click:
1) Top-left of the 8x8 grid
2) Bottom-right of the 8x8 grid
3) Center of tray pieces 0, 1, 2
4) A safe focus point inside the iPhone Mirroring window
5) For each tray piece: click and hold, press Space to capture pickup cursor, then drag to (2,6) and press Space, then to (7,1) and press Space

Shape Class Calibration
- Edit the class list in `blockblast_calibration.py` (CLASSES) to match your needs.
- Calibrate one class at a time (records tray pickup + a class-specific drag transform + class offset):
  python blockblast_calibration.py calibrate-class <class_name> <tray_index 0-2>
- See which classes still need calibration:
  python blockblast_calibration.py status
- Reset calibration data:
  python blockblast_calibration.py reset calibration
  python blockblast_calibration.py reset pair --class 3x3 --tray 0
  python blockblast_calibration.py reset class --class 3x3 --tray 0
- Infer missing tray/class offsets and transforms from other trays (stored separately):
  python blockblast_calibration.py infer
- Save inference cache (tray mappings + size-fit models):
  python blockblast_calibration.py cache

Calibration Summary
- Calibration now uses class-specific transforms:
  1) Tray pickup: where you click to pick up a piece (per tray).
  2) Class transform: how cursor motion maps to piece motion (per tray + class).
  3) Class offset: shape-specific pickup jump (constant offset) after the transform.
- `calibrate-class` records the class transform + class offset for that (tray, class).
- `place` uses the class transform + class offset (via `--class`).
  - Missing class transforms/offsets are inferred from other trays using tray-to-tray
    affine mappings and a size-fit fallback (stored in `calibration_inference.json`).

Place a Piece
conda activate blockblast
python blockblast_calibration.py place <tray_index 0-2> <row 0-7> <col 0-7> [--class <class_name>]

Example
python blockblast_calibration.py place 0 3 4

PPO Training (SB3)
- Wrapper lives in `blockblast_ppo.py` (Gymnasium + SB3).
- Optional action masking (requires sb3-contrib):
  python -m pip install sb3-contrib
- Run PPO without masking:
  python blockblast_ppo.py --timesteps 10000
- Run PPO with masking:
  python blockblast_ppo.py --timesteps 10000 --masking
- Run PPO with the simulated environment:
  python blockblast_ppo.py --sim
- Run PPO with the simulator and action masking:
  python blockblast_ppo.py --sim --masking --num-envs 4
- Resume masked PPO at a checkpoint using the simulator:
  python blockblast_ppo.py --sim --masking --resume-step 10000
- Resume PPO from a checkpoint step (loads `ppo_checkpoints/rl_model_<step>_steps.zip`):
  python blockblast_ppo.py --masking --resume-step 10000
- Flags:
  - `--timesteps`: total timesteps across all envs (rounded up to full rollouts)
  - `--save-freq`: checkpoint frequency (callback calls)
  - `--save-dir`: checkpoint directory (default `ppo_checkpoints`)
  - `--model-path`: final model path (default `ppo_blockblast.zip`)
  - `--masking`: enable action masking (sb3-contrib)
  - `--resume-step`: load `ppo_checkpoints/rl_model_<step>_steps.zip`
  - `--sim`: use the simulator instead of live screen capture
  - `--num-envs`: number of parallel sim envs (default 4)
  - `--log-episodes`: print episodic stats every N episodes
- Current PPO defaults (live + sim):
  - `gamma=0.95`, `gae_lambda=0.95`, `n_steps=128`
  - `net_arch=[256, 256, 128]` (shared for policy/value)
- Rollouts: each rollout collects `n_steps` per env, so size is `n_steps * num_envs`.
- Example: `num_envs=500`, `n_steps=128` ⇒ `64,000` timesteps per rollout. If you set
  `--timesteps 10_000`, SB3 still runs one full rollout (64k).
- Checkpoints are saved to `ppo_checkpoints/`; with VecEnv, effective step interval is
  `save_freq * num_envs` timesteps (one callback call per env step). Final model is
  saved to `--model-path`.

RL Agent (custom loop)
- The baseline policy loop is in `blockblast_agent.py`.
- It supports action masking with `get_valid_move_mask`.

Status Processing (blockblast_status.py)
- Shared scaling: coordinates from calibration.json (pyautogui-space) are scaled to the mss capture size.
- Tray classification:
  - Capture each tray box from `tray_boxes` with mss.
  - Convert to grayscale and threshold vs background to get a mask.
  - Compute the bounding box of foreground pixels.
  - Sample each cell center inside that box to build a binary block grid (1 = filled).
- Combo detection:
  - Sample `background_pixel` and `combo_pixel`.
  - If the background is stable and the combo pixel differs from background, combo is active.
- Score OCR:
  - Capture `score_box` with mss.
  - Convert to PIL and run pytesseract with `--psm 7` + digit whitelist.
- Board state:
  - Use `board_box` to compute 8x8 cell centers.
  - Compare center pixel vs a lower pixel in the same cell to mark occupancy.

Data Pipeline (batch_log.jsonl + unique_blocks.txt)
- `BlockBlastEnv.log_batch` appends one JSON record per line to `batch_log.jsonl`:
  - `board`: 8x8 grid
  - `trays`: 3x5x5 padded tray masks
- `unique_blocks.txt` stores unique raw tray shapes as `HxW|row/row/...`.
- `block_generation_net.py` includes helpers to:
  - load batches and map trays to unique block IDs
  - skip unknown shapes
  - train/test/val split and accuracy metrics
- The block generation model outputs 3 class IDs (one per tray).
- Use `--set-eval` to train/evaluate with set-invariant targets (sorted).
- Weights are saved to `block_generator/block_generation_model.pth`.

Simulation (blockblast_simulation.py)
- `BlockBlastSim` simulates the 8x8 grid and regenerates batches after 3 placements.
- Batches are sampled from the block generator weights and re-rolled until solvable.
- Combo logic is line-based: clearing within 3 moves increments a line streak; combo
  starts once the streak reaches 3 lines.
- Optional cluttered starts:
  - `generate_clutterered_board(min_clutter=0.4)` builds a random board by growing
    blocks (size ~ N(3, 2.5)) until a target clutter fraction is reached.
  - Set `clutter_prob` (and `min_clutter`) in `BlockBlastSim(...)` to start from a
    cluttered board on reset with probability `p`; otherwise resets to empty.
- Rewards mirror the live agent heuristic:
  - `score_increase/1000`, `lines_cleared**1.5`, combo bonuses, step bonus,
  clutter penalty (thresholded), and delta-holes bonus.
  - Invalid moves return `-5.0`; terminal loss returns `-7.5`.

Notes
- The script uses drag-and-drop from the tray pickup point to the target cell center.
- Failsafe: move the cursor to a screen corner to abort (pyautogui default).
