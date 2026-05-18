#!/usr/bin/env python3
"""
Flappy Bird AI Bot — Screen Reading + Neural Network
=====================================================
Loads a trained brain (JSON) from the HTML trainer and plays
the real Flappy Bird game by reading the screen.

Requirements:
    pip install mss numpy pyautogui opencv-python

Usage:
    python flappy_bot.py --brain flappy_brain_gen50_score10.json
    python flappy_bot.py --brain brain.json --game-region 100 100 400 600
    python flappy_bot.py --brain brain.json --click 500 400

Controls:
    Q = quit
    P = pause/resume
    S = screenshot debug
"""

import json
import time
import argparse
import sys
import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Run: pip install numpy")
    sys.exit(1)

try:
    import mss
    import mss.tools
except ImportError:
    print("ERROR: mss required. Run: pip install mss")
    sys.exit(1)

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
except ImportError:
    print("ERROR: pyautogui required. Run: pip install pyautogui")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python required. Run: pip install opencv-python")
    sys.exit(1)


# ══════════════════════════════════════════════
#  NEURAL NETWORK (mirrors the JS version)
# ══════════════════════════════════════════════
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_ih = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.weights_ho = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.bias_h = np.zeros(hidden_size)
        self.bias_o = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, inputs):
        """inputs: list or array of length input_size"""
        x = np.array(inputs, dtype=np.float64)
        h = self.relu(self.weights_ih @ x + self.bias_h)
        o = self.sigmoid(self.weights_ho @ h + self.bias_o)
        return o

    @staticmethod
    def load_from_brain_data(data):
        """Load from the JSON brain format exported by the HTML trainer"""
        nn = NeuralNetwork(data['inputSize'], data['hiddenSize'], data['outputSize'])
        arr = data['weights']
        idx = 0
        # weights_ih: hidden_size x input_size
        h, i = nn.hidden_size, nn.input_size
        nn.weights_ih = np.array(arr[idx:idx + h * i]).reshape(h, i)
        idx += h * i
        # weights_ho: output_size x hidden_size
        o = nn.output_size
        nn.weights_ho = np.array(arr[idx:idx + o * h]).reshape(o, h)
        idx += o * h
        # bias_h
        nn.bias_h = np.array(arr[idx:idx + h])
        idx += h
        # bias_o
        nn.bias_o = np.array(arr[idx:idx + o])
        return nn


# ══════════════════════════════════════════════
#  SCREEN READER — Detect game elements
# ══════════════════════════════════════════════
class GameVision:
    """
    Reads the screen to find:
    - Bird position (y)
    - Bird velocity (from frame-to-frame y change)
    - Next pipe position (x, gap top, gap bottom)

    Uses color-based detection with OpenCV.
    Adjust BIRD_COLOR_RANGE and PIPE_COLOR_RANGE for your game.
    """

    def __init__(self, region=None, debug=False):
        """
        region: (left, top, width, height) — the game area on screen
        """
        self.region = region
        self.debug = debug
        self.prev_bird_y = None
        self.sct = mss.mss()

        # ═══ TUNE THESE for your specific Flappy Bird game ═══
        # Default: yellow bird (HSV range)
        self.bird_color_lower = np.array([20, 100, 100])
        self.bird_color_upper = np.array([40, 255, 255])

        # Green pipes (HSV range)
        self.pipe_color_lower = np.array([35, 80, 60])
        self.pipe_color_upper = np.array([85, 255, 255])

        # Bird detection: minimum contour area
        self.min_bird_area = 50
        self.max_bird_area = 2000

    def capture(self):
        """Capture the game region, returns BGR image"""
        if self.region:
            left, top, w, h = self.region
            monitor = {"left": left, "top": top, "width": w, "height": h}
        else:
            monitor = self.sct.monitors[1]  # primary monitor

        img = np.array(self.sct.grab(monitor))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return bgr

    def find_bird(self, frame):
        """
        Find the bird in the frame.
        Returns (x, y) center of bird or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.bird_color_lower, self.bird_color_upper)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_bird_area < area < self.max_bird_area:
                if area > best_area:
                    best_area = area
                    best = c

        if best is not None:
            M = cv2.moments(best)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if self.debug:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                return (cx, cy)

        return None

    def find_pipes(self, frame):
        """
        Find pipes in the frame.
        Returns list of dicts: {x, gap_top, gap_bottom, width}
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.pipe_color_lower, self.pipe_color_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pipes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:  # skip small noise
                continue
            x, y, w, h = cv2.boundingRect(c)
            pipes.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'contour': c
            })

        # Sort by x position (left to right)
        pipes.sort(key=lambda p: p['x'])

        # Identify gap: look for vertical separation between top and bottom pipe
        result = []
        if len(pipes) >= 2:
            # Group pipes that are at similar x positions (same pipe column)
            groups = []
            current_group = [pipes[0]]
            for i in range(1, len(pipes)):
                if abs(pipes[i]['x'] - current_group[0]['x']) < 30:
                    current_group.append(pipes[i])
                else:
                    groups.append(current_group)
                    current_group = [pipes[i]]
            groups.append(current_group)

            for group in groups:
                if len(group) >= 2:
                    # Sort by y: top pipe first
                    group.sort(key=lambda p: p['y'])
                    top_pipe = group[0]
                    bottom_pipe = group[-1]

                    gap_top = top_pipe['y'] + top_pipe['h']
                    gap_bottom = bottom_pipe['y']
                    pipe_x = top_pipe['x']

                    if gap_bottom > gap_top:
                        result.append({
                            'x': pipe_x,
                            'gap_top': gap_top,
                            'gap_bottom': gap_bottom,
                            'w': top_pipe['w']
                        })

                        if self.debug:
                            cv2.line(frame, (pipe_x, gap_top), (pipe_x + top_pipe['w'], gap_top), (255, 0, 0), 2)
                            cv2.line(frame, (pipe_x, gap_bottom), (pipe_x + top_pipe['w'], gap_bottom), (255, 0, 0), 2)

        return result

    def get_game_state(self):
        """
        Returns the 5 inputs for the neural network:
        [bird_y_norm, bird_vel, pipe_dist_norm, gap_top_norm, gap_bottom_norm]
        Returns None if game over detected.
        """
        frame = self.capture()
        h, w = frame.shape[:2]

        bird_pos = self.find_bird(frame)
        pipes = self.find_pipes(frame)

        if bird_pos is None:
            return None, frame, None, None

        bird_x, bird_y = bird_pos

        # Calculate velocity
        if self.prev_bird_y is not None:
            bird_vel = bird_y - self.prev_bird_y
        else:
            bird_vel = 0
        self.prev_bird_y = bird_y

        # Find next pipe ahead of bird
        next_pipe = None
        for p in pipes:
            if p['x'] + p['w'] > bird_x:
                next_pipe = p
                break

        if next_pipe is None:
            # No pipe visible — use defaults
            pipe_dist = w
            gap_top = h * 0.3
            gap_bottom = h * 0.7
        else:
            pipe_dist = next_pipe['x'] - bird_x
            gap_top = next_pipe['gap_top']
            gap_bottom = next_pipe['gap_bottom']

        # Normalize inputs (same as JS trainer)
        inputs = [
            bird_y / h,
            bird_vel / 10.0,
            pipe_dist / w,
            gap_top / h,
            gap_bottom / h
        ]

        return inputs, frame, bird_pos, next_pipe

    def detect_game_over(self, frame):
        """
        Detect if game is over. Override this for your specific game.
        Looks for common game-over indicators.
        """
        # Method 1: Check if bird is at the very bottom
        bird_pos = self.find_bird(frame)
        if bird_pos:
            _, by = bird_pos
            h = frame.shape[0]
            if by > h - 40:
                return True

        # Method 2: Look for "Game Over" text (template matching or OCR)
        # This is game-specific — add your own detection here
        return False

    def calibrate(self):
        """
        Interactive calibration: shows what the bot sees.
        Press 'q' to quit calibration.
        """
        print("=== CALIBRATION MODE ===")
        print("Adjust the game region so the bird and pipes are visible.")
        print("Press 'q' to quit calibration.")
        print("Press 'b' to adjust bird color, 'p' for pipe color.")

        while True:
            frame = self.capture()
            display = frame.copy()

            bird = self.find_bird(frame)
            pipes = self.find_pipes(frame)

            if bird:
                cv2.circle(display, bird, 10, (0, 255, 0), 2)
                cv2.putText(display, f"Bird: {bird}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for i, p in enumerate(pipes):
                cv2.rectangle(display, (p['x'], p['y']),
                            (p['x'] + p['w'], p['y'] + p['h']), (0, 0, 255), 2)
                cv2.putText(display, f"Pipe {i}", (p['x'], p['y'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.putText(display, f"Pipes found: {len(pipes)}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Resize for display
            scale = 0.7
            display = cv2.resize(display, None, fx=scale, fy=scale)
            cv2.imshow("Flappy Bot - Calibration", display)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


# ══════════════════════════════════════════════
#  BOT CONTROLLER
# ══════════════════════════════════════════════
class FlappyBot:
    def __init__(self, brain_path, game_region=None, flap_key='space',
                 flap_coords=None, debug=False, calibrate=False):
        """
        brain_path: path to JSON brain file
        game_region: (left, top, width, height) or None for auto
        flap_key: keyboard key to flap ('space', 'up', etc.)
        flap_coords: (x, y) screen coordinates to click for flap, or None for keyboard
        """
        # Load brain
        with open(brain_path, 'r') as f:
            data = json.load(f)

        self.brain = NeuralNetwork.load_from_brain_data(data)
        self.flap_key = flap_key
        self.flap_coords = flap_coords
        self.debug = debug
        self.paused = False
        self.score = 0
        self.frames = 0

        print(f"🧠 Loaded brain: Gen {data.get('generation', '?')}, "
              f"Best Score: {data.get('bestScore', '?')}, "
              f"Fitness: {data.get('bestFitness', '?'):.1f}")
        print(f"   Network: {data['inputSize']}→{data['hiddenSize']}→{data['outputSize']}")

        # Initialize vision
        self.vision = GameVision(region=game_region, debug=debug)

        if calibrate:
            self.vision.calibrate()

    def flap(self):
        """Send flap input to the game"""
        if self.flap_coords:
            pyautogui.click(self.flap_coords[0], self.flap_coords[1])
        else:
            pyautogui.press(self.flap_key)

    def run(self, max_games=100):
        """Main loop: play the game"""
        print("\n🎮 Starting bot...")
        print("   Press 'p' in console to pause, 'q' to quit")
        print("   (The bot uses screen input, so focus the game window)\n")

        game_count = 0
        running = True

        # Give user time to focus the game window
        print("Starting in 3 seconds... focus the game window!")
        time.sleep(3)

        while running and game_count < max_games:
            game_count += 1
            print(f"\n--- Game {game_count} ---")
            self.play_one_game()

            # Brief pause between games
            print("Waiting 2 seconds before next game...")
            time.sleep(2)

            # Click restart if needed
            if self.flap_coords:
                pyautogui.click(self.flap_coords[0], self.flap_coords[1])
            else:
                pyautogui.press(self.flap_key)

            time.sleep(0.5)

        print(f"\n🏁 Finished {game_count} games!")

    def play_one_game(self):
        """Play a single game until death"""
        self.vision.prev_bird_y = None
        self.frames = 0
        consecutive_failures = 0

        while True:
            self.frames += 1

            # Get game state
            result = self.vision.get_game_state()
            inputs, frame, bird_pos, next_pipe = result

            if inputs is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print(f"  💀 Lost bird tracking. Game over. Frames: {self.frames}")
                    break
                time.sleep(0.01)
                continue

            consecutive_failures = 0

            # Neural network decision
            output = self.brain.forward(inputs)
            should_flap = output[0] > 0.5

            if should_flap:
                self.flap()

            # Check game over
            if self.vision.detect_game_over(frame):
                print(f"  💀 Game over detected. Frames survived: {self.frames}")
                break

            # Debug display
            if self.debug and self.frames % 10 == 0:
                print(f"  Frame {self.frames}: bird_y={inputs[0]:.3f}, "
                      f"vel={inputs[1]:.3f}, pipe_dist={inputs[2]:.3f}, "
                      f"gap=({inputs[3]:.3f},{inputs[4]:.3f}) → "
                      f"flap={'YES' if should_flap else 'no'} ({output[0]:.3f})")

            # Small delay to not overwhelm
            time.sleep(0.005)


# ══════════════════════════════════════════════
#  AUTO-DETECT GAME REGION
# ══════════════════════════════════════════════
def auto_detect_region():
    """
    Try to auto-detect the game window.
    Looks for a window with 'flappy' in the title.
    """
    try:
        import subprocess
        # On Windows, use pygetwindow if available
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle('')
            for w in windows:
                if 'flappy' in w.title.lower() or 'flappybird' in w.title.lower():
                    print(f"Found game window: '{w.title}' at ({w.left}, {w.top}, {w.width}, {w.height})")
                    return (w.left, w.top, w.width, w.height)
        except ImportError:
            pass
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Flappy Bird AI Bot')
    parser.add_argument('--brain', required=True, help='Path to brain JSON file')
    parser.add_argument('--region', nargs=4, type=int, metavar=('L', 'T', 'W', 'H'),
                       help='Game region: left top width height')
    parser.add_argument('--click', nargs=2, type=int, metavar=('X', 'Y'),
                       help='Screen coordinates to click for flap')
    parser.add_argument('--key', default='space', help='Keyboard key for flap (default: space)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration mode first')
    parser.add_argument('--games', type=int, default=100, help='Max games to play')
    parser.add_argument('--auto-region', action='store_true', help='Auto-detect game window')

    args = parser.parse_args()

    if not os.path.exists(args.brain):
        print(f"ERROR: Brain file not found: {args.brain}")
        sys.exit(1)

    # Determine game region
    game_region = None
    if args.region:
        game_region = tuple(args.region)
    elif args.auto_region:
        game_region = auto_detect_region()

    # Determine flap method
    flap_coords = None
    flap_key = args.key
    if args.click:
        flap_coords = tuple(args.click)
        flap_key = None

    # Create and run bot
    bot = FlappyBot(
        brain_path=args.brain,
        game_region=game_region,
        flap_key=flap_key,
        flap_coords=flap_coords,
        debug=args.debug,
        calibrate=args.calibrate
    )

    bot.run(max_games=args.games)


if __name__ == '__main__':
    main()
