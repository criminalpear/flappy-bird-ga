# 🐦 Flappy Bird — Genetic Algorithm Trainer

A self-contained Flappy Bird clone with a built-in genetic algorithm that evolves neural networks to play the game. Includes a Python screen-reading bot to apply trained brains to the real game.

## Quick Start

### 1. Train a Brain (Browser)
Open `index.html` in any browser — no server needed.

1. Press **▶ Start** to begin training
2. Crank **Speed** to 50x for fast results
3. Watch the fitness graph climb
4. Click **💾 Save Brain** when you're happy with the score

### 2. Play the Real Game (Python Bot)

```bash
pip install mss numpy pyautogui opencv-python

# Keyboard control (space to flap)
python flappy_bot.py --brain flappy_brain_gen50_score10.json --key space

# Click control (for browser games)
python flappy_bot.py --brain brain.json --click 500 400

# Calibration mode (tune detection for your game)
python flappy_bot.py --brain brain.json --calibrate
```

## How It Works

### Neural Network
- **5 inputs:** bird Y, velocity, distance to next pipe, gap top, gap bottom
- **8 hidden neurons** (ReLU)
- **1 output** (sigmoid → flap if > 0.5)

### Genetic Algorithm
- Population: 60 birds
- Tournament selection (k=5)
- Uniform crossover + Gaussian mutation (12% rate)
- Elitism: top 4 survive unchanged

### Fitness / Rewards
| Action | Fitness |
|---|---|
| Staying alive (per frame) | +0.1 |
| Passing a pipe | +50 |
| Off-center from gap | proportional penalty |
| Hitting ceiling | -10 |
| Hitting ground | -5 |

### Screen-Reading Bot
Uses `mss` + OpenCV for real-time screen capture and color-based detection of the bird and pipes. Feeds the same 5 normalized inputs to the trained neural network.

## Files
| File | Purpose |
|---|---|
| `index.html` | Game + GA trainer (open in browser) |
| `flappy_bot.py` | Python screen-reading bot |

