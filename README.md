# 👁️ Blink to Jump

A polished side-scrolling runner where you **blink your eyes to make the character jump** over obstacles — powered by real-time computer vision through your webcam.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-red)

## ✨ Features

- **Blink-to-jump controls** — close and reopen your eyes to make the character jump
- **Keyboard fallback** — press Space to jump if no webcam is available
- **Animated character** — a little creature with running legs, spikes, a tail, and an eye that mirrors *your* real eyes (closes when you blink!)
- **Day / night cycle** — the sky transitions between day and night with sun, moon, twinkling stars, and drifting clouds
- **Parallax scrolling** — layered hills and ground pebbles scroll at different speeds
- **Two obstacle types** — cacti (with arms & ribs) and rocky boulders, spawning in groups at higher speeds
- **Particle effects** — dust on jump/landing, explosion on crash
- **Screen shake** — impactful feedback on collision
- **Procedural sound effects** — jump, score milestone, and crash sounds generated with NumPy (no audio files needed)
- **Live webcam HUD** — in-game camera preview with eye-landmark overlay and an EAR meter bar
- **Adjustable sensitivity** — tune the EAR blink threshold in real time with ↑/↓ keys
- **High score tracking** — session best score displayed on the HUD and game-over screen
- **Score milestones** — score flashes every 100 points and game speed increases progressively
- **Pause & mute** — press P to pause, M to mute sounds
- **Forgiving hitboxes** — slightly shrunken collision rects so near-misses feel fair

## 🎮 How It Works

1. **Webcam captures your face** in real time using OpenCV.
2. **MediaPipe Face Mesh** tracks 468 facial landmarks to locate your eyes.
3. The **Eye Aspect Ratio (EAR)** is computed each frame — when both eyes close (EAR drops below the threshold) and reopen, a blink is registered.
4. Each blink triggers a **jump** in the Pygame runner, letting you dodge incoming obstacles.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A working webcam (optional — keyboard controls available)

### Installation

```bash
# Clone the repo
git clone https://github.com/exassaro/BlinktoJump.git
cd BlinktoJump

# Create a virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Game

```bash
python app.py
```

The game window opens with the runner, a live camera preview in the bottom-right corner, and an EAR meter in the bottom-left.

## 🎹 Controls

| Action | Input |
|---|---|
| **Jump** | Blink your eyes / `Space` |
| **Restart** | Blink (after delay) / `R` |
| **Pause / Resume** | `P` |
| **Mute / Unmute** | `M` |
| **Increase EAR threshold** | `↑` |
| **Decrease EAR threshold** | `↓` |
| **Quit** | `Q` / `Esc` |

## 📁 Project Structure

```
BlinktoJump/
├── app.py              # All game logic, rendering, blink detection & sound
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| **[Pygame](https://www.pygame.org/)** | Game loop, rendering, input, and audio playback |
| **[OpenCV](https://opencv.org/)** | Webcam capture and image processing |
| **[MediaPipe](https://mediapipe.dev/)** | Real-time face mesh and landmark detection |
| **[NumPy](https://numpy.org/)** | Procedural sound-effect generation |

## 🧠 How Blink Detection Works

The **Eye Aspect Ratio (EAR)** is calculated from 6 landmarks per eye:

$$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- When eyes are open, EAR is roughly 0.25–0.30.
- When eyes close, EAR drops below the threshold (default **0.21**, adjustable with ↑/↓).
- A blink is registered when the eyes close for ≥ 2 consecutive frames and then reopen.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
