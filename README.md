# 👁️ Blink to Jump

A fun side-scrolling game where you **blink your eyes to make the character jump** over obstacles! Built with Python using real-time blink detection via your webcam.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-red)

## How It Works

1. **Webcam captures your face** in real time using OpenCV.
2. **MediaPipe Face Mesh** tracks 468 facial landmarks to locate your eyes.
3. The **Eye Aspect Ratio (EAR)** is computed each frame — when both eyes close (EAR drops below a threshold) and reopen, a blink is registered.
4. Each blink triggers a **jump** in the Pygame side-scroller, letting you dodge incoming obstacles.

## Demo

```
┌─────────────────────────────────┐
│  Blink to Jump!    Score: 42    │
│                                 │
│         ██                      │
│        ████          ██         │
│  ────────────────────────────── │
└─────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.10+
- A working webcam

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

A webcam window will open showing your face with the EAR value, and the game window will launch alongside it.

## Controls

| Action | Input |
|---|---|
| **Jump** | Blink your eyes |
| **Restart** | Press `R` (after game over) |
| **Quit** | Press `Q` or close the window |

## Project Structure

```
BlinktoJump/
├── app.py              # Main game + blink detection logic
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Tech Stack

- **[Pygame](https://www.pygame.org/)** — Game rendering and loop
- **[OpenCV](https://opencv.org/)** — Webcam capture and image processing
- **[MediaPipe](https://mediapipe.dev/)** — Real-time face mesh and landmark detection

## How Blink Detection Works

The **Eye Aspect Ratio (EAR)** is calculated from 6 landmarks per eye:

$$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- When eyes are open, EAR is roughly 0.25–0.30.
- When eyes close, EAR drops below **0.21** (the threshold).
- A blink is registered when the eyes close for ≥ 2 consecutive frames and then reopen.

## License

This project is open source and available under the [MIT License](LICENSE).
