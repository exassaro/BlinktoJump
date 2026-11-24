import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math

# =====================================================
# 1. CONFIGURATION
# =====================================================

WIDTH = 900
HEIGHT = 500
GROUND_Y = 400
FPS = 60

PLAYER_W = 45
PLAYER_H = 60

# Webcam preview is drawn inside the game window.
# Set to True if you also want the separate OpenCV window.
SHOW_CV_WINDOW = False
PREVIEW_W = 108

EAR_MAX_DISPLAY = 0.40   # full length of the EAR meter bar

# Day / night colour themes
DAY = {
    "sky_top": (110, 185, 245),
    "sky_bottom": (225, 242, 255),
    "hill_far": (165, 200, 225),
    "hill_near": (135, 190, 145),
    "ground": (226, 204, 152),
    "ground_dark": (200, 175, 120),
    "ground_line": (115, 90, 55),
    "pebble": (180, 150, 100),
    "cloud": (255, 255, 255),
    "cactus": (52, 140, 64),
    "cactus_dark": (34, 100, 46),
    "rock": (150, 140, 130),
    "rock_dark": (110, 100, 92),
    "text": (40, 42, 54),
    "text_dim": (110, 112, 125),
}

NIGHT = {
    "sky_top": (10, 15, 40),
    "sky_bottom": (45, 52, 100),
    "hill_far": (38, 46, 82),
    "hill_near": (28, 42, 58),
    "ground": (62, 58, 74),
    "ground_dark": (48, 44, 60),
    "ground_line": (22, 22, 32),
    "pebble": (84, 80, 98),
    "cloud": (70, 80, 120),
    "cactus": (40, 96, 72),
    "cactus_dark": (24, 66, 48),
    "rock": (92, 90, 105),
    "rock_dark": (66, 64, 80),
    "text": (232, 232, 245),
    "text_dim": (150, 152, 175),
}

PLAYER_BODY = (0, 150, 80)
PLAYER_DARK = (0, 105, 55)
PLAYER_BELLY = (150, 225, 160)
PLAYER_SPIKE = (255, 190, 60)

STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_PAUSED = "paused"
STATE_GAME_OVER = "game_over"

# =====================================================
# 2. INITIALIZATION
# =====================================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blink to Jump - CV Game")

canvas = pygame.Surface((WIDTH, HEIGHT))   # world is drawn here (for screen shake)
clock = pygame.time.Clock()


def load_font(size, bold=False, mono=False):
    if mono:
        names = "consolas,dejavusansmono,couriernew,monospace"
    else:
        names = "segoeui,arial,helvetica,dejavusans"
    return pygame.font.SysFont(names, size, bold=bold)


font_tiny = load_font(15)
font_small = load_font(18, bold=True)
font = load_font(26, bold=True)
font_score = load_font(28, bold=True, mono=True)
big_font = load_font(62, bold=True)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not detected!")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indexes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

ear_threshold = 0.21
closed_frames = 0
blink_pending = False

# Camera / face state (used by the HUD)
current_ear = 0.0
display_ear = 0.0
face_found = False
eyes_closed = False
cam_surface = None

# =====================================================
# 3. SOUND (generated with numpy - no sound files)
# =====================================================

muted = False
sounds = {}


def make_tone(f_start, f_end, duration, volume=0.3, wave="sine"):
    init = pygame.mixer.get_init()

    if not init:
        return None

    rate, _, channels = init
    n = int(rate * duration)

    freqs = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(freqs) / rate

    if wave == "square":
        samples = np.sign(np.sin(phase))
    elif wave == "noise":
        samples = np.random.uniform(-1, 1, n)
    else:
        samples = np.sin(phase)

    envelope = np.linspace(1, 0, n) ** 1.5
    envelope[: max(1, n // 50)] *= np.linspace(0, 1, max(1, n // 50))

    data = (samples * envelope * volume * 32767).astype(np.int16)

    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)

    return pygame.sndarray.make_sound(np.ascontiguousarray(data))


try:
    sounds["jump"] = make_tone(420, 820, 0.12, 0.25)
    sounds["point"] = make_tone(880, 1320, 0.18, 0.18, "square")
    sounds["crash"] = make_tone(300, 60, 0.35, 0.35, "noise")
    sounds["start"] = make_tone(520, 1040, 0.20, 0.20)
except Exception:
    sounds = {}


def play_sound(name):
    sound = sounds.get(name)

    if sound and not muted:
        sound.play()


# =====================================================
# 4. BLINK DETECTION
# =====================================================

def calculate_distance(point1, point2):
    return math.dist(point1, point2)


def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:
        landmark = landmarks[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    vertical_1 = calculate_distance(points[1], points[5])
    vertical_2 = calculate_distance(points[2], points[4])

    horizontal = calculate_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2 * horizontal)


def detect_blink():
    """Reads one webcam frame, updates the face/EAR state and the
    in-game preview. Returns True when a full blink (close -> open) happens."""

    global closed_frames, blink_pending
    global current_ear, face_found, eyes_closed, cam_surface

    success, frame = cap.read()

    if not success:
        face_found = False
        return False

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    blink_detected = False

    h, w, _ = frame.shape

    face_found = bool(results.multi_face_landmarks)

    if face_found:

        face = results.multi_face_landmarks[0]

        left_ear = calculate_ear(face.landmark, LEFT_EYE, w, h)
        right_ear = calculate_ear(face.landmark, RIGHT_EYE, w, h)

        current_ear = (left_ear + right_ear) / 2

        eyes_closed = current_ear < ear_threshold

        # Check whether eyes are closed
        if eyes_closed:

            closed_frames += 1

            if closed_frames >= 2:
                blink_pending = True

        else:

            # Eyes reopened after closing
            if blink_pending:
                blink_detected = True

            closed_frames = 0
            blink_pending = False

        # Draw eye landmarks on the preview
        dot_color = (255, 90, 90) if eyes_closed else (90, 255, 150)
        radius = max(2, w // 130)

        for index in LEFT_EYE + RIGHT_EYE:
            lm = face.landmark[index]
            cv2.circle(rgb_frame, (int(lm.x * w), int(lm.y * h)), radius, dot_color, -1)

    else:
        closed_frames = 0
        blink_pending = False
        eyes_closed = False

    # Convert frame to a small pygame surface for the in-game preview
    preview_h = int(PREVIEW_W * h / w)
    small = cv2.resize(rgb_frame, (PREVIEW_W, preview_h))
    cam_surface = pygame.image.frombuffer(small.tobytes(), (PREVIEW_W, preview_h), "RGB")

    if SHOW_CV_WINDOW:
        cv2.imshow("Webcam - Blink Detection", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    return blink_detected


# =====================================================
# 5. DRAWING HELPERS
# =====================================================

def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


def theme(key):
    return lerp_color(DAY[key], NIGHT[key], night_t)


def make_gradient(top, bottom):
    surf = pygame.Surface((WIDTH, HEIGHT))

    for y in range(HEIGHT):
        pygame.draw.line(surf, lerp_color(top, bottom, y / HEIGHT), (0, y), (WIDTH, y))

    return surf


def draw_panel(surf, rect, color=(18, 22, 34), alpha=165, radius=14, border=None):
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=radius)

    if border:
        pygame.draw.rect(panel, (*border, 200), panel.get_rect(), 2, border_radius=radius)

    surf.blit(panel, rect.topleft)


def draw_text(surf, text, fnt, color, pos, anchor="topleft", shadow=True, alpha=255):
    rendered = fnt.render(text, True, color)
    rect = rendered.get_rect(**{anchor: pos})

    if shadow:
        shade = fnt.render(text, True, (0, 0, 0))
        shade.set_alpha(int(90 * alpha / 255))
        surf.blit(shade, (rect.x + 2, rect.y + 2))

    if alpha < 255:
        rendered.set_alpha(alpha)

    surf.blit(rendered, rect)
    return rect


# Pre-rendered sky, sun and moon
day_sky = make_gradient(DAY["sky_top"], DAY["sky_bottom"])
night_sky = make_gradient(NIGHT["sky_top"], NIGHT["sky_bottom"])

sun_surf = pygame.Surface((120, 120), pygame.SRCALPHA)
for r, a in ((58, 25), (48, 45), (40, 80)):
    pygame.draw.circle(sun_surf, (255, 225, 120, a), (60, 60), r)
pygame.draw.circle(sun_surf, (255, 214, 90, 255), (60, 60), 32)

moon_surf = pygame.Surface((120, 120), pygame.SRCALPHA)
pygame.draw.circle(moon_surf, (240, 240, 255, 30), (60, 60), 44)
pygame.draw.circle(moon_surf, (238, 236, 220, 255), (60, 60), 26)
pygame.draw.circle(moon_surf, (0, 0, 0, 0), (72, 52), 22)   # carve the crescent

stars = [
    (random.randint(0, WIDTH), random.randint(10, GROUND_Y - 150), random.random() * 6.28)
    for _ in range(70)
]

clouds = [
    [random.randint(0, WIDTH), random.randint(40, 190), random.uniform(0.6, 1.2)]
    for _ in range(5)
]

pebbles = [
    [random.randint(0, WIDTH), random.randint(GROUND_Y + 6, HEIGHT - 6), random.randint(2, 6)]
    for _ in range(40)
]


def draw_background(surf):
    surf.blit(day_sky, (0, 0))

    if night_t > 0.01:
        night_sky.set_alpha(int(255 * night_t))
        surf.blit(night_sky, (0, 0))

        # Twinkling stars
        sky_col = lerp_color(DAY["sky_top"], NIGHT["sky_top"], night_t)
        for sx, sy, ph in stars:
            twinkle = 0.6 + 0.4 * math.sin(frame_count * 0.05 + ph)
            col = lerp_color(sky_col, (255, 255, 255), night_t * twinkle)
            surf.fill(col, (sx, sy, 2, 2))

    # Sun & moon cross-fade
    if night_t < 0.99:
        sun_surf.set_alpha(int(255 * (1 - night_t)))
        surf.blit(sun_surf, (WIDTH - 200, 30))
    if night_t > 0.01:
        moon_surf.set_alpha(int(255 * night_t))
        surf.blit(moon_surf, (WIDTH - 200, 30))

    # Clouds
    cloud_col = theme("cloud")
    for cx, cy, s in clouds:
        pygame.draw.circle(surf, cloud_col, (int(cx), int(cy)), int(18 * s))
        pygame.draw.circle(surf, cloud_col, (int(cx + 22 * s), int(cy - 10 * s)), int(22 * s))
        pygame.draw.circle(surf, cloud_col, (int(cx + 46 * s), int(cy)), int(17 * s))
        pygame.draw.rect(surf, cloud_col, (int(cx), int(cy), int(46 * s), int(17 * s)),
                         border_radius=int(8 * s))

    # Parallax hills
    draw_hills(surf, world_x * 0.10, GROUND_Y - 70, 45, 0.006, theme("hill_far"), 1.3)
    draw_hills(surf, world_x * 0.30, GROUND_Y - 25, 30, 0.011, theme("hill_near"), 4.1)

    # Ground
    pygame.draw.rect(surf, theme("ground"), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.rect(surf, theme("ground_dark"), (0, GROUND_Y, WIDTH, 8))

    pebble_col = theme("pebble")
    for px, py, pw in pebbles:
        pygame.draw.rect(surf, pebble_col, (int(px), py, pw * 2, pw // 2 + 1), border_radius=2)

    pygame.draw.line(surf, theme("ground_line"), (0, GROUND_Y), (WIDTH, GROUND_Y), 3)


def draw_hills(surf, offset, base_y, amp, freq, color, seed):
    points = [(0, GROUND_Y)]

    for x in range(0, WIDTH + 21, 20):
        wx = x + offset
        y = base_y - amp * (0.6 * math.sin(wx * freq + seed)
                            + 0.4 * math.sin(wx * freq * 2.3 + seed * 1.7))
        points.append((x, y))

    points.append((WIDTH, GROUND_Y))
    pygame.draw.polygon(surf, color, points)


# =====================================================
# 6. PLAYER
# =====================================================

player = pygame.Rect(100, GROUND_Y - PLAYER_H, PLAYER_W, PLAYER_H)

player_velocity = 0
gravity = 0.8
jump_strength = -16

on_ground = True
squash = 0.0          # >0 squashed (landing), <0 stretched (jumping)
run_anim = 0


def try_jump():
    global player_velocity, squash, on_ground

    # Jump only when player is on ground
    if player.bottom >= GROUND_Y:
        player_velocity = jump_strength
        squash = -0.22
        on_ground = False
        spawn_dust(player.centerx, GROUND_Y, 8)
        play_sound("jump")


def draw_player(surf):
    w = int(PLAYER_W * (1 + squash))
    h = int(PLAYER_H * (1 - squash))
    x = player.centerx - w // 2
    leg_h = 10

    body = pygame.Rect(x, player.bottom - h, w, h - leg_h)

    # Shadow (shrinks while in the air)
    height_above = GROUND_Y - player.bottom
    sw = max(14, int(52 - height_above * 0.18))
    shadow = pygame.Surface((sw, 10), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 60), shadow.get_rect())
    surf.blit(shadow, (player.centerx - sw // 2, GROUND_Y - 5))

    # Tail
    pygame.draw.polygon(surf, PLAYER_BODY, [
        (body.left + 6, body.bottom - 24),
        (body.left - 13, body.bottom - 6),
        (body.left + 6, body.bottom - 4),
    ])

    # Legs (running animation)
    if on_ground and state == STATE_PLAYING:
        phase = (run_anim // 6) % 2
        leg1 = leg_h if phase == 0 else leg_h - 5
        leg2 = leg_h - 5 if phase == 0 else leg_h
    else:
        leg1 = leg2 = leg_h - 3

    pygame.draw.rect(surf, PLAYER_DARK, (body.x + 7, body.bottom - 3, 10, leg1 + 3), border_radius=3)
    pygame.draw.rect(surf, PLAYER_DARK, (body.right - 18, body.bottom - 3, 10, leg2 + 3), border_radius=3)

    # Back spikes
    for i in range(3):
        sx = body.x + 4 + i * 9
        pygame.draw.polygon(surf, PLAYER_SPIKE, [(sx, body.y + 6), (sx + 4, body.y - 5), (sx + 8, body.y + 6)])

    # Body + belly
    pygame.draw.rect(surf, PLAYER_BODY, body, border_radius=12)
    pygame.draw.rect(surf, PLAYER_DARK, body, 2, border_radius=12)

    belly = pygame.Rect(body.x + int(w * 0.42), body.y + int(body.h * 0.48),
                        int(w * 0.48), int(body.h * 0.42))
    pygame.draw.rect(surf, PLAYER_BELLY, belly, border_radius=8)

    # Eye - mirrors the player's real eyes!
    ex, ey = body.x + int(w * 0.70), body.y + 15

    if state == STATE_GAME_OVER:
        pygame.draw.line(surf, (30, 30, 30), (ex - 5, ey - 5), (ex + 5, ey + 5), 3)
        pygame.draw.line(surf, (30, 30, 30), (ex - 5, ey + 5), (ex + 5, ey - 5), 3)
    elif eyes_closed and face_found:
        pygame.draw.line(surf, (30, 30, 30), (ex - 7, ey + 1), (ex + 7, ey + 1), 3)
    else:
        look = -2 if player_velocity < 0 else (2 if player_velocity > 0 else 0)
        pygame.draw.circle(surf, (255, 255, 255), (ex, ey), 7)
        pygame.draw.circle(surf, (25, 25, 35), (ex + 2, ey + look), 3)
        pygame.draw.circle(surf, (255, 255, 255), (ex + 3, ey + look - 1), 1)

    # Cheek + mouth
    pygame.draw.circle(surf, (255, 150, 150), (body.right - 7, body.y + 27), 3)
    pygame.draw.line(surf, PLAYER_DARK, (body.right - 14, body.y + 31), (body.right - 4, body.y + 31), 2)


# =====================================================
# 7. OBSTACLES
# =====================================================

obstacles = []
next_gap = 350


def spawn_obstacle():
    global next_gap

    trunks = []

    if random.random() < 0.25:
        # Rock
        w = random.randint(36, 50)
        h = random.randint(24, 34)
        rect = pygame.Rect(WIDTH, GROUND_Y - h, w, h)
        poly = []
        for i in range(9):
            ang = math.pi + i * math.pi / 8
            jitter = random.uniform(0.85, 1.0)
            poly.append((w / 2 + math.cos(ang) * w / 2 * jitter,
                         h + math.sin(ang) * h * jitter))
        trunks.append({"rect": rect, "poly": poly})
        obstacles.append({"kind": "rock", "trunks": trunks})

    else:
        # Cactus group (more cacti at higher speeds)
        max_count = 1 if game_speed < 7 else (2 if game_speed < 10 else 3)
        count = random.randint(1, max_count)
        x = WIDTH

        for _ in range(count):
            obstacle_height = random.randint(35, 65)
            tw = random.randint(20, 26)
            rect = pygame.Rect(x, GROUND_Y - obstacle_height, tw, obstacle_height)
            trunks.append({
                "rect": rect,
                "la": random.uniform(0.35, 0.6) if random.random() < 0.7 else None,
                "ra": random.uniform(0.25, 0.5) if random.random() < 0.7 else None,
            })
            x += tw + 6

        obstacles.append({"kind": "cactus", "trunks": trunks})

    next_gap = random.randint(300, 450) + (game_speed - 6) * 25


def draw_obstacle(surf, ob):
    if ob["kind"] == "rock":
        t = ob["trunks"][0]
        r = t["rect"]
        pts = [(r.x + px, r.y + py) for px, py in t["poly"]]
        pygame.draw.polygon(surf, theme("rock"), pts)
        pygame.draw.polygon(surf, theme("rock_dark"), pts, 2)
        pygame.draw.circle(surf, lerp_color(theme("rock"), (255, 255, 255), 0.3),
                           (r.x + r.w // 3, r.y + r.h // 3 + 2), 3)
        return

    base = theme("cactus")
    dark = theme("cactus_dark")

    for t in ob["trunks"]:
        r = t["rect"]
        arm = max(6, r.w // 3)

        for side, frac in (("l", t["la"]), ("r", t["ra"])):
            if frac is None:
                continue
            ay = r.y + int(r.h * frac)
            top = max(r.y + 4, ay - arm * 2)
            ax = r.x - arm - 1 if side == "l" else r.right + 1
            hx = r.x - arm - 1 if side == "l" else r.right - 3
            pygame.draw.rect(surf, base, (hx, ay, arm + 4, arm), border_radius=3)
            pygame.draw.rect(surf, base, (ax, top, arm, ay + arm - top), border_radius=arm // 2)
            pygame.draw.rect(surf, dark, (ax, top, arm, ay + arm - top), 1, border_radius=arm // 2)

        pygame.draw.rect(surf, base, r, border_top_left_radius=r.w // 2, border_top_right_radius=r.w // 2)
        pygame.draw.rect(surf, dark, r, 2, border_top_left_radius=r.w // 2, border_top_right_radius=r.w // 2)

        # Ribs
        for i in (r.w // 3, 2 * r.w // 3):
            pygame.draw.line(surf, dark, (r.x + i, r.y + 8), (r.x + i, r.bottom - 4), 1)


# =====================================================
# 8. PARTICLES
# =====================================================

particles = []


def spawn_dust(x, y, count):
    for _ in range(count):
        particles.append({
            "x": x + random.uniform(-15, 15), "y": y - 2,
            "vx": random.uniform(-2.5, 0.5), "vy": random.uniform(-2.5, -0.5),
            "life": 25, "max": 25, "size": random.uniform(2, 4),
            "color": theme("ground_dark"),
        })


def spawn_explosion(x, y):
    for _ in range(35):
        ang = random.uniform(0, math.tau)
        spd = random.uniform(2, 7)
        particles.append({
            "x": x, "y": y,
            "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd - 2,
            "life": 45, "max": 45, "size": random.uniform(3, 6),
            "color": random.choice([PLAYER_BODY, PLAYER_SPIKE, (255, 110, 70), PLAYER_BELLY]),
        })


def update_particles():
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.25
        p["life"] -= 1

    particles[:] = [p for p in particles if p["life"] > 0]


def draw_particles(surf):
    for p in particles:
        size = max(1, int(p["size"] * p["life"] / p["max"]))
        pygame.draw.circle(surf, p["color"], (int(p["x"]), int(p["y"])), size)


# =====================================================
# 9. HUD & SCREENS
# =====================================================

def draw_hud(surf):
    text_col = theme("text")
    dim_col = theme("text_dim")

    # Score (top-right)  "HI 00123  00045"
    shown = score // 10
    visible = flash_timer <= 0 or (flash_timer // 8) % 2 == 0

    if visible:
        score_rect = draw_text(surf, f"{shown:05d}", font_score, text_col,
                               (WIDTH - 24, 22), "topright", shadow=False)
    else:
        score_rect = font_score.render("00000", True, text_col).get_rect(topright=(WIDTH - 24, 22))

    if high_score > 0:
        draw_text(surf, f"HI {high_score:05d}", font_score, dim_col,
                  (score_rect.left - 18, 22), "topright", shadow=False)

    # Level chip (top-left)
    if state in (STATE_PLAYING, STATE_PAUSED, STATE_GAME_OVER):
        chip = pygame.Rect(16, 16, 128, 34)
        draw_panel(surf, chip, radius=17)
        draw_text(surf, f"LEVEL {game_speed - 5}", font_small, (255, 214, 90),
                  chip.center, "center", shadow=False)

    if muted:
        draw_text(surf, "MUTED (M)", font_tiny, dim_col, (16, 58), shadow=False)

    draw_eye_meter(surf)
    draw_camera_preview(surf)

    # Control hints (bottom-centre)
    draw_text(surf, "SPACE jump  |  P pause  |  M mute  |  Q quit", font_tiny,
              lerp_color(theme("ground_line"), (255, 255, 255), night_t * 0.6),
              (WIDTH // 2 + 20, HEIGHT - 14), "midbottom", shadow=False)


def eye_status():
    if not face_found:
        return "NO FACE DETECTED", (255, 120, 90)
    if eyes_closed:
        return "EYES CLOSED", (255, 205, 80)
    return "EYES OPEN", (110, 230, 140)


def draw_eye_meter(surf):
    rect = pygame.Rect(16, GROUND_Y + 12, 280, 78)
    draw_panel(surf, rect)

    status, col = eye_status()
    pygame.draw.circle(surf, col, (rect.x + 18, rect.y + 17), 6)
    draw_text(surf, status, font_small, col, (rect.x + 32, rect.y + 6), shadow=False)

    # EAR bar
    bar = pygame.Rect(rect.x + 12, rect.y + 34, rect.w - 24, 12)
    pygame.draw.rect(surf, (55, 60, 78), bar, border_radius=6)

    if face_found:
        fill = min(1.0, display_ear / EAR_MAX_DISPLAY)
        fill_rect = pygame.Rect(bar.x, bar.y, max(6, int(bar.w * fill)), bar.h)
        pygame.draw.rect(surf, col, fill_rect, border_radius=6)

    tx = bar.x + int(bar.w * min(1.0, ear_threshold / EAR_MAX_DISPLAY))
    pygame.draw.line(surf, (255, 255, 255), (tx, bar.y - 4), (tx, bar.bottom + 4), 2)

    ear_txt = f"{display_ear:.2f}" if face_found else "--"
    draw_text(surf, f"EAR {ear_txt}    Threshold {ear_threshold:.2f}  (UP/DOWN)",
              font_tiny, (200, 205, 220), (rect.x + 12, rect.y + 54), shadow=False)


def draw_camera_preview(surf):
    if cam_surface is None:
        return

    pw, ph = cam_surface.get_size()
    rect = pygame.Rect(WIDTH - pw - 24, GROUND_Y + 6, pw + 8, ph + 8)
    _, col = eye_status()

    draw_panel(surf, rect, radius=10, border=col)
    surf.blit(cam_surface, (rect.x + 4, rect.y + 4))


def draw_menu(surf):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 60))
    surf.blit(overlay, (0, 0))

    panel = pygame.Rect(WIDTH // 2 - 280, 70, 560, 250)
    draw_panel(surf, panel, alpha=190, radius=22, border=(255, 214, 90))

    bob = int(math.sin(frame_count * 0.06) * 4)
    draw_text(surf, "BLINK TO JUMP", big_font, (255, 214, 90), (WIDTH // 2, panel.y + 52 + bob), "center")
    draw_text(surf, "a computer-vision runner", font_tiny, (190, 195, 215),
              (WIDTH // 2, panel.y + 95), "center", shadow=False)

    pulse = int(160 + 95 * math.sin(frame_count * 0.1))
    draw_text(surf, "Blink to start", font, (255, 255, 255), (WIDTH // 2, panel.y + 140),
              "center", alpha=pulse)

    draw_text(surf, "Close your eyes briefly, then open them to jump.", font_tiny,
              (210, 215, 230), (WIDTH // 2, panel.y + 182), "center", shadow=False)
    draw_text(surf, "No face detected? Press SPACE to play with the keyboard.", font_tiny,
              (210, 215, 230), (WIDTH // 2, panel.y + 204), "center", shadow=False)

    if high_score > 0:
        draw_text(surf, f"BEST  {high_score}", font_small, (255, 214, 90),
                  (WIDTH // 2, panel.y + 230), "center", shadow=False)


def draw_game_over(surf):
    fade = min(1.0, game_over_timer / 20)

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, int(110 * fade)))
    surf.blit(overlay, (0, 0))

    panel = pygame.Rect(WIDTH // 2 - 230, 80 - int((1 - fade) * 40), 460, 230)
    draw_panel(surf, panel, alpha=int(200 * fade), radius=22, border=(230, 70, 70))

    alpha = int(255 * fade)
    draw_text(surf, "GAME OVER", big_font, (240, 80, 80), (WIDTH // 2, panel.y + 50), "center", alpha=alpha)

    draw_text(surf, f"SCORE  {score // 10}", font, (255, 255, 255),
              (WIDTH // 2 - 90, panel.y + 112), "center", alpha=alpha)
    draw_text(surf, f"BEST  {high_score}", font, (255, 214, 90),
              (WIDTH // 2 + 90, panel.y + 112), "center", alpha=alpha)

    if new_best:
        scale = 1 + 0.06 * math.sin(frame_count * 0.2)
        badge = font_small.render("NEW BEST!", True, (255, 214, 90))
        badge = pygame.transform.rotozoom(badge, 0, scale)
        badge.set_alpha(alpha)
        surf.blit(badge, badge.get_rect(center=(WIDTH // 2, panel.y + 150)))

    if game_over_timer > 45:
        pulse = int(160 + 95 * math.sin(frame_count * 0.1))
        draw_text(surf, "Blink or press R to restart", font_small, (255, 255, 255),
                  (WIDTH // 2, panel.y + 185), "center", alpha=pulse)
        draw_text(surf, "Q to quit", font_tiny, (190, 195, 215),
                  (WIDTH // 2, panel.y + 210), "center", shadow=False)


def draw_paused(surf):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    surf.blit(overlay, (0, 0))
    draw_text(surf, "PAUSED", big_font, (255, 255, 255), (WIDTH // 2, HEIGHT // 2 - 60), "center")
    draw_text(surf, "Press P to resume", font_small, (220, 220, 235),
              (WIDTH // 2, HEIGHT // 2), "center")


# =====================================================
# 10. GAME STATE
# =====================================================

state = STATE_MENU

score = 0
high_score = 0
new_best = False
game_speed = 6

world_x = 0.0
night_t = 0.0
flash_timer = 0
shake = 0.0
game_over_timer = 0
frame_count = 0


def reset_game():
    global player_velocity, score, game_speed, flash_timer, new_best, on_ground, squash

    player.y = GROUND_Y - PLAYER_H
    player_velocity = 0
    on_ground = True
    squash = 0.0

    obstacles.clear()
    particles.clear()

    score = 0
    game_speed = 6
    flash_timer = 0
    new_best = False


def crash():
    global state, game_over_timer, shake, high_score, new_best

    state = STATE_GAME_OVER
    game_over_timer = 0
    shake = 14

    spawn_explosion(player.centerx, player.centery)
    play_sound("crash")

    if score // 10 > high_score:
        high_score = score // 10
        new_best = True


def update_playing():
    global player_velocity, on_ground, squash, score, game_speed, flash_timer, run_anim

    # -------- Player movement --------
    player_velocity += gravity
    player.y += int(player_velocity)

    # Ground collision
    if player.bottom >= GROUND_Y:
        player.bottom = GROUND_Y
        player_velocity = 0

        if not on_ground:          # just landed
            on_ground = True
            squash = 0.25
            spawn_dust(player.centerx, GROUND_Y, 6)

    run_anim += 1

    # -------- Obstacle generation --------
    if not obstacles or obstacles[-1]["trunks"][-1]["rect"].right < WIDTH - next_gap:
        spawn_obstacle()

    # Move obstacles
    for ob in obstacles:
        for t in ob["trunks"]:
            t["rect"].x -= game_speed

    # Remove obstacles outside screen
    obstacles[:] = [ob for ob in obstacles if ob["trunks"][-1]["rect"].right > -20]

    # -------- Collision detection (slightly forgiving hitboxes) --------
    hitbox = player.inflate(-12, -8)

    for ob in obstacles:
        for t in ob["trunks"]:
            shrink = (-8, -6) if ob["kind"] == "rock" else (-6, -4)
            if hitbox.colliderect(t["rect"].inflate(*shrink)):
                crash()
                return

    # -------- Score --------
    score += 1

    if score % 500 == 0 and game_speed < 15:
        game_speed += 1

    if score % 1000 == 0:          # every 100 displayed points
        flash_timer = 60
        play_sound("point")


def update_ambient(speed):
    global world_x, night_t, squash, shake, flash_timer, display_ear

    world_x += speed

    for c in clouds:
        c[0] -= speed * 0.15 + 0.2
        if c[0] < -110:
            c[0] = WIDTH + random.randint(0, 200)
            c[1] = random.randint(40, 190)
            c[2] = random.uniform(0.6, 1.2)

    for p in pebbles:
        p[0] -= speed
        if p[0] < -15:
            p[0] = WIDTH + random.randint(0, 40)
            p[1] = random.randint(GROUND_Y + 6, HEIGHT - 6)

    # Day/night cycle: night during the last 40% of every 700 points
    phase = ((score // 10) % 700) / 700
    target = 1.0 if (phase > 0.6 and state != STATE_MENU) else 0.0
    night_t += (target - night_t) * 0.02

    squash *= 0.8
    if state == STATE_MENU:
        squash = 0.03 * math.sin(frame_count * 0.08)   # idle breathing

    shake *= 0.85
    if flash_timer > 0:
        flash_timer -= 1

    display_ear += (current_ear - display_ear) * 0.35

    update_particles()


# =====================================================
# 11. GAME LOOP
# =====================================================

running = True

while running:

    clock.tick(FPS)
    frame_count += 1

    key_jump = False

    # Handle keyboard events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

            elif event.key == pygame.K_SPACE:
                key_jump = True

            elif event.key == pygame.K_r and state == STATE_GAME_OVER:
                reset_game()
                state = STATE_PLAYING
                play_sound("start")

            elif event.key == pygame.K_p and state in (STATE_PLAYING, STATE_PAUSED):
                state = STATE_PAUSED if state == STATE_PLAYING else STATE_PLAYING

            elif event.key == pygame.K_m:
                muted = not muted

            elif event.key == pygame.K_UP:
                ear_threshold = min(0.35, round(ear_threshold + 0.01, 2))

            elif event.key == pygame.K_DOWN:
                ear_threshold = max(0.10, round(ear_threshold - 0.01, 2))

    # -------- Blink control (camera is always read so the preview stays live) --------
    blink = detect_blink()
    action = blink or key_jump

    if state == STATE_MENU:
        if action:
            reset_game()
            state = STATE_PLAYING
            play_sound("start")
        update_ambient(2)

    elif state == STATE_PLAYING:
        if action:
            try_jump()
        update_playing()
        update_ambient(game_speed if state == STATE_PLAYING else 0)

    elif state == STATE_GAME_OVER:
        game_over_timer += 1
        if action and game_over_timer > 45:
            reset_game()
            state = STATE_PLAYING
            play_sound("start")
        update_ambient(0)

    # -------- Draw world --------
    draw_background(canvas)

    for ob in obstacles:
        draw_obstacle(canvas, ob)

    if state != STATE_GAME_OVER or game_over_timer > 50 or (game_over_timer // 6) % 2 == 0:
        draw_player(canvas)

    draw_particles(canvas)

    # Screen shake
    if shake > 0.5:
        screen.fill((0, 0, 0))
        offset = (random.randint(-int(shake), int(shake)), random.randint(-int(shake), int(shake)))
    else:
        offset = (0, 0)

    screen.blit(canvas, offset)

    # -------- HUD & overlays --------
    draw_hud(screen)

    if state == STATE_MENU:
        draw_menu(screen)
    elif state == STATE_PAUSED:
        draw_paused(screen)
    elif state == STATE_GAME_OVER:
        draw_game_over(screen)

    pygame.display.update()


# =====================================================
# 12. CLEANUP
# =====================================================

cap.release()
face_mesh.close()

cv2.destroyAllWindows()

pygame.quit()