
import cv2
import mediapipe as mp
import pygame
import random
import math

# -----------------------------
# 1. INITIALIZATION
# -----------------------------

pygame.init()

WIDTH = 900
HEIGHT = 500
GROUND_Y = 400

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blink to Jump - CV Game")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 30)
big_font = pygame.font.SysFont("Arial", 55)

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

EAR_THRESHOLD = 0.21
closed_frames = 0
blink_pending = False

# -----------------------------
# 2. BLINK DETECTION
# -----------------------------

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

    ear = (vertical_1 + vertical_2) / (2 * horizontal)

    return ear


def detect_blink():

    global closed_frames, blink_pending

    success, frame = cap.read()

    if not success:
        return False

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    blink_detected = False

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        left_ear = calculate_ear(
            face.landmark, LEFT_EYE, w, h
        )

        right_ear = calculate_ear(
            face.landmark, RIGHT_EYE, w, h
        )

        average_ear = (left_ear + right_ear) / 2

        # Check whether eyes are closed
        if average_ear < EAR_THRESHOLD:

            closed_frames += 1

            if closed_frames >= 2:
                blink_pending = True

        else:

            # Eyes reopened after closing
            if blink_pending:
                blink_detected = True

            closed_frames = 0
            blink_pending = False

        cv2.putText(
            frame,
            f"EAR: {average_ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "BLINK DETECTED!" if blink_detected else "Keep Playing",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Webcam - Blink Detection", frame)

    return blink_detected


# -----------------------------
# 3. PLAYER SETTINGS
# -----------------------------

player = pygame.Rect(100, GROUND_Y - 60, 45, 60)

player_velocity = 0
gravity = 0.8
jump_strength = -16

# -----------------------------
# 4. GAME SETTINGS
# -----------------------------

obstacles = []

game_over = False
score = 0
game_speed = 6

# -----------------------------
# 5. GAME LOOP
# -----------------------------

running = True

while running:

    clock.tick(60)

    # Handle keyboard events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_q:
                running = False

            if event.key == pygame.K_r and game_over:

                player.y = GROUND_Y - 60
                player_velocity = 0

                obstacles.clear()

                score = 0
                game_speed = 6

                game_over = False

    # -------------------------
    # 6. BLINK CONTROL
    # -------------------------

    if not game_over:

        blink = detect_blink()

        # Jump only when player is on ground
        if blink and player.bottom >= GROUND_Y:

            player_velocity = jump_strength

    # -------------------------
    # 7. PLAYER MOVEMENT
    # -------------------------

    if not game_over:

        player_velocity += gravity
        player.y += int(player_velocity)

        # Ground collision
        if player.bottom >= GROUND_Y:

            player.bottom = GROUND_Y
            player_velocity = 0

    # -------------------------
    # 8. OBSTACLE GENERATION
    # -------------------------

    if not game_over:

        if not obstacles or obstacles[-1].x < WIDTH - random.randint(300, 450):

            obstacle_height = random.randint(35, 65)

            obstacle = pygame.Rect(
                WIDTH,
                GROUND_Y - obstacle_height,
                25,
                obstacle_height
            )

            obstacles.append(obstacle)

        # Move obstacles
        for obstacle in obstacles:

            obstacle.x -= game_speed

        # Remove obstacles outside screen
        obstacles = [
            obstacle for obstacle in obstacles
            if obstacle.right > 0
        ]

    # -------------------------
    # 9. COLLISION DETECTION
    # -------------------------

    if not game_over:

        for obstacle in obstacles:

            if player.colliderect(obstacle):

                game_over = True

    # -------------------------
    # 10. SCORE
    # -------------------------

    if not game_over:

        score += 1

        if score % 500 == 0:
            game_speed += 1

    # -------------------------
    # 11. DRAW GAME
    # -------------------------

    screen.fill((240, 248, 255))

    # Ground
    pygame.draw.line(
        screen,
        (50, 50, 50),
        (0, GROUND_Y),
        (WIDTH, GROUND_Y),
        3
    )

    # Player
    pygame.draw.rect(
        screen,
        (0, 150, 80),
        player,
        border_radius=8
    )

    # Player eye
    pygame.draw.circle(
        screen,
        (255, 255, 255),
        (player.x + 32, player.y + 15),
        5
    )

    # Obstacles
    for obstacle in obstacles:

        pygame.draw.rect(
            screen,
            (200, 50, 50),
            obstacle,
            border_radius=4
        )

    # Score
    score_text = font.render(
        f"Score: {score // 10}",
        True,
        (0, 0, 0)
    )

    screen.blit(score_text, (WIDTH - 180, 30))

    # Game instructions
    instruction = font.render(
        "Blink to Jump!",
        True,
        (50, 50, 50)
    )

    screen.blit(instruction, (30, 30))

    # Game over screen
    if game_over:

        game_over_text = big_font.render(
            "GAME OVER",
            True,
            (200, 0, 0)
        )

        restart_text = font.render(
            "Press R to Restart | Q to Quit",
            True,
            (0, 0, 0)
        )

        screen.blit(
            game_over_text,
            (WIDTH // 2 - 170, HEIGHT // 2 - 60)
        )

        screen.blit(
            restart_text,
            (WIDTH // 2 - 190, HEIGHT // 2 + 10)
        )

    pygame.display.update()


# -----------------------------
# 12. CLEANUP
# -----------------------------

cap.release()
face_mesh.close()

cv2.destroyAllWindows()

pygame.quit()
