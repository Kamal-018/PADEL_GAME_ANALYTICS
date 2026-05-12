# shot_classifier.py

SMASH_SPEED_THRESHOLD = 25  
SMASH_Y_THRESHOLD = 400    
SMASH_VY_THRESHOLD = -8     


def _get_ball_velocity(ball_history, window=4):
    n = min(window, len(ball_history))
    if n < 2:
        return 0.0, 0.0, 0.0
    p_start = ball_history[-n]
    p_last = ball_history[-1]
    frames = n - 1
    vx = (p_last[0] - p_start[0]) / frames
    vy = (p_last[1] - p_start[1]) / frames
    speed = (vx * vx + vy * vy) ** 0.5
    return speed, vx, vy


def classify_shot(ball_history, hitter_box):
    x1, y1, x2, y2 = hitter_box
    player_center_x = (x1 + x2) // 2
    ball_x = ball_history[-1][0]
    ball_y = ball_history[-1][1]
    speed, vx, vy = _get_ball_velocity(ball_history)

    if ball_y < SMASH_Y_THRESHOLD and vy < SMASH_VY_THRESHOLD and speed > SMASH_SPEED_THRESHOLD:
        return "SMASH"

    # Forehand vs Backhand — which side of player is ball on
    if ball_x > player_center_x:
        return "FOREHAND"
    else:
        return "BACKHAND"