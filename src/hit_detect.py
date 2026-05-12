# hit_detector.py
import numpy as np

PROXIMITY_THRESHOLD = 200   
SPEED_THRESHOLD = 12       
MIN_FRAMES_BETWEEN_HITS = 15 
MIN_PLAYER_DIST_DELTA = 40 


SINGLE_PLAYER_MAX_DIST = 80  
PEAK_FACTOR = 1.3            
DOT_FACTOR = 0.2           

last_hit_frame = -999


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_player_center(box):
    x1, y1, x2, y2 = box
    
    h = y2 - y1
    contact_y = y1 + int(h * 0.35)
    return ((x1 + x2) // 2, contact_y)


def get_ball_velocity(ball_history, window=3):

    n = min(window, len(ball_history))
    if n < 2:
        return 0.0, 0.0, 0.0
    p_start = ball_history[-n]
    p_last = ball_history[-1]
    frames = n - 1
    dx = (p_last[0] - p_start[0]) / frames
    dy = (p_last[1] - p_start[1]) / frames
    speed = np.sqrt(dx * dx + dy * dy)
    return speed, dx, dy


def get_speed_window(ball_history, end_idx, window=3):

    if end_idx < 1:
        return 0.0
    n = min(window, end_idx + 1)
    start_idx = end_idx - n + 1
    p_start = ball_history[start_idx]
    p_end = ball_history[end_idx]
    frames = n - 1
    if frames <= 0:
        return 0.0
    dx = (p_end[0] - p_start[0]) / frames
    dy = (p_end[1] - p_start[1]) / frames
    return np.sqrt(dx * dx + dy * dy)


def detect_hit(ball_history, players, frame_num):
    global last_hit_frame

    if len(ball_history) < 2 or not players:
        return False, None

   
    if frame_num - last_hit_frame < MIN_FRAMES_BETWEEN_HITS:
        return False, None

    ball_pos = ball_history[-1]   # latest ball position
    speed, vx, vy = get_ball_velocity(ball_history)
    prev_idx = len(ball_history) - 2
    prev_speed = get_speed_window(ball_history, prev_idx, window=3) if prev_idx >= 0 else 0.0

    
    dists = []
    for i, box in enumerate(players):
        player_center = get_player_center(box)
        d = get_distance(ball_pos, player_center)
        dists.append((d, i))

    dists.sort()
    min_dist, min_idx = dists[0]
    if len(dists) > 1:
        second_dist = dists[1][0]
    else:
        second_dist = float('inf')

   
    player_center = get_player_center(players[min_idx])
    rx = ball_pos[0] - player_center[0]
    ry = ball_pos[1] - player_center[1]
    r_norm = np.sqrt(rx * rx + ry * ry) if (rx or ry) else 0.0
    dot = vx * rx + vy * ry

    # Debug info
    print(f"Frame {frame_num} | speed={speed:.1f} vx={vx:.1f} vy={vy:.1f} prev_speed={prev_speed:.1f} peak={(speed>prev_speed*PEAK_FACTOR)} dot={dot:.1f} r_norm={r_norm:.1f} min_dist={min_dist:.1f} second_dist={second_dist:.1f} -> idx={min_idx}")


    if len(players) == 1:
        if not (min_dist < SINGLE_PLAYER_MAX_DIST and speed > SPEED_THRESHOLD):
            return False, None

    
    peak_ok = speed > max(SPEED_THRESHOLD, prev_speed * PEAK_FACTOR)
    motion_ok = (r_norm > 0 and dot > DOT_FACTOR * speed * r_norm)

    proximity_ok = min_dist < PROXIMITY_THRESHOLD and speed > SPEED_THRESHOLD

    unambiguous_ok = (second_dist - min_dist) > MIN_PLAYER_DIST_DELTA

    if proximity_ok and (unambiguous_ok or len(players) == 1 or peak_ok or motion_ok):
        last_hit_frame = frame_num
        return True, min_idx

    return False, None