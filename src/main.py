# main.py
import cv2
from ball_tracker import detect_ball, COURT_PTS
from player_detect import detect_players
from hit_detect import detect_hit
from shot_classifier import classify_shot
from output import save_results

VIDEO_PATH = "data/sample.mp4"
SHOT_DISPLAY_DURATION = 45  
YOLO_EVERY_N_FRAMES = 6     

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("Padel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Padel", 1280, 720)

# State
ball_history = []  
players = []          
shots = []              

trackers = []
player_ids = []
next_player_id = 1

def make_tracker():
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        try:
            return cv2.TrackerKCF_create()
        except Exception:
            try:
                return cv2.legacy.TrackerCSRT_create()
            except Exception:
                return None

shot_label = ""
shot_player_box = None
shot_display_frames = 0

last_ball_frame = -999

frame_num = 0

writer = None
out_video_path = "output/output_video.mp4"

# Live shot counters
forehand_count = 0
backhand_count = 0
smash_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Ball detection 
    ball = detect_ball(frame)
    if ball:
        cx, cy, x, y, w, h = ball
        ball_history.append((cx, cy))
        last_ball_frame = frame_num
        if len(ball_history) > 10:
            ball_history.pop(0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:   
        if ball_history and (frame_num - last_ball_frame) > 5:
            ball_history.clear()
    

    # Player detection
    if frame_num % YOLO_EVERY_N_FRAMES == 0:
        new_players = detect_players(frame)

        if len(new_players) >= 2:
            new_players = sorted(new_players, key=lambda b: ((b[0] + b[2]) // 2))

            def centers(boxes):
                return [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes]

            def match_boxes(prev_boxes, new_boxes, max_dist=200):
                pcent = centers(prev_boxes)
                ncent = centers(new_boxes)
                used_p = set()
                mapping = {}  # new_idx -> prev_idx
                for new_i, nc in enumerate(ncent):
                    best = None
                    best_d = None
                    for prev_i, pc in enumerate(pcent):
                        if prev_i in used_p:
                            continue
                        d = (nc[0]-pc[0])**2 + (nc[1]-pc[1])**2
                        if best is None or d < best_d:
                            best = prev_i
                            best_d = d
                    if best is not None and best_d is not None and best_d <= max_dist*max_dist:
                        mapping[new_i] = best
                        used_p.add(best)
                return mapping

            mapping = {}
            if players and player_ids:
                mapping = match_boxes(players, new_players, max_dist=250)

            new_ids = []
            for i in range(len(new_players)):
                if i in mapping:
                    new_ids.append(player_ids[mapping[i]])
                else:
                    new_ids.append(next_player_id)
                    next_player_id += 1

            players = new_players
            player_ids = new_ids

            trackers = []
            for box in players:
                t = make_tracker()
                if t is None:
                    trackers = []
                    break
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                try:
                    t.init(frame, (x1, y1, w, h))
                except Exception:
                    trackers = []
                    break
                trackers.append(t)
        else:
            if len(players) < 2:
                players = new_players

    if trackers:
        tracked_boxes = []
        ok_all = True
        for t in trackers:
            ok, bb = t.update(frame)
            if not ok:
                ok_all = False
                break
            x, y, w, h = map(int, bb)
            tracked_boxes.append((x, y, x + w, y + h))
        if ok_all and len(tracked_boxes) >= 1:
            players = tracked_boxes
            # keep player_ids aligned with trackers (if counts differ, regenerate ids)
            if len(player_ids) != len(tracked_boxes):
                # reassign ids sequentially
                player_ids = list(range(1, len(tracked_boxes)+1))
                next_player_id = max(player_ids) + 1
        else:
            # lost tracking
            trackers = []

    for i, (x1, y1, x2, y2) in enumerate(players):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        pid = player_ids[i] if i < len(player_ids) else (i+1)
        cv2.putText(frame, f"P{pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # Hit detection
    hit, hitter_idx = detect_hit(ball_history, players, frame_num)

    if hit and hitter_idx is not None and hitter_idx < len(players):
        hitter_box = players[hitter_idx]
        shot = classify_shot(ball_history, hitter_box)
        timestamp = round(frame_num / fps, 2)

        # Store result
        shots.append({
            "frame": frame_num,
            "timestamp": timestamp,
            "shot_type": shot,
            "player_box": list(hitter_box),
            "ball_pos": list(ball_history[-1]),
            "ball_frame": last_ball_frame,
            "player_id": hitter_idx + 1
        })

        # Display
        shot_label = shot
        shot_player_box = hitter_box
        shot_display_frames = SHOT_DISPLAY_DURATION
        print(f"Frame {frame_num} | Player P{hitter_idx+1} | {shot} | t={timestamp}s")

        # update live counters
        s = shot.upper() if isinstance(shot, str) else str(shot)
        if "FOREHAND" in s:
            forehand_count += 1
        elif "BACKHAND" in s:
            backhand_count += 1
        elif "SMASH" in s:
            smash_count += 1

    # Show shot label
    if shot_display_frames > 0 and shot_player_box:
        x1, y1, x2, y2 = shot_player_box
        color = (0, 0, 255)   if shot_label == "SMASH"     else \
                (255, 0, 0)   if shot_label == "FOREHAND"  else \
                (0, 255, 255)
        cv2.putText(frame, shot_label, (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        shot_display_frames -= 1

    # Court boundary
    cv2.polylines(frame, [COURT_PTS], isClosed=True,
                  color=(0, 255, 255), thickness=2)

    # Live overlay counts 
    overlay_w, overlay_h = 260, 70
    overlay_tl = (10, 10)
    overlay_br = (overlay_tl[0] + overlay_w, overlay_tl[1] + overlay_h)
    cv2.rectangle(frame, overlay_tl, overlay_br, (50, 50, 50), -1)
    cv2.putText(frame, f"Forehand: {forehand_count}", (overlay_tl[0]+10, overlay_tl[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Backhand: {backhand_count}", (overlay_tl[0]+10, overlay_tl[1]+45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Smash: {smash_count}", (overlay_tl[0]+10, overlay_tl[1]+65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

   
    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            writer = cv2.VideoWriter(out_video_path, fourcc, fps if fps>0 else 30.0, (w, h))
        except Exception:
            writer = None

    
    if writer is not None:
        writer.write(frame)

    cv2.imshow("Padel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()

if writer is not None:
    writer.release()
    print(f"Saved output video: {out_video_path}")

# Save results
save_results(shots, fps)