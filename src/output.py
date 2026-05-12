# output_writer.py
import json
import csv

def save_results(shots, fps):
    """
    shots: list of dicts like:
    {
        "frame": 120,
        "timestamp": 4.0,
        "shot_type": "FOREHAND",
        "player_box": [x1, y1, x2, y2]
    }
    """

    # ── Save JSON ──────────────────────────────────
    with open("shots.json", "w") as f:
        json.dump(shots, f, indent=4)
    print(f"Saved shots.json — {len(shots)} shots")

    # ── Save CSV ───────────────────────────────────
    with open("shots.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "timestamp", "shot_type", "player_box", "ball_pos", "ball_frame", "player_id"])
        writer.writeheader()
        writer.writerows(shots)
    print(f"Saved shots.csv — {len(shots)} shots")

    # ── Print summary ──────────────────────────────
    forehand  = sum(1 for s in shots if s["shot_type"] == "FOREHAND")
    backhand  = sum(1 for s in shots if s["shot_type"] == "BACKHAND")
    smash     = sum(1 for s in shots if s["shot_type"] == "SMASH")

    print("\n── Shot Summary ──────────────")
    print(f"  FOREHAND  : {forehand}")
    print(f"  BACKHAND  : {backhand}")
    print(f"  SMASH     : {smash}")
    print(f"  TOTAL     : {len(shots)}")