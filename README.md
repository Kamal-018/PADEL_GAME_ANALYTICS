Padel Shot Tracker
===================

Compact pipeline to detect the padel ball, track two players, detect hits and classify shots (forehand / backhand / smash).

Key points

## Ball detection

-  It was the hardest problem: very fast motion, motion blur and a tiny, often pixelated ball make per-frame detection noisy. A plain YOLO detector reliably finds players but not the ball in many frames.

- To solve this I used a Roboflow workflow: trained a YOLOv8s model on a padel-ball dataset (fine‑tuned weights saved as `best.pt`) and ran local inference + simple tracking to yield a stable red box on the ball. I tried the Roboflow Inference SDK / endpoint but ended up running local inference because calling the endpoint from the local pipeline was harder to integrate.

## Player Detection
- It  uses a standard YOLO model to find the two bottom‑half players; simple per-player trackers keep IDs stable across short occlusions.

##  Hit detection
- Ball position tracked each frame with speed + velocity (dx, dy)
- Hit triggered when:
  - Ball within 200px of a player center
  - Ball speed above threshold (12px/frame)
  - Cooldown of 15 frames between hits to avoid double counting
- Closest player to ball = assigned hitter
- Extra confidence checks:
  - Speed spike vs previous frames (peak detection)
  - Ball moving away from player (dot product check)

## Shot Classification Logic

- Runs only when a hit is confirmed
- Three shot types classified by pure geometry:
  - `SMASH` → ball Y position in upper court (y < 400) + speed above 60px/frame
  - `FOREHAND` → ball X is to the right of player center X
  - `BACKHAND` → ball X is to the left of player center X
- No model training required — top-down camera angle makes pose estimation unreliable, ball position relative to player center is sufficient

Run (example)
---------------
Create a venv, install deps and run the main script:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

Outputs
- Annotated video: `output/output_video.mp4`
- Shot records: `shots.json` and `shots.csv`


