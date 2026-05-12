# player_detector.py
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

BOTTOM_HALF_Y = 550  

def detect_players(frame):
    results = model(frame, verbose=False)
    players = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cy = (y1 + y2) // 2

            if cy > BOTTOM_HALF_Y:
                area = (x2 - x1) * (y2 - y1)
                players.append((x1, y1, x2, y2, area))

    
    players = sorted(players, key=lambda p: p[4], reverse=True)[:2]

   
    return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in players]