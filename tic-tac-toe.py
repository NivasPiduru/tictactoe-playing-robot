#!/usr/bin/env python3

import cv2, time, numpy as np, os
from collections import deque, defaultdict

# ================== TOGGLE ROBOT ==================
USE_ROBOT = True

if USE_ROBOT:
    try:
        from pydobot.dobot import Dobot, MODE_PTP  # <- public API (lowercase module)
    except Exception as e:
        raise RuntimeError(f"Robot mode requested but Dobot libs not available: {e}")
else:
    class Dobot:
        def __init__(self, *a, **kw): pass
        def suck(self, on): pass
        def move_to(self, *a, **kw): pass
        def speed(self, *a, **kw): pass
        def home(self): pass
        def close(self): pass
    class MODE_PTP:
        MOVJ_XYZ = 0
        MOVJ_ANGLE = 1

# =============== CAMERA/BOARD CONFIG ===============
CAM_INDEX   = 2
WARP_SIZE   = 600
CROP_MARGIN = 0.12
OPEN_K, CLOSE_K = 3, 3

# Lower-latency capture tuning
FRAME_W = 640
FRAME_H = 360
TARGET_FPS = 30
USE_MJPEG = True
PROCESS_EVERY_N_FRAMES = 2
FLUSH_BEFORE_RETRIEVE   = 3

# -------- Color detection (BLUE + RED) ----------
# BLUE (HSV) — tune if needed
HSV_BLUE = (np.array([95, 120, 70]),  np.array([130, 255, 255]))
# RED wraps hue; use two ranges and OR them
HSV_RED1 = (np.array([0, 120, 70]),   np.array([10, 255, 255]))
HSV_RED2 = (np.array([170, 120, 70]), np.array([180, 255, 255]))

HIST_LEN    = 5
COLOR_MIN_AREA_FRAC = 0.12

# --- Quad detection controls ---
USE_AUTO_QUAD = False
STATIC_QUAD   = None
CANNY_QUAD    = (90, 180)

# --- Detection cadence & motion settle times ---
DETECTION_INTERVAL_SEC   = 0.50
ROBOT_PLACE_SETTLE_SEC   = 0.3
HUMAN_PLACE_SETTLE_SEC   = 1.0

# --- Auto human capture window (no keypress) ---
HUMAN_CAPTURE_WINDOW_SEC = 2.0
CAP_MIN_HITS = 2
PREARM_MIN_CONSEC = 2

# Outcome logging
OUTCOME_FILE = "game_result.txt"
last_status_msg = ""

# =============== ROBOT CONFIG (YOUR COORDS) ===============
# Default: robot plays RED unless human takes RED first.
PICKUP_REDS = [
    (207.290, -238.231, -46.774, 0.0),
    (207.074, -214.722, -48.999, 0.0),
    (208.906, -196.078, -48.898, 0.0),
    (210.088, -173.647, -50.333, 0.0),
    (227.367, -235.950, -46.727, 0.0),
]
# If you have a separate BLUE stock tray, put those coords here.
# For now, reuse reds so code runs even without blue tray defined.
PICKUP_BLUES = PICKUP_REDS

# Mapping:
# (0,0)=grid1 ... (2,2)=grid9
CELL_POSE = {
    (0,0): (346.010,  39.404, -39.576, 0.0),  # grid 1
    (0,1): (338.995, -12.871, -34.897, 0.0),  # grid 2
    (0,2): (337.557, -73.005, -32.579, 0.0),  # grid 3
    (1,0): (284.937,  46.998, -37.192, 0.0),  # grid 4
    (1,1): (282.810, -17.880, -35.978, 0.0),  # grid 5
    (1,2): (280.547, -72.094, -32.325, 0.0),  # grid 6
    (2,0): (237.860,  51.530, -37.425, 0.0),  # grid 7
    (2,1): (237.448, -17.842, -32.285, 0.0),  # grid 8
    (2,2): (236.316, -76.721, -28.717, 0.0),  # grid 9
}

SAFE_Z   =  60.0
PICK_Z   = -45.0
PLACE_Z  = -45.0
HOVER_DWELL = 0.05

# Retreat/inspection pose after placing (your camera best view)
ROBOT_RETREAT_POSE = (234.759, -8.859, 60.909, 0.0)
RETREAT_WAIT_SEC   = 1.0
UPDATE_X_AFTER_RETREAT = True

# Optional motion speed
ROBOT_SPEED_L = 50
ROBOT_SPEED_J = 50

# =============== DOBOT CONNECTION / HELPERS ===============
if USE_ROBOT:
    device = Dobot(port="/dev/ttyACM0")
    try: device.speed(ROBOT_SPEED_L, ROBOT_SPEED_J)
    except Exception: pass
    try:
        device.home()
        time.sleep(1.5)
    except Exception: pass
    try:
        x,y,z,r = ROBOT_RETREAT_POSE
        device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x, y=y, z=z, r=r)
        time.sleep(0.1)
    except Exception: pass
else:
    device = Dobot()  # stub

def go_xyz(x, y, z, r=0.0):
    if USE_ROBOT:
        device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x, y=y, z=z, r=r)
        time.sleep(HOVER_DWELL)

def send_coordinates(x, y, z, r=0.0, wait=0.0):
    if USE_ROBOT:
        go_xyz(x, y, z, r)
        if wait > 0:
            time.sleep(wait)

def set_tool(on: bool):
    if not USE_ROBOT: return
    device.suck(bool(on))
    time.sleep(0.1)

def pick_at(pose):
    if not USE_ROBOT: return
    x,y,_,r = pose
    send_coordinates(x, y, SAFE_Z, r, HOVER_DWELL)
    set_tool(False)
    send_coordinates(x, y, PICK_Z, r, HOVER_DWELL)
    set_tool(True)
    time.sleep(0.15)
    send_coordinates(x, y, SAFE_Z, r, HOVER_DWELL)

def place_at(pose):
    if not USE_ROBOT: return
    x,y,_,r = pose
    send_coordinates(x, y, SAFE_Z, r, HOVER_DWELL)
    send_coordinates(x, y, PLACE_Z, r, HOVER_DWELL)
    set_tool(False)
    time.sleep(0.15)
    send_coordinates(x, y, SAFE_Z, r, HOVER_DWELL)

def retreat_to(pose):
    if not USE_ROBOT: return
    x, y, z, r = pose
    send_coordinates(x, y, z, r, HOVER_DWELL)

# =============== VISION HELPERS ===============
def open_cam():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    if USE_MJPEG:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if not cap.isOpened(): raise RuntimeError("no camera")
    return cap

def order_pts(pts):
    pts = np.array(pts, np.float32)
    s = pts.sum(1); d = np.diff(pts,1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)

def find_board_quad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    e = cv2.Canny(g, *CANNY_QUAD)
    cnts,_ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; bestA=0
    for c in cnts:
        peri=cv2.arcLength(c,True); a=cv2.approxPolyDP(c,0.02*peri,True)
        if len(a)==4:
            area=cv2.contourArea(a)
            if area>bestA: bestA=area; best=a.reshape(-1,2)
    return best

def warp_with_quad(frame, quad):
    dst = np.array([[0,0],[WARP_SIZE-1,0],[WARP_SIZE-1,WARP_SIZE-1],[0,WARP_SIZE-1]], np.float32)
    M = cv2.getPerspectiveTransform(order_pts(quad), dst)
    return cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

def warp_no_edge(frame, static_quad=None):
    h0, w0 = frame.shape[:2]
    if static_quad is None:
        src = np.array([[0,0],[w0-1,0],[w0-1,h0-1],[0,h0-1]], np.float32)
    else:
        src = static_quad.astype(np.float32)
    dst = np.array([[0,0],[WARP_SIZE-1,0],[WARP_SIZE-1,WARP_SIZE-1],[0,WARP_SIZE-1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

def mask_open_close(mask):
    if OPEN_K:  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((OPEN_K,OPEN_K), np.uint8))
    if CLOSE_K: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE_K,CLOSE_K), np.uint8))
    return mask

def present_by_area(mask, area_thr):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= area_thr:
            return True
    return False

def detect_color_present(cell_bgr, color: str):
    """Return True if 'color' (blue|red) occupies enough area in the crop."""
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    area_thr = COLOR_MIN_AREA_FRAC * (h * w)
    if color.lower() == "blue":
        mask = cv2.inRange(hsv, *HSV_BLUE)
        mask = mask_open_close(mask)
        return present_by_area(mask, area_thr)
    else:  # red
        m1 = cv2.inRange(hsv, *HSV_RED1)
        m2 = cv2.inRange(hsv, *HSV_RED2)
        mask = cv2.bitwise_or(m1, m2)
        mask = mask_open_close(mask)
        return present_by_area(mask, area_thr)

# =============== GAME LOGIC ===============
def lines_of(board):
    L=[]
    for i in range(3):
        L.append(board[i])
        L.append([board[0][i],board[1][i],board[2][i]])
    L.append([board[0][0],board[1][1],board[2][2]])
    L.append([board[0][2],board[1][1],board[2][0]])
    return L

def winner(board):
    for L in lines_of(board):
        if L[0] and L[0]==L[1]==L[2]:
            return L[0]
    if all(board[r][c] for r in range(3) for c in range(3)):
        return "TIE"
    return ""

def minimax(board, turn):
    w = winner(board)
    if w=="X": return 1, None
    if w=="O": return -1, None
    if w=="TIE": return 0, None
    best = (-2, None) if turn=="X" else (2, None)
    for r in range(3):
        for c in range(3):
            if board[r][c]!="": continue
            board[r][c]=turn
            score,_ = minimax(board, "O" if turn=="X" else "X")
            board[r][c]=""
            if turn=="X":
                if score>best[0]: best=(score,(r,c))
            else:
                if score < best[0]: best = (score, (r, c))
    return best

def find_strike(board):
    for i in range(3):
        if board[i][0] and board[i][0]==board[i][1]==board[i][2]:
            return (i,0),(i,2)
        if board[0][i] and board[0][i]==board[1][i]==board[2][i]:
            return (0,i),(2,i)
    if board[0][0] and board[0][0]==board[1][1]==board[2][2]:
        return (0,0),(2,2)
    if board[0][2] and board[0][2]==board[1][1]==board[2][0]:
        return (0,2),(2,0)
    return None

def draw_overlay(vis, board, strike=None):
    s=WARP_SIZE//3; pad=int(0.15*s)
    for k in (1,2):
        cv2.line(vis,(s*k,0),(s*k,WARP_SIZE),(255,255,255),2)
        cv2.line(vis,(0,s*k),(WARP_SIZE,s*k),(255,255,255),2)
    for r in range(3):
        for c in range(3):
            x0,y0=c*s,r*s; cx,cy=x0+s//2,y0+s//2
            v=board[r][c]
            if v=="X":
                cv2.line(vis,(x0+pad,y0+pad),(x0+s-pad,y0+s-pad),(0,255,0),3)
                cv2.line(vis,(x0+s-pad,y0+pad),(x0+pad,y0+s-pad),(0,255,0),3)
            elif v=="O":
                cv2.circle(vis,(cx,cy),int(s/2-pad),(0,0,255),3)
    if strike:
        (r0,c0),(r1,c1)=strike
        p0=(c0*s+s//2, r0*s+s//2); p1=(c1*s+s//2, r1*s+s//2)
        cv2.line(vis,p0,p1,(0,255,255),6)

def print_matrix(board, last=[None]):
    view = []
    for r in range(3):
        row = []
        for c in range(3):
            v = board[r][c]
            row.append("_" if v=="" else ("X" if v=="X" else "o"))
        view.append(row)
    if view == last[0]:
        return
    last[0] = [row[:] for row in view]
    for r in view:
        print(" ".join(r))
    print("-")

def write_outcome(msg):
    global last_status_msg
    last_status_msg = msg
    try:
        with open(OUTCOME_FILE, "w") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    print(msg)

def draw_status_bar(img, text):
    if not text: return
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (img.shape[1], 34), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# =============== MAIN ===============
def main():
    global last_status_msg
    try:
        start = input("Who goes first? [h]=human / [r]=robot: ").strip().lower()
    except Exception:
        start = "h"
    human_first = (start != "r")

    # ---- Dynamic color assignment ----
    HUMAN_COLOR = None                 # will become "blue" or "red" after first human move is captured
    ROBOT_COLOR = "red"               # default until HUMAN_COLOR is known (your requirement)
    color_to_stock = {"red": PICKUP_REDS, "blue": PICKUP_BLUES}
    stock_index = {"red": 0, "blue": 0}

    print("[mode] Robot:", "ENABLED" if USE_ROBOT else "DISABLED (dry-run)")
    print("[quad] Auto:", USE_AUTO_QUAD, "| Static quad provided:", STATIC_QUAD is not None)
    print(f"[detect] Interval: {DETECTION_INTERVAL_SEC:.2f}s  | pauses: robot={ROBOT_PLACE_SETTLE_SEC}s, human={HUMAN_PLACE_SETTLE_SEC}s | capture window={HUMAN_CAPTURE_WINDOW_SEC}s")
    print(f"[camera] {FRAME_W}x{FRAME_H}, fps~{TARGET_FPS}, MJPG={USE_MJPEG}, process every {PROCESS_EVERY_N_FRAMES} frame(s)")
    print(f"[colors] Default -> Robot: {ROBOT_COLOR.upper()}  (will flip if human takes RED first)")

    try:
        if os.path.exists(OUTCOME_FILE): os.remove(OUTCOME_FILE)
    except Exception:
        pass

    cap = open_cam()
    cv2.namedWindow("preview")
    cv2.namedWindow("board (warped)")

    board = [["" for _ in range(3)] for _ in range(3)]
    # Keep separate short histories for blue/red to stabilize detection
    blue_hist = [[deque(maxlen=HIST_LEN) for _ in range(3)] for _ in range(3)]
    red_hist  = [[deque(maxlen=HIST_LEN) for _ in range(3)] for _ in range(3)]

    last_detect = 0.0
    detection_pause_until = 0.0
    frame_id = 0

    # Human capture-window state (AUTO)
    human_cap_active = False
    human_cap_start  = 0.0
    human_cap_counts = defaultdict(int)
    prearm_counts = defaultdict(int)
    prearm_ready  = False

    def end_game_and_exit(msg, vis, disp):
        write_outcome(msg)
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
        print_matrix(board)
        draw_status_bar(disp, last_status_msg)
        cv2.imshow("board (warped)", vis)
        cv2.imshow("preview", disp)
        cv2.waitKey(1500)
        return True

    def get_fresh_frame():
        for _ in range(FLUSH_BEFORE_RETRIEVE):
            if not cap.grab(): break
        ok, f = cap.retrieve()
        return ok, f

    while True:
        if not cap.grab(): break
        frame_id += 1
        if frame_id % PROCESS_EVERY_N_FRAMES != 0:
            ok, disp = cap.retrieve()
            if not ok: break
            draw_status_bar(disp, last_status_msg)
            cv2.imshow("preview", disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            continue

        ok, frame = get_fresh_frame()
        if not ok: break
        disp = frame.copy()

        if USE_AUTO_QUAD:
            quad = find_board_quad(frame)
            if quad is None:
                draw_status_bar(disp, last_status_msg)
                cv2.imshow("preview", disp)
                if (cv2.waitKey(1) & 0xFF)==ord('q'): break
                continue
            cv2.polylines(disp, [quad.astype(int)], True, (0,255,255), 2)
            w = warp_with_quad(frame, quad)
        else:
            w = warp_no_edge(frame, STATIC_QUAD)

        now = time.time()
        do_detect = (now - last_detect) >= DETECTION_INTERVAL_SEC and (now >= detection_pause_until)
        if do_detect:
            last_detect = now
            s = WARP_SIZE//3; dx=int(s*CROP_MARGIN); dy=int(s*CROP_MARGIN)

            # Per-cell detection for BLUE and RED; then decide O based on HUMAN_COLOR rule
            detectedBlue = [["" for _ in range(3)] for _ in range(3)]
            detectedRed  = [["" for _ in range(3)] for _ in range(3)]
            detectedO    = [["" for _ in range(3)] for _ in range(3)]

            for r in range(3):
                for c in range(3):
                    cell = w[r*s:(r+1)*s, c*s:(c+1)*s]
                    cell = cell[dy:s-dy, dx:s-dx]

                    okB = detect_color_present(cell, "blue")
                    okR = detect_color_present(cell, "red")

                    blue_hist[r][c].append("B" if okB else "")
                    red_hist[r][c].append("R" if okR else "")

                    strongB = (sum(1 for v in blue_hist[r][c] if v=="B") >= (HIST_LEN//2+1))
                    strongR = (sum(1 for v in red_hist[r][c]  if v=="R") >= (HIST_LEN//2+1))

                    detectedBlue[r][c] = "B" if strongB else ""
                    detectedRed[r][c]  = "R" if strongR else ""

                    # Only human O’s are detected from vision; rule depends on chosen HUMAN_COLOR
                    if HUMAN_COLOR is None:
                        # Before assignment, treat either color as a candidate O
                        if strongB or strongR:
                            detectedO[r][c] = "O"
                    else:
                        # After assignment, only the human’s color counts as O
                        if HUMAN_COLOR == "blue" and strongB:
                            detectedO[r][c] = "O"
                        if HUMAN_COLOR == "red" and strongR:
                            detectedO[r][c] = "O"

            current_new = {(r,c) for r in range(3) for c in range(3)
                           if board[r][c]=="" and detectedO[r][c]=="O"}

            moves = sum(board[r][c] != "" for r in range(3) for c in range(3))
            human_turn = (moves % 2 == (0 if human_first else 1))
            robot_turn = (moves % 2 == (1 if human_first else 0))

            if human_turn:
                if not human_cap_active:
                    if current_new:
                        seen_now = set(current_new)
                        for rc in list(prearm_counts.keys()):
                            if rc not in seen_now:
                                prearm_counts[rc] = 0
                        for rc in seen_now:
                            prearm_counts[rc] += 1
                        prearm_ready = any(cnt >= PREARM_MIN_CONSEC for cnt in prearm_counts.values())
                        if prearm_ready:
                            human_cap_active = True
                            human_cap_start = now
                            human_cap_counts.clear()
                            for rc in current_new: human_cap_counts[rc] += 1
                            last_status_msg = "Human placing... (capturing)"
                    else:
                        for rc in list(prearm_counts.keys()):
                            prearm_counts[rc] = 0
                else:
                    for rc in current_new:
                        human_cap_counts[rc] += 1
                    if now - human_cap_start >= HUMAN_CAPTURE_WINDOW_SEC:
                        stable = [rc for rc,cnt in human_cap_counts.items() if cnt >= CAP_MIN_HITS]
                        if len(stable) == 1:
                            rr,cc = stable[0]

                            # ---- Decide HUMAN_COLOR from what’s actually in the cell (first time only) ----
                            if HUMAN_COLOR is None:
                                # Use the stronger of BLUE/RED at (rr,cc) according to the rolling hist
                                b_votes = sum(1 for v in blue_hist[rr][cc] if v=="B")
                                r_votes = sum(1 for v in red_hist[rr][cc]  if v=="R")
                                HUMAN_COLOR = "blue" if b_votes >= r_votes else "red"
                                ROBOT_COLOR = "red" if HUMAN_COLOR == "blue" else "blue"
                                print(f"[colors] Human color: {HUMAN_COLOR.upper()} | Robot color: {ROBOT_COLOR.upper()}")
                                last_status_msg = f"Human {HUMAN_COLOR.upper()} / Robot {ROBOT_COLOR.upper()}"

                            # Record human O move
                            board[rr][cc] = "O"
                            detection_pause_until = max(detection_pause_until, time.time() + HUMAN_PLACE_SETTLE_SEC)
                            print(f"Human placed {HUMAN_COLOR.upper()} (O) at ({rr},{cc})")

                        elif len(stable) == 0:
                            last_status_msg = "No stable human move detected"
                        else:
                            vis = w.copy()
                            draw_overlay(vis, board)
                            if end_game_and_exit("ERROR: Double entry detected.", vis, disp): break
                        human_cap_active = False
                        human_cap_counts.clear()
                        prearm_counts.clear()
                        prearm_ready = False
            else:
                human_cap_active = False
                human_cap_counts.clear()
                prearm_counts.clear()
                prearm_ready = False

            if robot_turn:
                _, move = minimax([row[:] for row in board], "X")
                if move:
                    r,c = move
                    if board[r][c]=="":
                        if USE_ROBOT:
                            # Pick from appropriate stock list for the robot’s assigned color
                            stock_list = color_to_stock[ROBOT_COLOR]
                            idx = stock_index[ROBOT_COLOR]
                            if idx >= len(stock_list):
                                vis = w.copy()
                                draw_overlay(vis, board)
                                if end_game_and_exit(f"ERROR: Out of {ROBOT_COLOR.upper()} blocks.", vis, disp): break
                            pick_at(stock_list[idx]); stock_index[ROBOT_COLOR] = idx + 1
                            place_at(CELL_POSE[(r, c)])
                            retreat_to(ROBOT_RETREAT_POSE)
                            if UPDATE_X_AFTER_RETREAT:
                                time.sleep(RETREAT_WAIT_SEC)
                            board[r][c] = "X"
                        else:
                            if UPDATE_X_AFTER_RETREAT:
                                time.sleep(RETREAT_WAIT_SEC)
                            board[r][c] = "X"

                        detection_pause_until = max(detection_pause_until, time.time() + max(0.0, ROBOT_PLACE_SETTLE_SEC))
                        last_status_msg = f"Robot ({ROBOT_COLOR.upper()}) placed X at ({r},{c})"

            wnr = winner(board)
            strike = find_strike(board) if wnr in ("X","O") else None
            vis = w.copy()
            draw_overlay(vis, board, strike=strike)
            if wnr:
                msg = "Robot wins!" if wnr=="X" else ("Human wins!" if wnr=="O" else "Draw!")
                if end_game_and_exit(msg, vis, disp): break

            print_matrix(board)
            cv2.imshow("board (warped)", vis)

        footer = "[q]=quit  [r]=reset  [h]=toggle first  |  mode:" + ("robot" if USE_ROBOT else "dry-run")
        draw_status_bar(disp, last_status_msg or footer)
        cv2.imshow("preview", disp)

        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'): break
        if k==ord('h'):
            human_first = not human_first
            last_status_msg = "Human first" if human_first else "Robot first"
        if k==ord('r'):
            board = [["" for _ in range(3)] for _ in range(3)]
            blue_hist = [[deque(maxlen=HIST_LEN) for _ in range(3)] for _ in range(3)]
            red_hist  = [[deque(maxlen=HIST_LEN) for _ in range(3)] for _ in range(3)]
            detection_pause_until = 0.0
            # Reset colors to default on hard reset
            HUMAN_COLOR = None
            ROBOT_COLOR = "red"
            stock_index = {"red": 0, "blue": 0}
            last_status_msg = "Reset (Robot default color = RED)"
            human_cap_active = False
            human_cap_counts.clear()
            prearm_counts.clear()

    cap.release(); cv2.destroyAllWindows()
    try:
        if USE_ROBOT: device.close()
    except Exception:
        pass

if __name__=="__main__":
    main()
