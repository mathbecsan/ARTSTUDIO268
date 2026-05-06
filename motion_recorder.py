import cv2
import mediapipe as mp
import csv
import time
import numpy as np
import platform

# --- CONFIGURATION ---
OUTPUT_FILE = 'baked_motion.csv'
SMOOTHING = 0.6          
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# 13 Key MediaPipe Joints
JOINTS_MAP = {
    0: 'head', 
    11: 'left_shoulder', 12: 'right_shoulder',
    13: 'left_elbow',    14: 'right_elbow', 
    15: 'left_wrist',    16: 'right_wrist',
    23: 'left_hip',      24: 'right_hip', 
    25: 'left_knee',     26: 'right_knee',
    27: 'left_ankle',    28: 'right_ankle'
}

ALL_COLUMNS = [
    'head', 'neck', 'spine', 'pelvis',
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# --- SETUP MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# --- HELPER FUNCTIONS ---
def draw_ui_text(img, text, pos, color=(0, 255, 0), scale=1.0):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), int(scale * 4))
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, int(scale * 2))

def open_camera(index, use_dshow=True):
    """
    Tries to open camera. On Windows, DSHOW is often needed for OBS.
    """
    print(f"Attempting to open Camera {index}...")
    
    # 1. Try with DirectShow (Windows only, best for OBS)
    if platform.system() == 'Windows' and use_dshow:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
        
    # 2. If DSHOW failed or not used, try standard
    if not cap.isOpened() and platform.system() == 'Windows':
         print(f"   > DSHOW failed for {index}, trying default...")
         cap = cv2.VideoCapture(index)

    return cap

def select_camera_manual():
    """
    Manual cycling: Press C to increment index. No auto-skipping.
    """
    current_idx = 0
    fps_target = 30
    
    cap = open_camera(current_idx)
    
    # Try to set resolution immediately
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)

    print("--- CAMERA SELECTOR ---")
    print("[C] Next Camera | [F] Toggle FPS | [SPACE] Select")

    while True:
        ret, frame = cap.read()
        
        # If camera is broken/empty, create a "No Signal" screen
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            draw_ui_text(frame, "NO SIGNAL", (400, 300), (0, 0, 255), 2.0)
            draw_ui_text(frame, f"Checking Index: {current_idx}", (450, 400), (255, 255, 255), 1.0)
            draw_ui_text(frame, "Press 'C' to try next", (430, 500), (100, 100, 100), 0.8)

        # Draw UI
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        status_color = (0, 255, 0) if ret else (0, 0, 255)
        status_text = "ACTIVE" if ret else "EMPTY"
        
        draw_ui_text(frame, f"CAM {current_idx}: {status_text}", (30, 60), status_color, 1.2)
        draw_ui_text(frame, f"FPS REQ: {fps_target}", (30, 100), (200, 200, 0), 0.8)
        draw_ui_text(frame, "[C] Next Cam (+1)   [F] FPS   [SPACE] Start", (400, 80), (200, 200, 200), 0.7)

        cv2.imshow('Pro Motion Recorder', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            # Force close and increment
            cap.release()
            current_idx += 1
            if current_idx > 10: current_idx = 0 # Loop back after 10
            
            cap = open_camera(current_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, fps_target)

        elif key == ord('f') or key == ord('F'):
            fps_target = 60 if fps_target == 30 else 30
            cap.set(cv2.CAP_PROP_FPS, fps_target)
            print(f"Setting FPS to {fps_target}")

        elif key == ord(' '):
            if not ret:
                print("Cannot start: No valid camera signal!")
            else:
                print(f"Selected Camera {current_idx}")
                return cap
            
        elif key == ord('q'):
            cap.release()
            exit()

# --- MAIN APP ---

# 1. RUN SELECTOR
cap = select_camera_manual()

# 2. SETUP RECORDER
STATE_IDLE = 0
STATE_COUNTDOWN = 1
STATE_RECORDING = 2

state = STATE_IDLE
start_time = 0
countdown_start = 0
prev_landmarks = {} 

headers = ['timestamp']
for name in ALL_COLUMNS:
    headers.extend([f'{name}_x', f'{name}_y', f'{name}_z'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Process Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # UI Logic
    if state == STATE_IDLE:
        draw_ui_text(image, "READY", (50, 100), (255, 200, 0), 2)
        draw_ui_text(image, "Press SPACE to Record", (50, 160), (200, 200, 200), 1)
        
    elif state == STATE_COUNTDOWN:
        elapsed = time.time() - countdown_start
        left = 3 - int(elapsed)
        if left > 0:
            draw_ui_text(image, str(left), (image.shape[1]//2 - 50, image.shape[0]//2), (0, 255, 255), 5)
        else:
            state = STATE_RECORDING
            start_time = time.time()
            with open(OUTPUT_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
    
    elif state == STATE_RECORDING:
        cv2.circle(image, (50, 50), 20, (0, 0, 255), -1) 
        draw_ui_text(image, "REC", (80, 65), (0, 0, 255), 1)
        
        if results.pose_landmarks:
            current_time = time.time() - start_time
            current_pose = {} 
            
            for idx, name in JOINTS_MAP.items():
                lm = results.pose_landmarks.landmark[idx]
                vec = np.array([lm.x, lm.y, lm.z])
                if name in prev_landmarks:
                    vec = (SMOOTHING * vec) + ((1 - SMOOTHING) * prev_landmarks[name])
                current_pose[name] = vec
                prev_landmarks[name] = vec

            neck = (current_pose['left_shoulder'] + current_pose['right_shoulder']) / 2
            current_pose['neck'] = neck
            pelvis = (current_pose['left_hip'] + current_pose['right_hip']) / 2
            current_pose['pelvis'] = pelvis
            spine = (neck + pelvis) / 2
            current_pose['spine'] = spine

            row = [current_time]
            center_point = pelvis 
            for col_name in ALL_COLUMNS:
                vec = current_pose[col_name]
                norm_vec = vec - center_point
                row.extend(norm_vec.tolist())
                
            with open(OUTPUT_FILE, 'a', newline='') as f:
                csv.writer(f).writerow(row)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pro Motion Recorder', image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord(' ') and state == STATE_IDLE:
        state = STATE_COUNTDOWN
        countdown_start = time.time()
    elif key == ord(' ') and state == STATE_RECORDING:
        state = STATE_IDLE
        print(f"Stopped. Data saved to {OUTPUT_FILE}")

cap.release()
cv2.destroyAllWindows()