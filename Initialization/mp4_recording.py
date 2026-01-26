import os
import cv2
import time
import datetime
import numpy as np
import mediapipe as mp
import math
from collections import deque
from insightface.app import FaceAnalysis
import json
import hashlib
import platform
import uuid
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

# ===================== 参数 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "action_record.mp4")
INTERACTION_KEY_PATH = os.path.join(BASE_DIR, "interaction_key.jfhzy")
TIME_JSON_PATH = os.path.join(BASE_DIR, "action_time.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "face_landmarker.task")
USER_NAME = "User_001"
HOLD_TIME = 0.6
REPEAT = 3
FPS = 20
ACTION_TIMEOUT = 15.0  # 秒

ACTIONS = [
    ("look_up", "Look Up"),
    ("look_down", "Look Down"),
    ("turn_left", "Turn Left"),
    ("turn_right", "Turn Right"),
    ("pout", "Pout Lips"),
    ("mouth_open", "Open Mouth"),
    ("brow_up", "Raise Eyebrows"),
    ("brow_frown", "Frown Brows"),
    ("eye_close", "Close Eyes"),
]

# ===================== MediaPipe =====================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
latest_landmarks = None

def mp_callback(result, output_image, timestamp_ms):
    global latest_landmarks
    latest_landmarks = result.face_landmarks[0] if result.face_landmarks else None

def Mediapipe_Auto_GPU(model_path, callback, num_faces=1):
    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=BaseOptions.Delegate.GPU
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=callback,
            num_faces=num_faces,
        )
        return FaceLandmarker.create_from_options(options)
    except:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=callback,
            num_faces=num_faces,
        )
        return FaceLandmarker.create_from_options(options)

landmarker = Mediapipe_Auto_GPU(MODEL_PATH, mp_callback)

# ===================== InsightFace =====================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)  # 间隔抽帧
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame.copy())
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

def generate_embeddings(frames):
    embeddings = []
    for frame in frames:
        faces = face_app.get(frame)
        if faces:
            embeddings.append(faces[0].embedding)
    return embeddings

def get_path_salt(file_path: str) -> bytes:
    real_path = os.path.realpath(file_path).lower()
    return hashlib.sha256(real_path.encode("utf-8")).digest()

def get_device_salt() -> bytes:
    info = f"{platform.system()}-{platform.machine()}-{uuid.getnode()}"
    return hashlib.sha256(info.encode("utf-8")).digest()

def get_time_salt(file_path):
    if not os.path.exists(file_path):
        return hashlib.sha256(b"0").digest()
    create_time = os.path.getctime(file_path)
    dt = datetime.datetime.fromtimestamp(create_time)
    minute_bytes = str(dt.minute).encode("utf-8")
    return hashlib.sha256(minute_bytes).digest()

def derive_env_key(file_path: str) -> bytes:
    material = (
        get_path_salt(file_path)
        + get_device_salt()
        + get_time_salt(file_path)
    )

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"assistive-face-anti-interference",
        backend=default_backend()
    )
    return hkdf.derive(material)

def save_encrypted_embedding(path: str, embedding: np.ndarray):
    import io, base64
    # 转 np.float32
    key = base64.urlsafe_b64encode(derive_env_key(path))
    fernet = Fernet(key)

    buffer = io.BytesIO()
    np.save(buffer, embedding, allow_pickle=False)
    raw = buffer.getvalue()

    encrypted = fernet.encrypt(raw)
    with open(path, "wb") as f:
        f.write(encrypted)

# ===================== 工具函数 =====================
roll_history = deque(maxlen=5)
yaw_history = deque(maxlen=5)

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def smooth_value(history_deque, new_value):
    history_deque.append(new_value)
    return sum(history_deque) / len(history_deque)

def calc_pair_angle(p_left, p_right):
    dx = p_right[0] - p_left[0]
    dy = p_right[1] - p_left[1]
    return math.degrees(math.atan2(dy, dx))

def compensate_roll(pts, roll_angle_deg, center_point):
    roll_rad = -math.radians(roll_angle_deg)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    cx, cy = center_point
    return [((x-cx)*cos_r-(y-cy)*sin_r+cx, (x-cx)*sin_r+(y-cy)*cos_r+cy) for x,y in pts]

def face_scale(pts):
    face_width = dist(pts[234], pts[454])
    face_height = dist(pts[10], pts[152])
    return face_width, face_height

def compute_head_pose(pts):
    left_eye, right_eye, nose = pts[33], pts[263], pts[1]
    left_ear, right_ear = pts[93], pts[323]
    left_face, right_face = pts[206], pts[426]

    roll = smooth_value(roll_history,
                        (calc_pair_angle(left_eye, right_eye) +
                         calc_pair_angle(left_ear, right_ear) +
                         calc_pair_angle(left_face, right_face)) / 3)
    left_eye_c, right_eye_c = compensate_roll([left_eye, right_eye], roll, nose)
    left_face_c, right_face_c = compensate_roll([left_face, right_face], roll, nose)

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    face_width = abs(right_face[0]-left_face[0]) + 1e-6
    yaw_ratio = smooth_value(yaw_history, (nose[0]-eye_center_x)/face_width)

    top_y = (left_eye_c[1] + right_eye_c[1]) / 2
    bottom_y = (left_face_c[1] + right_face_c[1]) / 2
    pitch_ratio = (nose[1] - top_y) / (bottom_y - top_y + 1e-6)

    return roll, yaw_ratio, (pitch_ratio - 0.70)

# ===================== 动作判断 =====================
def is_lips_pout(pts, face_width, face_height):
    mouth_width = dist(pts[78], pts[308])
    mouth_height = dist(pts[13], pts[14])
    return (mouth_width / face_width < 0.25) and (mouth_height / face_height < 0.03)

def is_mouth_open(pts, face_height):
    return dist(pts[13], pts[14])/face_height > 0.05

def is_brow_raised(pts, roll_angle):
    left_pts = compensate_roll([pts[i] for i in [67,69,66,65]], roll_angle, pts[1])
    right_pts = compensate_roll([pts[i] for i in [300,302,301,297]], roll_angle, pts[1])
    left_eye_top = compensate_roll([pts[159]], roll_angle, pts[1])[0]
    right_eye_top = compensate_roll([pts[386]], roll_angle, pts[1])[0]
    ratio = ((abs(left_eye_top[1]-left_pts[0][1])/(abs(left_pts[2][1]-left_pts[1][1])+1e-6)) +
             (abs(right_eye_top[1]-right_pts[0][1])/(abs(right_pts[2][1]-right_pts[1][1])+1e-6))) / 2
    return ratio > 2.6

def is_brow_frown(pts, roll_angle, face_height):
    l = compensate_roll([pts[i] for i in [66,65,222,28,159,67,69]], roll_angle, pts[1])
    r = compensate_roll([pts[i] for i in [296,295,443,258,386,300,302]], roll_angle, pts[1])
    left_ratio = abs(l[0][1]-l[2][1])/face_height
    right_ratio = abs(r[0][1]-r[2][1])/face_height
    return (left_ratio + right_ratio)/2 > 0.04

LEFT_EYE_IDX = {"up":159,"down":145,"left":33,"right":133}
RIGHT_EYE_IDX = {"up":386,"down":374,"left":362,"right":263}
def is_eye_closed(pts):
    l_ratio = dist(pts[LEFT_EYE_IDX["up"]], pts[LEFT_EYE_IDX["down"]])/dist(pts[LEFT_EYE_IDX["left"]], pts[LEFT_EYE_IDX["right"]])
    r_ratio = dist(pts[RIGHT_EYE_IDX["up"]], pts[RIGHT_EYE_IDX["down"]])/dist(pts[RIGHT_EYE_IDX["left"]], pts[RIGHT_EYE_IDX["right"]])
    return l_ratio<0.16 and r_ratio<0.16

# ===================== 摄像头 =====================
cap = cv2.VideoCapture(0)
w, h = 640, 480
cap.set(3, w)
cap.set(4, h)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (w,h))

# ===================== 状态机 =====================
action_idx = 0
repeat_count = 0
hold_start = None
frames_collected = {a[0]: [] for a in ACTIONS}
time_records = {a[0]: [] for a in ACTIONS}
action_start_time = {}

timestamp = 0
cancelled = False

#print("📷 开始动作录制，按 q 取消")

# ===================== 注册用户人脸 =====================
user_face_saved = False

while cap.isOpened() and action_idx < len(ACTIONS):
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    writer.write(frame)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarker.detect_async(mp_image, timestamp)
    timestamp += 1

    if latest_landmarks:
        pts = [(lm.x*w, lm.y*h) for lm in latest_landmarks]
        roll, yaw_ratio, pitch_ratio = compute_head_pose(pts)
        face_width, face_height = face_scale(pts)

        action_key, action_text = ACTIONS[action_idx]
        active = False

        if action_key=="look_up": active = pitch_ratio<-0.12
        elif action_key=="look_down": active = pitch_ratio>0.12
        elif action_key=="turn_left": active = yaw_ratio<-0.07
        elif action_key=="turn_right": active = yaw_ratio>0.07
        elif action_key=="pout": active = is_lips_pout(pts, face_width, face_height)
        elif action_key=="mouth_open": active = is_mouth_open(pts, face_height)
        elif action_key=="brow_up": active = is_brow_raised(pts, roll)
        elif action_key=="brow_frown": active = is_brow_frown(pts, roll, face_height)
        elif action_key=="eye_close": active = is_eye_closed(pts)

        now = time.time()
        # ===================== 非头部动作超时处理 =====================
        if action_key in ["pout", "mouth_open", "brow_up", "brow_frown", "eye_close"]:
            if action_key not in action_start_time or action_start_time[action_key] is None:
                action_start_time[action_key] = now  # 动作开始计时

            elif now - action_start_time[action_key] > 15.0:  # 超时 15 秒
                #print(f"⚠️ {action_text} 超时 15 秒，跳过动作")
                time_records[action_key] = []  # 空列表表示未完成
                action_idx += 1
                repeat_count = 0
                hold_start = None
                continue  # 跳过当前帧逻辑
    
        if active:
            if hold_start is None: hold_start = now
            elif now-hold_start>=HOLD_TIME:
                frames_collected[action_key].append(frame.copy())
                repeat_count += 1
                if action_key not in action_start_time: action_start_time[action_key]=hold_start
                time_records[action_key].append(now)
                hold_start = None
                #print(f"✔ {action_text} {repeat_count}/{REPEAT}")
        else:
            hold_start = None

        if repeat_count>=REPEAT:
            action_idx +=1
            repeat_count=0
            hold_start=None

    # ===================== UI =====================
    cv2.putText(frame,f"User: {USER_NAME}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,200,0),2)
    if action_idx<len(ACTIONS):
        cv2.putText(frame,f"Action: {ACTIONS[action_idx][1]}",(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    cv2.putText(frame,f"Progress: {repeat_count}/{REPEAT}",(20,115),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,200,255),2)

    cv2.imshow("Action Recorder", frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        cancelled=True
        break

# ===================== 清理 =====================
cap.release()
writer.release()
cv2.destroyAllWindows()
landmarker.close()

if cancelled:
    if os.path.exists(VIDEO_PATH): os.remove(VIDEO_PATH)
    #print("❌ 手动取消，未生成文件")
    exit()

# ===================== 时间 JSON =====================
json_records = {}
for k,times in time_records.items():
    if len(times)>=REPEAT:
        json_records[k] = {"start": action_start_time[k], "end": times[REPEAT-1]}
    else:
        json_records[k] = {"start": action_start_time.get(k,None), "end": times[-1] if times else None}

with open(TIME_JSON_PATH,"w") as f:
    json.dump(json_records,f,indent=4)

#print("✅ 动作录制完成")
#print("📁 文件列表：")
#print(" - 视频:", VIDEO_PATH)
frames = extract_frames(VIDEO_PATH, max_frames=10)
embeddings_list = generate_embeddings(frames)
if embeddings_list:
    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)
    save_encrypted_embedding(INTERACTION_KEY_PATH, embeddings)
    #print(" - 初始化成功", INTERACTION_KEY_PATH)
#else:
    #print("⚠️ 初始化失败")
#print(" - 时间 JSON:", TIME_JSON_PATH)
