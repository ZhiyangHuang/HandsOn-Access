import os
import cv2
import numpy as np
import mediapipe as mp
import sys
import os
import math
from collections import deque
import win32api
import win32con
import threading
import time
from insightface.app import FaceAnalysis
import json

# ================== 脚本根目录 ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== Keyboard 模块路径 ==================
keyboard_path = os.path.join(BASE_DIR, "../Keyboard")
if keyboard_path not in sys.path:
    sys.path.append(keyboard_path)
from vosk_recording import VoiceKeyboardSystem
from whisper_recording import RealTimeDictation
from speech_copy_win32 import ContinuousDictationController

# ================== 配置 ==================
USER_SETTING_FILE = os.path.join(BASE_DIR, "..", "user_setting.json")
if not os.path.exists(USER_SETTING_FILE):
    raise FileNotFoundError("❌ 未检测到 user_setting.json，系统未初始化，程序退出。")

with open(USER_SETTING_FILE, "r", encoding="utf-8") as f:
    USER_SETTINGS = json.load(f)

MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "face_landmarker.task")
USER_FACE_PATH = os.path.join(BASE_DIR, "..", "Initialization", "USER_FACE.npy")
# ================== 本人校验状态机 ==================
STATE_NO_FACE = 0
STATE_VERIFYING = 1
STATE_LOCKED = 2
STATE_REJECT = 3

state = STATE_NO_FACE
SIM_THRESHOLD = 0.5

mouth_open_timer = None  # 记录嘴巴张开的时间
mouth_triggered = False  # 是否已经触发动作
MOUTH_DELAY = 3         # 张嘴倒计时，单位秒

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

face_present = False

mp_busy = False
latest_pts = None

# ---------------- 全局状态 ----------------
mouse_state = {
    "current": None,   # None / "left_click" / "right_click" / "left_down"
    "down_time": None,
    "timer": None,
    "move_state": False,      # Move 状态
    "move_timer": None        # Move 状态倒计时线程
}
mouse_enabled = True  # True 表示鼠标操作允许，False 表示禁止
mouse_state_lock = threading.Lock()  # 🔒
landmarks_lock = threading.Lock()    # 🔒
dictation_lock = threading.Lock()    # 🔒

# =============== json状态列表 ====================
try:
    with open(USER_SETTING_FILE, "r", encoding="utf-8") as f:
        user_settings = json.load(f)
except FileNotFoundError:
    user_settings = {}

def execute_action(is_active, function_name):
    global mouth_open_timer, mouth_triggered
    
    if function_name == "Left click" and is_active:
        if pitch_ratio > 0.07 or pitch_ratio < -0.07 or yaw_ratio > 0.07 or yaw_ratio < -0.07:
            return
        Right_Click()

    elif function_name == "Right click" and is_active:
        if pitch_ratio > 0.07 or pitch_ratio < -0.07 or yaw_ratio > 0.07 or yaw_ratio < -0.07:
            return
        Left_Click()

    elif function_name == "Drag":
        if is_active:
            Left_ClickDown()
        else:
            Left_ClickUp()

    elif function_name == "Keyboard":
        toggle_mouse(not is_active)
        current_time = time.time()
        if pitch_ratio > 0.07 or pitch_ratio < -0.07 or yaw_ratio > 0.07 or yaw_ratio < -0.07:
            with dictation_lock:  # 🔒
                if is_active and dictation_system.paused:
                    # 嘴巴张开，开始或重置倒计时
                    mouth_open_timer = current_time
                    mouth_triggered = False  # 重置触发状态
                    dictation_system.set_paused(False)  # 张嘴处理语音
                elif not is_active and not dictation_system.paused:
                    if mouth_open_timer is not None:
                        elapsed = current_time - mouth_open_timer
                        if elapsed >= MOUTH_DELAY and not mouth_triggered:
                            dictation_system.set_paused(True)   # 闭嘴暂停处理
                            mouth_triggered = True

# ---------------- 键盘总开关 ----------------
def vosk_test():
    return VoiceKeyboardSystem() # VOSK

def whisper_test(model):
    controller = ContinuousDictationController()
    dictation_system = RealTimeDictation(controller, model)
    return dictation_system# tiny / base / small / medium / large

def keyboard_controller():
    dictation_system = None
    if user_settings["voice_model"] == "VOSK":
        dictation_system = vosk_test()
    else:
        dictation_system = whisper_test(user_settings["voice_model"])
    return dictation_system
# 1. 创建控制器
dictation_system = keyboard_controller()
# 后台开启
dictation_system.start()

# ---------------- 鼠标总开关 ----------------
def toggle_mouse(using_Mouse):
    """
    切换鼠标操作开关
    """
    global mouse_enabled  # 声明我们要修改全局变量
    mouse_enabled = using_Mouse
    status = "开启" if mouse_enabled else "暂停"
    #print(f"鼠标操作已{status}")

# ---------------- Move 状态管理 ----------------
def start_move_timer():
    """
    启动 1 秒倒计时，如果 1 秒内没有刷新状态则取消 Move 状态
    """
    def timer_func():
        time.sleep(1)  # 等待 1 秒
        with mouse_state_lock:  # 🔒
            # 如果超过 1 秒没有刷新状态，取消 Move 状态
            if mouse_state["move_state"]:
                mouse_state["move_state"] = False
                mouse_state["move_timer"] = None
                #print("Move 状态已取消")

    # 取消已有倒计时线程（如果有）
    if mouse_state["move_timer"] is not None:
        # 这里简单不直接杀线程，下一行刷新状态即可
        mouse_state["move_timer"] = None

    # 启动新倒计时线程
    t = threading.Thread(target=timer_func)
    t.start()
    with mouse_state_lock:  # 🔒
        mouse_state["move_timer"] = t

def set_move_state():
    """
    设置 Move 状态并刷新倒计时
    """
    with mouse_state_lock:  # 🔒
        mouse_state["move_state"] = True
        #print("Move 状态已触发")
    start_move_timer()

# ---------------- 鼠标移动函数 ----------------
def Move_Up(pixels):
    if not mouse_enabled:  # 如果被禁用，直接返回
        return
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, -pixels, 0, 0)
    set_move_state()

def Move_Down(pixels):
    if not mouse_enabled:  # 如果被禁用，直接返回
        return
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, pixels, 0, 0)
    set_move_state()

def Move_Left(pixels):
    if not mouse_enabled:  # 如果被禁用，直接返回
        return
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -pixels, 0, 0, 0)
    set_move_state()

def Move_Right(pixels):
    if not mouse_enabled:  # 如果被禁用，直接返回
        return
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, pixels, 0, 0, 0)
    set_move_state()

def Left_Click():
    with mouse_state_lock:  # 🔒
        if not mouse_enabled:  # 如果被禁用，直接返回
            return
        if mouse_state["current"] is None and  not mouse_state["move_state"]:
            mouse_state["current"] = "left_click"
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            mouse_state["current"] = None  # 完成后释放状态

def Right_Click():
    with mouse_state_lock:  # 🔒
        if not mouse_enabled:  # 如果被禁用，直接返回
            return
        if mouse_state["current"] is None and  not mouse_state["move_state"]:
            mouse_state["current"] = "right_click"
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            mouse_state["current"] = None

# 自动释放函数
def auto_release():
    time.sleep(10)  # 等待10秒
    with mouse_state_lock:
        # 如果状态仍然是 left_down，自动释放
        if mouse_state["current"] == "left_down":
            #print("自动释放 Left_ClickUp（超过10秒）")
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            mouse_state["current"] = None
            mouse_state["down_time"] = None
            mouse_state["timer"] = None

# 左键按下
def Left_ClickDown():
    if not mouse_enabled:  # 如果被禁用，直接返回
        return
    with mouse_state_lock:
        if mouse_state["current"] is None:
            mouse_state["current"] = "left_down"
            mouse_state["down_time"] = time.time()
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            # 启动计时器线程
            t = threading.Thread(target=auto_release)
            t.start()
            mouse_state["timer"] = t
            #print("Left_ClickDown 已触发")

# 左键释放
def Left_ClickUp():
    with mouse_state_lock:  # 🔒
        if mouse_state["current"] == "left_down":
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            mouse_state["current"] = None
            mouse_state["down_time"] = None
            mouse_state["timer"] = None
            #print("Left_ClickUp 已手动释放")

# ================== MediaPipe 初始化 ==================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_landmarks = None
landmarks_lock = threading.Lock()

def mp_callback(result, output_image, timestamp_ms):
    global latest_landmarks, face_present, mp_busy
    if result.face_landmarks:
        with landmarks_lock:  # 🔒 只保护共享数据
            latest_landmarks = result.face_landmarks[0]
            face_present = True
    else:
        with landmarks_lock:
            latest_landmarks = None
            face_present = False

    # ✅ 无论成功还是失败，必须释放 busy
    mp_busy = False

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate="GPU"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=mp_callback,
    num_faces=1,
)

landmarker = FaceLandmarker.create_from_options(options)

# ================ 本人验证函数 ================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_identity(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    if len(faces) == 0:
        return False, 0.0
    feature = faces[0].normed_embedding
    face_template = np.load(USER_FACE_PATH, mmap_mode='r')
    sim = cosine_similarity(feature, face_template)
    return sim > SIM_THRESHOLD, sim

# ================== 工具函数 ==================
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
    """绕中心点做 roll 补正"""
    roll_rad = -math.radians(roll_angle_deg)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    cx, cy = center_point
    return [( (x-cx)*cos_r-(y-cy)*sin_r+cx, (x-cx)*sin_r+(y-cy)*cos_r+cy ) for x,y in pts]

def face_scale(pts):
    """脸部宽高"""
    face_width = dist(pts[234], pts[454])
    face_height = dist(pts[10], pts[152])
    return face_width, face_height

# ================== 头部姿态 ==================
def compute_head_pose(pts):
    left_eye, right_eye, nose = pts[33], pts[263], pts[1]
    left_ear, right_ear = pts[93], pts[323]
    left_face, right_face = pts[206], pts[426]

    # Roll
    roll = smooth_value(roll_history, 
                        (calc_pair_angle(left_eye, right_eye) +
                         calc_pair_angle(left_ear, right_ear) +
                         calc_pair_angle(left_face, right_face)) / 3)

    # ==== Roll 补正后的点 ====
    left_eye_c, right_eye_c = compensate_roll([left_eye, right_eye], roll, nose)
    left_face_c, right_face_c = compensate_roll([left_face, right_face], roll, nose)

    # Yaw
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    face_width = abs(right_face[0]-left_face[0]) + 1e-6
    yaw_ratio = smooth_value(yaw_history, (nose[0]-eye_center_x)/face_width)

    # Pitch with Roll补正
    top_y = (left_eye_c[1] + right_eye_c[1]) / 2
    bottom_y = (left_face_c[1] + right_face_c[1]) / 2
    pitch_ratio = (nose[1] - top_y) / (bottom_y - top_y + 1e-6)

    return roll, yaw_ratio, (pitch_ratio - 0.70)

# ================== 动作检测 ==================
def is_brow_raised(pts, roll_angle):
    nose = pts[1]
    left_pts = compensate_roll([pts[i] for i in [67,69,66,65]], roll_angle, nose)
    right_pts = compensate_roll([pts[i] for i in [300,302,301,297]], roll_angle, nose)
    left_eye_top = compensate_roll([pts[159]], roll_angle, nose)[0]
    right_eye_top = compensate_roll([pts[386]], roll_angle, nose)[0]

    left_ratio = (1/(abs(left_pts[2][1]-left_pts[1][1])+1e-6)) * abs(left_eye_top[1]-left_pts[0][1])
    right_ratio = (1/(abs(right_pts[2][1]-right_pts[1][1])+1e-6)) * abs(right_eye_top[1]-right_pts[0][1])
    ratio = (left_ratio + right_ratio)/2
    state = ratio > 2.6
    return ratio, state

def is_brow_frown(pts, roll_angle, face_height):
    nose = pts[1]
    # 左眉点
    left_pts = compensate_roll([pts[i] for i in [66,65,222,28,159,67,69]], roll_angle, nose)
    l66,l65,l222,l28,l159,l67,l69 = left_pts
    # 右眉点
    right_pts = compensate_roll([pts[i] for i in [296,295,443,258,386,300,302]], roll_angle, nose)
    r296,r295,r443,r258,r386,r300,r302 = right_pts

    left_shrink = abs(l66[1]-l222[1])/face_height
    left_eye_dist = abs(l159[1]-l66[1])/face_height
    right_shrink = abs(r296[1]-r443[1])/face_height
    right_eye_dist = abs(r386[1]-r296[1])/face_height

    shrink_ratio = (left_shrink+right_shrink)/2
    eye_ratio = (left_eye_dist+right_eye_dist)/2

    # 竖线辅助
    left_ratio = (1/(abs(l66[1]-l69[1])+1e-6))*abs(l159[1]-l67[1])
    right_ratio = (1/(abs(r300[1]-r302[1])+1e-6))*abs(r386[1]-r300[1])
    ratio = (left_ratio+right_ratio)/2

    frown_ratio = (1-shrink_ratio/0.045)+(1-eye_ratio/0.06)+ratio
    state = frown_ratio>0.4
    return frown_ratio, state

def is_lips_pout(pts, face_width, face_height):
    mouth_width = dist(pts[78], pts[308])
    mouth_height = dist(pts[13], pts[14])
    width_ratio = mouth_width / face_width
    height_ratio = mouth_height / face_height
    state = width_ratio < 0.25 and height_ratio < 0.03
    return (width_ratio, height_ratio), state

def is_mouth_open(pts, face_height):
    top, bottom = pts[13], pts[14]
    ratio = dist(top,bottom)/face_height
    return ratio, ratio>0.05

def detect_blink(pts, eye_idx, threshold=0.16):
    p_up = np.array(pts[eye_idx["up"]])
    p_down = np.array(pts[eye_idx["down"]])
    p_left = np.array(pts[eye_idx["left"]])
    p_right = np.array(pts[eye_idx["right"]])
    vertical_dist = np.linalg.norm(p_up - p_down)
    horizontal_dist = np.linalg.norm(p_left - p_right)
    ratio = vertical_dist / (horizontal_dist+1e-6)
    return ratio, ratio>threshold

# ================== 主循环 ==================
cap = cv2.VideoCapture(0)
timestamp = 0
#print("📷 开始实时检测：q 键退出")

LEFT_EYE = {"up":159,"down":145,"left":33,"right":133}
RIGHT_EYE = {"up":386,"down":374,"left":362,"right":263}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h,w,_=frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not mp_busy and timestamp % 4 == 0:   # 只处理 7~8 FPS
        mp_busy = True
        landmarker.detect_async(mp_image, timestamp)
    timestamp+=1

    # ================== 本人校验状态机 ==================
    if state == STATE_NO_FACE:
        cv2.putText(frame, "STATE: NO FACE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        with landmarks_lock:  # 🔒
            if face_present:
                state = STATE_VERIFYING
                #print("👀 检测到人脸，进入身份验证")

    elif state == STATE_VERIFYING:
        cv2.putText(frame, "STATE: VERIFYING", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        ok, sim = verify_identity(frame)
        if ok:
            state = STATE_LOCKED
            #print(f"🔒 身份确认成功，相似度: {sim:.2f}")
        else:
            state = STATE_REJECT
            #print(f"❌ 非本人，相似度: {sim:.2f}")

    elif state == STATE_REJECT:
        cv2.putText(frame, "STATE: REJECT (NOT YOU)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        state = STATE_NO_FACE

    elif state == STATE_LOCKED:
        cv2.putText(frame, "STATE: LOCKED (SAFE)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # ================== 这里开始：面部开关的逻辑 ==================
        with landmarks_lock:  # 🔒
            latest_pts = [(int(lm.x*w), int(lm.y*h)) for lm in latest_landmarks] if latest_landmarks else None
            face_detected = face_present

        if latest_pts:
            roll, yaw_ratio, pitch_ratio = compute_head_pose(latest_pts)
            mouse_x = int(yaw_ratio*20)
            if yaw_ratio > 0.07:
                Move_Left(mouse_x)
            elif yaw_ratio < -0.07:
                Move_Right(-mouse_x)

            mouse_y = int(pitch_ratio*20)
            if pitch_ratio > 0.07:
                Move_Down(mouse_y)
            elif pitch_ratio < -0.07:
                Move_Up(-mouse_y)
            
            face_width, face_height = face_scale(latest_pts)
            brow_up_ratio,brow_up_state = is_brow_raised(latest_pts, roll)
            brow_frown_ratio,brow_frown_state = is_brow_frown(latest_pts, roll, face_height)
            lips_ratio, lips_pout_state = is_lips_pout(latest_pts, face_width, face_height)

            left_eye_ratio,left_eye_closed = detect_blink(latest_pts, LEFT_EYE)
            right_eye_ratio,right_eye_closed = detect_blink(latest_pts, RIGHT_EYE)
            
            mouth_ratio,mouth_open_state = is_mouth_open(latest_pts, face_height)
            
            action_states = {
                "pout": lips_pout_state,
                "mouth_open": mouth_open_state,
                "brow_up": brow_up_state,
                "brow_frown": brow_frown_state,
                "eye_close": not left_eye_closed or not right_eye_closed
            }
            
                
            for action_name, is_active in action_states.items():
                function_name = user_settings.get(action_name)
                if not function_name:
                    continue

                execute_action(is_active, function_name)

            # 可视化
            cv2.putText(frame,f"Roll:{roll:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame,f"Yaw:{yaw_ratio:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame,f"Pitch:{pitch_ratio:.2f}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.putText(frame,f"Brow Raised:{brow_up_state} ({brow_up_ratio:.2f})",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,200),2)
            cv2.putText(frame,f"Brow Frown:{brow_frown_state} ({brow_frown_ratio:.2f})",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,200),2)
            cv2.putText(frame,f"Mouth Open:{mouth_open_state} ({mouth_ratio:.2f})",(10,180),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)
            cv2.putText(frame, f"Lips Pout: {lips_pout_state} ({lips_ratio[0]:.2f},{lips_ratio[1]:.2f})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
            cv2.putText(frame,f"Left Eye Closed:{left_eye_closed} ({left_eye_ratio:.2f})",(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)
            cv2.putText(frame,f"Right Eye Closed:{right_eye_closed} ({right_eye_ratio:.2f})",(10,270),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

        # 如果本人离开，自动回锁
        with landmarks_lock:  # 🔒
            if not face_present:
                state = STATE_NO_FACE
                #print("👤 本人消失，回到 NO_FACE 状态")

    cv2.imshow("Face Actions",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        dictation_system.stop()
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
#dictation_system.stop()
