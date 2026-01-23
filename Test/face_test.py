import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque
import json

# ----------------- 脚本根目录 -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- 文件路径 -----------------
USER_SETTING_FILE = os.path.join(BASE_DIR, "..", "user_setting.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "face_landmarker.task")
MOUSE_PNG_PATH = os.path.join(BASE_DIR, "mouse_box.png")

# ----------------- 检查用户设置 -----------------
if not os.path.exists(USER_SETTING_FILE):
    raise FileNotFoundError("❌ 未检测到 user_setting.json，系统未初始化，程序退出。")

try:
    with open(USER_SETTING_FILE, "r", encoding="utf-8") as f:
        USER_SETTINGS = json.load(f)
except FileNotFoundError:
    USER_SETTINGS = {}
# ================== MediaPipe 初始化 ==================
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

# =============== json状态列表 ====================
def execute_action(is_active, function_name):
    if function_name == "Left click" and is_active:
        if pitch_ratio > 0.07 or pitch_ratio < -0.07 or yaw_ratio > 0.07 or yaw_ratio < -0.07:
            return
        cv2.circle(canvas, (red_x, point_y), 5, (0,0,255), -1) #红色

    elif function_name == "Right click" and is_active:
        if pitch_ratio > 0.07 or pitch_ratio < -0.07 or yaw_ratio > 0.07 or yaw_ratio < -0.07:
            return
        cv2.circle(canvas, (blue_x, point_y), 5, (255,0,0), -1) #蓝色

    elif function_name == "Drag" and is_active:
        cv2.circle(canvas, (red_x, point_y), 5, (0,0,255), -1) #红色

    elif function_name == "Keyboard" and is_active:
        if pitch_ratio > 0.07 or pitch_ratio < -0.07 or yaw_ratio > 0.07 or yaw_ratio < -0.07:
            return

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
    """抬眉"""
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
    """皱眉"""
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
    """张嘴"""
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

# ==========================================
def resize_with_white_bg(img, target_w, target_h):
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

    # 🔥 返回映射信息
    return canvas, scale, offset_x, offset_y

# ====================图片png加载===========
png = cv2.imread(MOUSE_PNG_PATH, cv2.IMREAD_UNCHANGED)  # 支持透明通道
assert png is not None, "PNG 加载失败"

png_h, png_w = png.shape[:2]
png_small = cv2.resize(
    png,
    (png_w // 10, png_h // 10),
    interpolation=cv2.INTER_AREA
)

def overlay_image(bg, fg, x, y):
    """
    fg 可以是带 alpha 或不带 alpha 的图片
    """
    h, w = fg.shape[:2]
    if fg.shape[2] == 4:  # 有 alpha
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = alpha * fg[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c]
    else:  # 无 alpha
        bg[y:y+h, x:x+w] = fg
# 左上角位置
png_x = 30
png_y = 30

# ================== 主循环 ==================
cap = cv2.VideoCapture(0)
timestamp = 0
#print("📷 开始实时检测：q 键退出")

LEFT_EYE = {"up":159,"down":145,"left":33,"right":133}
RIGHT_EYE = {"up":386,"down":374,"left":362,"right":263}

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret: break
    h, w = frame.shape[:2]

    # ============== 白色背景画布（自适应摄像头大小） ==============
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 缩小摄像头画面
    video_small = cv2.resize(frame, (w // 3, h // 3))
    vh, vw, _ = video_small.shape
    
    # 视频
    overlay_image(canvas, video_small, w-vw-20, h-vh-20)
    # ==========Mediapipe图片识别=================
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarker.detect_async(mp_image, timestamp)
    timestamp+=1

    if latest_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in latest_landmarks]
        roll, yaw_ratio, pitch_ratio = compute_head_pose(pts)
        face_width, face_height = face_scale(pts)

        mouse_x = int(yaw_ratio*20)
        if yaw_ratio > 0.07 or yaw_ratio < -0.07:
            png_x -= mouse_x

        mouse_y = int(pitch_ratio*20)
        if pitch_ratio > 0.07 or pitch_ratio < -0.07:
            png_y += mouse_y

        canvas_h, canvas_w = canvas.shape[:2]
        png_h, png_w = png_small.shape[:2]
        png_x = max(0, min(canvas_w - png_w, png_x))
        png_y = max(0, min(canvas_h - png_h, png_y))

        # 动作
        brow_up_ratio,brow_up_state = is_brow_raised(pts, roll)
        brow_frown_ratio,brow_frown_state = is_brow_frown(pts, roll, face_height)
        mouth_ratio,mouth_open_state = is_mouth_open(pts, face_height)
        lips_ratio, lips_pout_state = is_lips_pout(pts, face_width, face_height)
        left_eye_ratio,left_eye_closed = detect_blink(pts, LEFT_EYE)
        right_eye_ratio,right_eye_closed = detect_blink(pts, RIGHT_EYE)

        action_states = {
            "pout": lips_pout_state,
            "mouth_open": mouth_open_state,
            "brow_up": brow_up_state,
            "brow_frown": brow_frown_state,
            "eye_close": not left_eye_closed or not right_eye_closed
        }

        # 可视化
        cv2.putText(canvas,f"Roll:{roll:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(canvas,f"Yaw:{yaw_ratio:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(canvas,f"Pitch:{pitch_ratio:.2f}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.putText(canvas,f"Brow Raised:{brow_up_state} ({brow_up_ratio:.2f})",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,200),2)
        cv2.putText(canvas,f"Brow Frown:{brow_frown_state} ({brow_frown_ratio:.2f})",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,200),2)
        cv2.putText(canvas,f"Mouth Open:{mouth_open_state} ({mouth_ratio:.2f})",(10,180),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)
        cv2.putText(canvas, f"Lips Pout: {lips_pout_state} ({lips_ratio[0]:.2f},{lips_ratio[1]:.2f})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
        cv2.putText(canvas,f"Left Eye Closed:{left_eye_closed} ({left_eye_ratio:.2f})",(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)
        cv2.putText(canvas,f"Right Eye Closed:{right_eye_closed} ({right_eye_ratio:.2f})",(10,270),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)


    else:
        cv2.putText(canvas,"No face detected",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        # 没检测到人脸时，给变量默认值
        roll = yaw_ratio = pitch_ratio = 0
        brow_up_ratio = brow_frown_ratio = mouth_ratio = 0
        lips_ratio = [0,0]
        brow_up_state = brow_frown_state = mouth_open_state = lips_pout_state = False
        left_eye_ratio = right_eye_ratio = 0
        left_eye_closed = right_eye_closed = False

        action_states = {
            "pout": False,
            "mouth_open": False,
            "brow_up": False,
            "brow_frown": False,
            "eye_close": False
        }

    # 确保窗口可缩放
    cv2.namedWindow("Face Actions", cv2.WINDOW_NORMAL)

    # PNG
    red_x = int(png_x + png_w*1/4)
    point_y = int(png_y + png_h*1/4)
    blue_x = int(png_x + png_w*3/4)
    
    # 画圆点：标记左右键
    for action_name, is_active in action_states.items():
        function_name = USER_SETTINGS.get(action_name)
        if not function_name:
            continue

        execute_action(is_active, function_name)
    # 主循环里：
    overlay_image(canvas, png_small, png_x, png_y)
    x, y, win_w, win_h = cv2.getWindowImageRect("Face Actions")
    
    display, scale, ox, oy = resize_with_white_bg(canvas, win_w, win_h)
    cv2.imshow("Face Actions", display)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
