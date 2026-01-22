import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FACE_FILE = r"Initialization\USER_FACE.npy"
ACTION_FILE = r"Initialization\action_time.json"
USER_SETTING_FILE = "user_setting.json"

running_processes = []  # 存放正在运行的子程序
ACTION_OPTIONS = ["None", "Left click", "Right click", "Drag", "Keyboard"]
VOICE_MODELS = ["base", "large", "medium", "small", "tiny", "VOSK"]

# ------------------- 默认用户设置 -------------------
DEFAULT_USER_SETTINGS = {
    "pout": "Drag",
    "mouth_open": "Keyboard",
    "brow_up": "Left click",
    "brow_frown": "None",
    "eye_close": "Right click",
    "voice_model": "base"
}

if not os.path.exists(USER_SETTING_FILE):
    # 第一次打开，生成默认配置
    user_settings = DEFAULT_USER_SETTINGS.copy()
    with open(USER_SETTING_FILE, "w", encoding="utf-8") as f:
        json.dump(user_settings, f, ensure_ascii=False, indent=4)
else:
    # 文件存在则读取
    with open(USER_SETTING_FILE, "r", encoding="utf-8") as f:
        user_settings = json.load(f)

# ======================= 子程序运行函数 =======================
def run_subprocess(script_name, hide_window=True):
    global running_processes

    if hide_window:
        root.withdraw()

    # 把相对路径转成绝对路径
    script_path = os.path.join(BASE_DIR, script_name)

    proc = subprocess.Popen(
        [sys.executable, script_path],
        cwd=BASE_DIR   # ⭐⭐ 核心：强制所有子程序都在 Desktop 运行
    )

    running_processes.append(proc)
    proc.wait()
    running_processes.remove(proc)

    if hide_window:
        root.deiconify()

def run_mp4_recording():
    run_subprocess(r"Initialization\mp4_recording.py")

def run_face_test_then_mouth_test():
    run_subprocess(r"Test\face_test.py")
    run_subprocess(r"Test\mouth_test.py")

def run_keyboard_muse():
    run_subprocess(r"Main\keyboard_mouse.py")

def update_user_face():
    script_path = os.path.join(BASE_DIR, "Initialization", "train_face.py")
    cmd = [sys.executable, script_path]  # 只调用脚本，不传图片路径
    proc = subprocess.Popen(cmd, cwd=BASE_DIR)
    
    running_processes.append(proc)
    proc.wait()
    running_processes.remove(proc)

def close_all_processes():
    """终止所有子程序"""
    global running_processes
    for proc in running_processes:
        try:
            proc.terminate()
        except:
            pass
    running_processes = []
    messagebox.showinfo("Hint", "All subroutines have been shut down！")

# ======================= 初始模式 =======================
def initial_mode():
    global root
    root = tk.Tk()
    root.title("User Initialization Panel")
    run_mp4_recording()
    run_face_test_then_mouth_test()
    run_keyboard_muse()
    tk.Button(root, text="Close all program", width=30, command=close_all_processes).pack(pady=5)
    root.mainloop()

# ======================= GUI =======================
def gui_mode():
    global root
    root = tk.Tk()
    root.title("User Control Panel")
    root.geometry("500x650")

    # ---------- 动作控制下拉框 ----------
    tk.Label(root, text="Control settings", font=("Arial", 12, "bold")).pack(pady=10)

    # 读取 action_time.json
    try:
        with open(ACTION_FILE, "r", encoding="utf-8") as f:
            actions = json.load(f)
    except FileNotFoundError:
        actions = {}

    # 读取已有用户设置
    try:
        with open(USER_SETTING_FILE, "r", encoding="utf-8") as f:
            user_settings = json.load(f)
    except FileNotFoundError:
        user_settings = {}

    frames = {}
    for action_name in ["pout", "mouth_open", "brow_up", "brow_frown", "eye_close"]:
        action_info = actions.get(action_name, {})
        if action_info.get("end") is None:
            user_settings[action_name] = "None"
            continue
        frame = tk.Frame(root)
        frame.pack(pady=5, anchor="w", padx=20)
        tk.Label(frame, text=f"{action_name} method：").pack(side="left")
        combo = ttk.Combobox(frame, values=ACTION_OPTIONS, state="readonly")

        if action_name == "mouth_open":
            combo.set("Keyboard")
            combo.config(state="disabled")  # 禁止修改
        else:
            combo.set(user_settings.get(action_name, ACTION_OPTIONS[0]))
        combo.pack(side="left")
        frames[action_name] = combo

    # ---------- 语音模型下拉框 ----------
    tk.Label(root, text="Voice model selection", font=("Arial", 12, "bold")).pack(pady=10)
    voice_frame = tk.Frame(root)
    voice_frame.pack(pady=5, anchor="w", padx=20)
    tk.Label(voice_frame, text="Speech model：").pack(side="left")
    voice_combo = ttk.Combobox(voice_frame, values=VOICE_MODELS, state="readonly")
    voice_combo.set(user_settings.get("voice_model", VOICE_MODELS[0]))
    voice_combo.pack(side="left")

    # ---------- 保存按钮 ----------
    def save_settings():
        for action_name, combo in frames.items():
            user_settings[action_name] = combo.get()
        user_settings["voice_model"] = voice_combo.get()
        with open(USER_SETTING_FILE, "w", encoding="utf-8") as f:
            json.dump(user_settings, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("Complete", "User settings saved！")

    tk.Button(root, text="Save settings", width=30, command=save_settings).pack(pady=20)

    # ---------- 子程序控制 ----------
    tk.Label(root, text="Main program control", font=("Arial", 12, "bold")).pack(pady=5)

    tk.Button(root, text="Initialize recording", width=30, command=run_mp4_recording).pack(pady=5)
    tk.Button(root, text="Update user facial data", width=30, command=update_user_face).pack(pady=5)
    tk.Button(root, text="Beginner's Tutorial", width=30, command=run_face_test_then_mouth_test).pack(pady=5)
    tk.Button(root, text="Enable keyboard control", width=30, command=run_keyboard_muse).pack(pady=5)
    tk.Button(root, text="Close all program", width=30, command=close_all_processes).pack(pady=5)

    root.mainloop()

# ======================= 主逻辑 =======================
if __name__ == "__main__":

    # 读取 action_time.json
    try:
        with open(ACTION_FILE, "r", encoding="utf-8") as f:
            actions = json.load(f)
    except FileNotFoundError:
        actions = {}

    mouth_end = actions.get("mouth_open", {}).get("end")

    # 如果没有用户脸数据 或 mouth_open.end 为 None → 初始模式
    if not os.path.exists(USER_FACE_FILE) or mouth_end is None:
        #print("mouth_open 未完成或无 USER_FACE.npy，进入初始模式...")
        initial_mode()
    else:
        #print("条件满足，进入 GUI 模式...")
        gui_mode()
