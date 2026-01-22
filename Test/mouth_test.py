import os
import sys
import cv2
import numpy as np
import time
import json
from playsound import playsound

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

# 指令映射
COMMAND_AUDIO_FOLDER = os.path.join(BASE_DIR, "Command_Speech")
PNG_FILE = os.path.join(BASE_DIR, "keyboard_box.png")

String_MAP = {
    "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
    "i": "I", "k": "K", "l": "L", "n": "N", "o": "O",
    "p": "P", "q": "Q", "r": "R", "u": "U", "w": "W",
    "x": "X", "y": "Y",
    "apple": "A", "banana": "B", "candle": "C", "dragon": "D", "elephant": "E",
    "forest": "F", "goat": "G", "hotel": "H", "igloo": "I", "jungle": "J",
    "monkey": "M","pumpkin": "P", "snake": "S", "tiger": "T","uniform": "U",
    "violin": "V", "zebra": "Z",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "back tick": "`", "minus sign": "-", "equals sign": "=",
    "square open": "[", "square close": "]",
    "back line": "\\", "semi": ";", "tick": "'",
    "comma": ",", "dot": ".", "forward line": "/",
    "fox one": "F1", "fox two": "F2", "fox three": "F3", "fox four": "F4",
    "fox five": "F5", "fox six": "F6", "fox seven": "F7", "fox eight": "F8",
    "fox nine": "F9", "fox ten": "F10", "fox eleven": "F11", "fox twelve": "F12",
    "get out": "Esc", "space jump": "Tab", "capital lock": "Caps Lock",
    "upper": "Shift", "control": "Ctrl", "window": "Win",
    "option": "Alt", "scroll lock": "scrolllock",
    "space": "s p a c e", "back space": "Backspace", "enter": "Enter", "delete": "Delete",
    "up": "up", "down": "down", "left": "left", "right": "right"
}

# ================== 辅助函数 ==================
def vosk_test():
    return VoiceKeyboardSystem(True)  # VOSK

def whisper_test(model):
    return RealTimeDictation(True, model)  # tiny / base / small / medium / large

# ================== 主程序 ==================
def main():
    try:
        with open(USER_SETTING_FILE, "r", encoding="utf-8") as f:
            user_settings = json.load(f)
    except FileNotFoundError:
        user_settings = {}

    if user_settings["voice_model"] == "VOSK":
        dictation_system = vosk_test()
    else:
        dictation_system = whisper_test(user_settings["voice_model"])

    dictation_system.start()
    print("🎤 系统已启动，模拟语音识别...")

    # OpenCV窗口初始化
    window_name = "Speech Output"
    cv2.namedWindow(window_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    line_type = cv2.LINE_AA
    padding = 20

    # 读取 PNG 图片
    png_img = cv2.imread(PNG_FILE, cv2.IMREAD_UNCHANGED)
    if png_img is None:
        print("⚠ 未找到 PNG 图片，文字下方将不显示图片")

    last_text = ""
    test_results = {}

    if user_settings["voice_model"] == "VOSK":
        try:
            success = "Time Out..."
            for command, answer in String_MAP.items():
                # ---- 播放对应的指令音频 ----
                audio_file_path = os.path.join(COMMAND_AUDIO_FOLDER, f"{command}.mp3")
                if os.path.exists(audio_file_path):
                    try:
                        playsound(audio_file_path)
                    except Exception as e:
                        print(f"⚠ 无法播放 {audio_file_path}: {e}")
                else:
                        print(f"⚠ 指令音频不存在: {audio_file_path}")
                        
                test_results[command] = []
                for trial in range(3):
                    start_time = time.time()
                    recognized_text = ""
                    while time.time() - start_time < 15:
                        recognized_text = dictation_system.outprint or ""
                        if recognized_text.strip():
                            break
                        time.sleep(0.1)

                        # ---- 在窗口显示当前指令和上次识别 ----
                        display_lines = [
                            f"Instruction: {command}",
                            f"Pre_Results: {'3' if trial == 0 else trial} : {success}"
                        ]

                        # 文字尺寸计算
                        text_heights = []
                        max_width = 0
                        for line in display_lines:
                            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                            text_heights.append(h)
                            max_width = max(max_width, w)
                        total_text_height = sum(text_heights) + padding*(len(display_lines)+1)

                        # 图片尺寸
                        img_height = total_text_height
                        img_width = max_width + 2*padding
                        if png_img is not None:
                            img_h, img_w = png_img.shape[:2]
                            img_height += img_h + padding
                            img_width = max(img_width, img_w + 2*padding)

                        # 创建白色背景
                        canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

                        # 写入文字（窗口正中）
                        y = padding
                        for i, line in enumerate(display_lines):
                            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                            x = (img_width - w) // 2
                            canvas[y+h, x] = 0
                            cv2.putText(canvas, line, (x, y+h), font, font_scale, (0,0,0), thickness, line_type)
                            y += h + padding

                        # 贴入 PNG 图片（文字下方）
                        if png_img is not None:
                            x_offset = (img_width - png_img.shape[1]) // 2
                            y_offset = y
                            if png_img.shape[2] == 4:
                                alpha_s = png_img[:, :, 3] / 255.0
                                alpha_l = 1.0 - alpha_s
                                for c in range(3):
                                    canvas[y_offset:y_offset+png_img.shape[0], x_offset:x_offset+png_img.shape[1], c] = (
                                        alpha_s * png_img[:, :, c] + alpha_l * canvas[y_offset:y_offset+png_img.shape[0], x_offset:x_offset+png_img.shape[1], c]
                                    )
                            else:
                                canvas[y_offset:y_offset+png_img.shape[0], x_offset:x_offset+png_img.shape[1]] = png_img

                        cv2.imshow(window_name, canvas)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                    # 识别结果记录
                    if recognized_text.strip():
                        success = "Time Out..."
                        if recognized_text.strip() == answer:
                            success = "Success"
                        else:
                            success = "Error"
                        print(f"⏱ 第 {trial+1} 次识别: '{recognized_text.strip()}' -> {success}")
                        test_results[command].append(recognized_text.strip())
                        dictation_system.outprint = ""  # 重置
                    else:
                        print(f"⏱ 第 {trial+1} 次超时，跳过")
                        success = "Time Out..."
                        test_results[command].append(None)
        finally:
            dictation_system.stop()
            cv2.destroyAllWindows()

        print("🎉 所有指令测试完成！")
        for cmd, results in test_results.items():
            print(f"{cmd}: {results}")

        return test_results

    else:
        # Whisper 实时显示逻辑
        try:
            while True:
                text = dictation_system.outprint or ""
                if text != last_text:
                    last_text = text
                display_text = text if text.strip() else "Waiting for speech..."

                (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                img_height = text_height + 2*padding
                img_width = text_width + 2*padding

                if png_img is not None:
                    img_h, img_w = png_img.shape[:2]
                    img_height += img_h + padding
                    img_width = max(img_width, img_w + 2*padding)

                canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

                org = ((img_width - text_width)//2, text_height + padding)
                cv2.putText(canvas, display_text, org, font, font_scale, (0,0,0), thickness, line_type)

                if png_img is not None:
                    x_offset = (img_width - png_img.shape[1]) // 2
                    y_offset = text_height + 2*padding
                    if png_img.shape[2] == 4:
                        alpha_s = png_img[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(3):
                            canvas[y_offset:y_offset+png_img.shape[0], x_offset:x_offset+png_img.shape[1], c] = (
                                alpha_s * png_img[:, :, c] + alpha_l * canvas[y_offset:y_offset+png_img.shape[0], x_offset:x_offset+png_img.shape[1], c]
                            )
                    else:
                        canvas[y_offset:y_offset+png_img.shape[0], x_offset:x_offset+png_img.shape[1]] = png_img

                cv2.imshow(window_name, canvas)
                if cv2.waitKey(1)&0xFF==ord('q'): break
        finally:
            dictation_system.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

