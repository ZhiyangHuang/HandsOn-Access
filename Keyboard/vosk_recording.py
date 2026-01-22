import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import win32api
import win32con
import time
import threading

# ================ win32键盘对照表 =================
class _VoiceKeyboardConfig:
    MODEL_PATH = r"../Model\Vosk"
    SAMPLE_RATE = 16000
    MIN_INTERVAL = 0.5
    COLLECT_TIMEOUT = 0.35   # 350ms 语音组合窗口
    MAX_TOKENS = 3

    # 只允许识别这些词（Grammar）
    GRAMMAR = [
        "a", "b", "c", "d", "e", "i", "k","l", "n", "o",
        "p", "q", "r", "u", "w", "x", "y",

        "apple", "banana", "candle", "dragon", "elephant", "forest", "goat",
        "hotel", "igloo", "jungle", "monkey", "pumpkin",
        "snake", "tiger", "violin", "zebra",

        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "zero",

        "get out", "space jump", "capital lock", "upper", "control",
        "window", "option", "scroll lock",

        "space", "back space", "enter","delete",
        "up", "down", "left", "right",

        "back tick", "minus sign", "equals sign",
        "square open", "square close", "back line", "semi",
        "tick", "comma", "dot", "forward line",
    
        "fox one", "fox two", "fox three", "fox four",
        "fox five", "fox six", "fox seven", "fox eight",
        "fox nine", "fox ten", "fox eleven", "fox twelve"
    ]

    VK_MAP = {
        # 字母 A-Z
        "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45,
        "i": 0x49, "k": 0x4B, "l": 0x4C, "n": 0x4E, "o": 0x4F,
        "p": 0x50, "q": 0x51, "r": 0x52, "u": 0x55, "w": 0x57,
        "x": 0x58, "y": 0x59,
    
        "apple": 0x41, "banana": 0x42, "candle": 0x43, "dragon": 0x44, "elephant": 0x45,
        "forest": 0x46, "goat": 0x47, "hotel": 0x48, "igloo": 0x49, "jungle": 0x4A,
        "monkey": 0x4D, "pumpkin": 0x50, "snake": 0x53, "tiger": 0x54, "uniform": 0x55,
        "violin": 0x56, "zebra": 0x5A,

        # 数字 0-9
        "zero": 0x30, "one": 0x31, "two": 0x32, "three": 0x33, "four": 0x34,
        "five": 0x35, "six": 0x36, "seven": 0x37, "eight": 0x38, "nine": 0x39,

        # 符号
        "back tick": 0xC0,
        "minus sign": 0xBD, "equals sign": 0xBB,
        "square open": 0xDB, "square close": 0xDD,
        "back line": 0xDC, "semi": 0xBA, "tick": 0xDE,
        "comma": 0xBC, "dot": 0xBE,
        "forward line": 0xBF,

        # 功能键
        "fox one": 0x70, "fox two": 0x71, "fox three": 0x72, "fox four": 0x73,
        "fox five": 0x74, "fox six": 0x75, "fox seven": 0x76, "fox eight": 0x77,
        "fox nine": 0x78, "fox ten": 0x79, "fox eleven": 0x7A, "fox twelve": 0x7B,

        # 控制键
        "get out": 0x1B, "space jump": 0x09, "capital lock": 0x14,
        "upper": 0x10, "control": 0x11, "window": 0x5B,
        "option": 0x12, "scroll lock": 0x91,
    
        # 编辑及导航
        "space": 0x20, "back space": 0x08, "enter": 0x0D, "delete": 0x2E,
        "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27
    }

    String_MAP = {
        # 字母 A-Z
        "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
        "i": "I", "k": "K", "l": "L", "n": "N", "o": "O",
        "p": "P", "q": "Q", "r": "R",  "u": "U", "w": "W",
        "x": "X", "y": "Y",
        
        "apple": "A", "banana": "B", "candle": "C", "dragon": "D", "elephant": "E",
        "forest": "F", "goat": "G", "hotel": "H", "igloo": "I", "jungle": "J",
        "monkey": "M","pumpkin": "P", "snake": "S", "tiger": "T","uniform": "U",
        "violin": "V", "zebra": "Z",

        # 数字 0-9
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",

        # 符号
        "back tick": "`",
        "minus sign": "-", "equals sign": "=",
        "square open": "[", "square close": "]",
        "back line": "\\", "semi": ";", "tick": "'",
        "comma": ",", "dot": ".",
        "forward line": "/",

        # 功能键
        "fox one": "F1", "fox two": "F2", "fox three": "F3", "fox four": "F4",
        "fox five": "F5", "fox six": "F6", "fox seven": "F7", "fox eight": "F8",
        "fox nine": "F9", "fox ten": "F10", "fox eleven": "F11", "fox twelve": "F12",

        # 控制键
        "get out": "Esc", "space jump": "Tab", "capital lock": "Caps Lock",
        "upper": "Shift", "control": "Ctrl", "window": "Win",
        "option": "Alt", "scroll lock": "scrolllock",
    
        # 编辑及导航
        "space": "s p a c e", "back space": "Backspace", "enter": "Enter", "delete": "Delete",
        "up": "up", "down": "down", "left": "left", "right": "right"
    }

# ======================= 键盘执行逻辑 =========================
class VoiceKeyboardExecutor:
    def __init__(self, vk_map, string_map=None, test_mode=False):
        self.VK_MAP = vk_map
        self.STRING_MAP = string_map or vk_map
        self.test_mode = test_mode
        self.sorted_grammar = sorted(_VoiceKeyboardConfig.GRAMMAR, key=len, reverse=True)
        self.outprint = None

    def _emit(self, keys):
        if self.test_mode:
            self.outprint = '+'.join(self.STRING_MAP.get(k, k) for k in keys)
        else:
            for k in keys:
                vk = self.VK_MAP.get(k)
                if vk:
                    win32api.keybd_event(vk, 0, 0, 0)
            for k in reversed(keys):
                vk = self.VK_MAP.get(k)
                if vk:
                    win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)

    def single_key(self, name):
        self._emit([name])

    def dual_key(self, a, b):
        if a == b:
            return
        self._emit([a, b])

    def triple_key(self, a, b, c):
        if len({a, b, c}) < 3:
            return
        self._emit([a, b, c])

    def split_by_grammar(self, text):
        text = text.lower().strip()
        words = []
        while text:
            match = None
            for g in self.sorted_grammar:
                if text.startswith(g):
                    match = g
                    break
            if match:
                words.append(match)
                text = text[len(match):].strip()
            else:
                text = text[1:].strip()
        return words

# ======================= 主系统 =========================
class VoiceKeyboardSystem:
    def __init__(self, test_mode=False):
        cfg = _VoiceKeyboardConfig

        self.model = Model(cfg.MODEL_PATH)
        self.recognizer = KaldiRecognizer(
            self.model, cfg.SAMPLE_RATE, json.dumps(cfg.GRAMMAR)
        )

        # executor
        if test_mode:
            self.executor = VoiceKeyboardExecutor(cfg.VK_MAP, string_map=cfg.String_MAP, test_mode=True)
        else:
            self.executor = VoiceKeyboardExecutor(cfg.VK_MAP)

        self.outprint = None  # 最终输出
        self.audio_queue = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.token_buffer = []
        self.executing = False

        self.running = False
        self.paused = False
        self.last_trigger_time = 0
        self.min_interval = cfg.MIN_INTERVAL
        self.last_token_time = 0
        self.end_grace_period = 0.5

    def audio_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def start(self):
        if self.running:
            return
        self.running = True

        # 音频流
        self.stream = sd.InputStream(
            samplerate=_VoiceKeyboardConfig.SAMPLE_RATE,
            channels=1,
            dtype='int16',
            callback=self.audio_callback
        )
        self.stream.start()

        # 循环线程
        threading.Thread(target=self.process_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

    # ---------------- 内部处理 ----------------
    def handle_token(self, word):
        with self.lock:
            if self.executing:
                return
            if self.token_buffer and self.token_buffer[-1] == word:
                return
            self.token_buffer.append(word)
            self.last_token_time = time.time()

    def execute_buffer(self):
        with self.lock:
            if not self.token_buffer:
                return
            self.executing = True
            try:
                b = self.token_buffer
                if len(b) == 1:
                    self.executor.single_key(b[0])
                elif len(b) == 2:
                    self.executor.dual_key(b[0], b[1])
                elif len(b) == 3:
                    self.executor.triple_key(b[0], b[1], b[2])
                # 将 executor.outprint 映射到系统 outprint
                if self.executor.outprint:
                    self.outprint = self.executor.outprint
                    self.executor.outprint = None
            finally:
                self.token_buffer.clear()
                self.executing = False

    def should_execute(self):
        with self.lock:
            if not self.token_buffer:
                return False
            if len(self.token_buffer) >= _VoiceKeyboardConfig.MAX_TOKENS:
                return True
            if time.time() - self.last_token_time > _VoiceKeyboardConfig.COLLECT_TIMEOUT:
                return True
            return False

    # ---------------- 主循环 ----------------
    def process_loop(self):
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if data is None:
                break

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()
                if not text:
                    continue
                now = time.time()
                if now - self.last_trigger_time < self.min_interval:
                    continue
                for w in self.executor.split_by_grammar(text):
                    self.handle_token(w)
                self.last_trigger_time = now

            if self.should_execute():
                self.execute_buffer()
