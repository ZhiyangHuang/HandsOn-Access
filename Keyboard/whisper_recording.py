import sounddevice as sd
import numpy as np
import threading
import queue
import time
import torch
import whisper

# =================== Whisper 模型加载 ===================
def load_whisper_model_auto(model_name="tiny"):
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    fp16 = use_gpu
    if not use_gpu:
        print("⚠ 未检测到 CUDA GPU，使用 CPU + FP32")
    model = whisper.load_model(model_name)
    if use_gpu:
        model = model.to(device)
    return model, device, fp16

# =================== 音频采集类 ===================
class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1, block_duration=3):
        self.samplerate = samplerate
        self.channels = channels
        self.block_duration = block_duration

        self.q = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=5)

        self.running = False
        self.stream = None

        self.block_samples = int(samplerate * block_duration)
        self.audio_buffer = np.zeros((self.block_samples, channels), dtype=np.float32)
        self.write_pos = 0

        self.buffer_lock = threading.Lock()  # 🔒

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.audio_callback
        )
        self.stream.start()
        threading.Thread(target=self._process_audio_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def _process_audio_loop(self):
        while self.running:
            try:
                data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            data = data.astype(np.float32)
            frames = data.shape[0]
            read_pos = 0

            with self.buffer_lock:
                while frames > 0:
                    space = self.block_samples - self.write_pos
                    write_len = min(space, frames)

                    self.audio_buffer[
                        self.write_pos : self.write_pos + write_len
                    ] = data[read_pos : read_pos + write_len]

                    self.write_pos += write_len
                    read_pos += write_len
                    frames -= write_len

                    # buffer 满 → 送入识别队列
                    if self.write_pos >= self.block_samples:
                        audio_block = self.audio_buffer[:, 0].copy()

                        self.write_pos = 0

                        try:
                            self.audio_queue.put_nowait(audio_block)
                        except queue.Full:
                            try:
                                self.audio_queue.get_nowait()
                                self.audio_queue.put_nowait(audio_block)
                            except queue.Empty:
                                pass

# =================== Whisper识别器 ===================
class WhisperRecognizer:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()

    def recognize(self, audio_block):
        with self.lock:
            result = self.model.transcribe(audio_block, language="en")
        return result.get("text", "").strip()

# =================== 智能延迟开关 ===================
class SmartToggle:
    def __init__(self, delay=4.0):
        self.active = False          # 当前是否 start() 激活
        self.delay = delay
        self.last_trigger_time = None  # start() 最后触发时间
        self.stop_trigger_time = None  # stop() 第一次触发时间
        self.lock = threading.Lock()  # 🔒

    def trigger_on(self):
        """start() 刷新激活状态"""
        with self.lock:  # 🔒
            self.active = True
            self.last_trigger_time = time.time()
            self.stop_trigger_time = None  # start() 时清空延迟计时

    def trigger_off(self):
        """stop() 触发延迟关闭，只记录第一次"""
        with self.lock:  # 🔒
            if self.stop_trigger_time is None:
                self.stop_trigger_time = time.time()
            self.active = False

    def is_active(self):
        with self.lock:  # 🔒
            if self.active:
                return True
            if self.stop_trigger_time is None:
                return False
            # 延迟还没过
            return (time.time() - self.stop_trigger_time) < self.delay

    def deactivate(self):
        with self.lock:  # 🔒
            self.active = False
            self.last_trigger_time = None
            self.stop_trigger_time = None

# =================== 实时语音听写控制器 ===================
class RealTimeDictation:
    def __init__(self, controller = None, model="tiny", block_duration=3, delay=3.0):
        self.controller = controller
        model, device, fp16_flag = load_whisper_model_auto(model)  # 可换成 base / small / medium
        self.recognizer = WhisperRecognizer(model)
        self.audio_recorder = AudioRecorder(block_duration=block_duration)
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.delay = delay
        self._pause_thread = None
        self._pause_thread_lock = threading.Lock()

        # 异步识别线程
        self._worker_thread = threading.Thread(target=self._recognize_worker, daemon=True)
        self._worker_thread.start()

        self.string_only_mode = False
        if controller is None or controller is True:
            self.string_only_mode = True

        self.outprint = None

    # 异步识别线程
    def _recognize_worker(self):
        while True:
            if not self.running:
                time.sleep(0.1)
                continue
            try:
                audio_block = self.audio_recorder.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.lock:
                if self.paused:
                    continue

            try:
                text = self.recognizer.recognize(audio_block)
                if text:
                    if self.string_only_mode:
                        self.outprint = text
                    else:
                        self.controller.on_new_recognized_text(text)
            except Exception as e:
                #print("Whisper识别异常:", e)
                pass

    # ---------------- 启动 ----------------
    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
        if not self.string_only_mode:
            self.controller.start()
        self.audio_recorder.start()

    # ---------------- 停止（带延迟停用） ----------------
    def stop(self):
        with self.lock:
            self.paused = True
        threading.Thread(target=self._delayed_stop_thread, daemon=True).start()

    def _delayed_stop_thread(self):
        stop_time = time.time()
        while True:
            time.sleep(0.1)
            with self.lock:
                if not self.paused:
                    return  # 被 resume 了
                if time.time() - stop_time >= self.delay:
                    break
        if not self.string_only_mode:
            self.controller.stop()
        self.audio_recorder.stop()
        with self.lock:
            self.running = False

    # ---------------- 统一暂停/恢复 ----------------
    def set_paused(self, paused: bool):
        with self.lock:
            if self.paused == paused:
                return
            self.paused = paused

        if paused:
            # 延迟线程管理
            with self._pause_thread_lock:
                if self._pause_thread is None or not self._pause_thread.is_alive():
                    def delayed_pause():
                        start_time = time.time()
                        while True:
                            time.sleep(0.1)
                            with self.lock:
                                if not self.paused:
                                    return
                                if time.time() - start_time >= self.delay:
                                    break
                        if not self.string_only_mode:
                            self.controller.pause()
                    self._pause_thread = threading.Thread(target=delayed_pause, daemon=True)
                    self._pause_thread.start()
        else:
            if not self.string_only_mode:
                self.controller.resume()
