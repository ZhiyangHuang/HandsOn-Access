import time
import win32api
import win32con
import win32clipboard

# ===================== 安全粘贴（优化版） =====================
class SafeClipboard:
    def __init__(self):
        self._cached_clipboard = None

    def paste_text(self, text):
        if not text:
            return

        try:
            # 缓存原剪贴板
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                self._cached_clipboard = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            else:
                self._cached_clipboard = None
            win32clipboard.CloseClipboard()
        except:
            try: win32clipboard.CloseClipboard()
            except: pass

        # 写入新内容
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
            win32clipboard.CloseClipboard()
        except:
            try: win32clipboard.CloseClipboard()
            except: pass

        # 模拟 Ctrl+V
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('V'), 0, 0, 0)
        time.sleep(0.02)
        win32api.keybd_event(ord('V'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

        # 恢复原剪贴板
        if self._cached_clipboard is not None:
            try:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, self._cached_clipboard)
                win32clipboard.CloseClipboard()
            except:
                try: win32clipboard.CloseClipboard()
                except: pass

# ===================== 状态机 =====================
class InputStateMachine:
    IDLE = 0
    LISTENING = 1
    PAUSED = 2

    def __init__(self):
        self.state = self.IDLE

    def start(self):
        self.state = self.LISTENING

    def pause(self):
        self.state = self.PAUSED

    def resume(self):
        self.state = self.LISTENING

    def stop(self):
        self.state = self.IDLE

    def allow_output(self):
        return self.state == self.LISTENING

# ===================== 增量解析 =====================
class IncrementalTextBuffer:
    def __init__(self):
        self.last_text = ""

    def reset(self):
        self.last_text = ""

    def get_delta(self, new_text):
        if not new_text:
            return ""

        # 优化：只用切片
        if new_text.startswith(self.last_text):
            delta = new_text[len(self.last_text):]
        else:
            # 如果不是前缀，直接全量替换
            delta = new_text

        self.last_text = new_text
        return delta

# ===================== 主控制器 =====================
class ContinuousDictationController:
    def __init__(self):
        self.state_machine = InputStateMachine()
        self.text_buffer = IncrementalTextBuffer()
        self.clipboard = SafeClipboard()

    def on_new_recognized_text(self, text):
        if not self.state_machine.allow_output():
            return

        delta = self.text_buffer.get_delta(text)
        if delta.strip():
            self.clipboard.paste_text(delta)

    def start(self):
        self.text_buffer.reset()
        self.state_machine.start()

    def pause(self):
        self.state_machine.pause()

    def resume(self):
        self.state_machine.resume()

    def stop(self):
        self.state_machine.stop()
        self.text_buffer.reset()
