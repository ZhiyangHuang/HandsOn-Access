import os
import cv2
import time
import datetime
import numpy as np
from tkinter import filedialog, messagebox, Tk
from insightface.app import FaceAnalysis
import hashlib
import platform
import uuid
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERACTION_KEY_PATH = os.path.join(BASE_DIR, "interaction_key.jfhzy")

# 初始化 InsightFace
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))  # 新版API用det_size

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

def train_user_face():
    """选择图片并生成 interaction_key.jfhzy"""
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    image_paths = filedialog.askopenfilenames(
        title="Select photos for training.(ten photos recommended)",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not image_paths:
        messagebox.showinfo("Hint", "No images were selected.")
        return

    user_embedding = []
    success_count = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取图片: {path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        if len(faces) == 0:
            print(f"{path} 未检测到人脸")
            continue

        user_embedding.append(faces[0].normed_embedding)
        success_count += 1

    if success_count == 0:
        messagebox.showwarning("Warn", "Failed to extract any facial features！")
        return

    all_emb = np.stack(user_embedding, axis=0).astype(np.float32)
    print("训练 embedding shape:", all_emb.shape)
    save_encrypted_embedding(INTERACTION_KEY_PATH, all_emb)
    messagebox.showinfo("Complete", f"The selected photos have been updated！")

if __name__ == "__main__":
    train_user_face()
