import os
import cv2
import numpy as np
from tkinter import filedialog, messagebox, Tk
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FACE_FILE = os.path.join(BASE_DIR, "USER_FACE.npy")

# 初始化 InsightFace
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))  # 新版API用det_size

def train_user_face():
    """选择图片并生成 USER_FACE.npy"""
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    image_paths = filedialog.askopenfilenames(
        title="Select photos for training.(ten photos recommended)",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not image_paths:
        messagebox.showinfo("Hint", "No images were selected.")
        return

    person_features = []
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

        feature = faces[0].normed_embedding
        person_features.append(feature)
        success_count += 1

    if success_count == 0:
        messagebox.showwarning("Warn", "Failed to extract any facial features！")
        return

    np.save(USER_FACE_FILE, np.array(person_features))
    messagebox.showinfo("Complete", f"The selected photos have been updated！")

if __name__ == "__main__":
    train_user_face()
