# HandsOn-Access

> 🎯 **Hands-free Human–Computer Interaction System**
> A multimodal access system based on **Face Recognition + Head Movement + Voice Recognition**, designed for accessibility and hands-free control on Windows.

---

## ✨ Project Overview

**HandsOn-Access** integrates multiple AI technologies to enable users to control a computer without using hands:

* 👤 **Face recognition** for user identity verification
* 🙂 **Face & head movement detection** for mouse control
* 🎤 **Speech recognition** (command-level & dictation)
* ⌨️ **Voice-driven keyboard & clipboard control**

This project is especially suitable for:

* Accessibility / assistive technology
* Human–Computer Interaction (HCI) research
* AI + CV + Speech integration demos

---

## 🧠 Technologies Used

* **InsightFace** – Face recognition
* **MediaPipe Face Landmarker** – Face & head pose tracking
* **Whisper (OpenAI)** – High-accuracy speech-to-text
* **Vosk** – Lightweight command-based speech recognition
* **OpenCV** – Real-time camera processing
* **PyTorch** – Model inference backend
* **Tkinter** – GUI interface (Windows)

---

## 📦 Environment Requirements

* OS: **Windows 10 / 11**
* Python: **3.9 – 3.10 (recommended)**
* GPU: Optional (CUDA supported but not required)

---

## 📚 Python Dependencies

Install required packages:

```bash
pip install opencv-python numpy mediapipe insightface torch torchvision torchaudio
pip install sounddevice vosk whisper playsound pywin32
```

---

## 🧠 AI Model Download & Placement Guide (IMPORTANT)

Some AI models are **auto-downloaded to system cache**, while others **must be manually placed** in the `Model/` directory.

---

### 🔹 Whisper (OpenAI Speech Recognition)

#### 📌 Where to find Whisper model download links?

All official Whisper model URLs are defined here:

👉 [https://github.com/openai/whisper/blob/main/whisper/__init__.py](https://github.com/openai/whisper/blob/main/whisper/__init__.py)

Inside this file you will find:

```python
_MODELS = {
    "tiny": "...",
    "base": "...",
    "small": "...",
    "medium": "...",
    "large": "...",
}
```

Each entry corresponds to an official model download link.

---

#### 📂 Whisper default installation location

Whisper models are automatically downloaded to:

```
C:\Users\<YourUsername>\.cache\whisper\
```

⚠️ If you want to **download manually**, please move the **.pt** file to the corresponding folder.

---

### 🔹 InsightFace (Face Recognition)

#### 📥 Official model zoo

👉 [https://github.com/deepinsight/insightface/tree/master/model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)

Recommended model:

```
buffalo_l
```

---

#### 📂 InsightFace default installation location

By default, InsightFace downloads models to:

```
C:\Users\<YourUsername>\.insightface\models\
```

---

### 🔹 Vosk (Command-level Speech Recognition)

#### 📥 Download Vosk models

👉 [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

Recommended English model:

```
vosk-model-small-en-us-0.15
```

---

#### 📂 Required placement (IMPORTANT)

After downloading and extracting, **rename the folder to `Vosk`** and place it here:

```
Model/Vosk/
```

Directory structure example:

```
Model/
 └─ Vosk/
    ├─ am
    ├─ conf
    ├─ graph
    └─ ivector
```

---

### 🔹 MediaPipe Face Landmarker

#### 📥 Download model

Official page:

👉 [https://developers.google.com/mediapipe/solutions/vision/face_landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

Direct download (.task file):

👉 [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

---

#### 📂 Required placement

```
Model/face_landmarker.task
```

⚠️ If you want to **download manually**, please move the **buffalo_l/.onnx** file to the corresponding folder.

---

## ✅ Model Placement Summary

| Model          | Download Method | Location                          |
| -------------- | --------------- | --------------------------------- |
| Whisper        | Auto / Manual   | `C:\Users\<User>\.cache\whisper\` |
| InsightFace    | Auto / Manual   | `C:\Users\<User>\.insightface\`   |
| Vosk           | Manual          | `Model/Vosk/`                     |
| MediaPipe Face | Manual          | `Model/face_landmarker.task`      |

---

## ▶️ How to Run

```bash
python main.py
```

> Make sure your **camera and microphone** are connected and accessible.

---

## 📂 Recommended Project Structure

```
HandsOn-Access/
 ├─ Model/
 │  ├─ Vosk/
 │  ├─ buffalo_l/
 │  └─ face_landmarker.task
 ├─ Keyboard/
 ├─ main.py
 ├─ user_setting.json
 └─ README.md
```

---

## ⚠️ Notes

* This project is **Windows-only** (uses `pywin32`)
* Microphone permission is required
* First run may take time due to model downloads

---

## 🤝 Credits

* OpenAI Whisper
* InsightFace
* MediaPipe
* Vosk Speech Recognition

---

## 📬 Contact · Feedback · Collaboration

HandsOn-Access is not a “finished” project.  
It is an ongoing exploration of accessible human–computer interaction.

The core goal of this project is to support people who **cannot use a keyboard and mouse at the same time**, or who face barriers with traditional input devices, by providing **more natural and flexible ways to interact with computers**.  
At the same time, we recognize that **a single developer’s perspective is always limited**.

### 🌱 Directions We Hope to Explore Together

- 🧍‍♂️ **More diverse gestures and orientations**  
  Different users have different physical abilities, motion ranges, and comfort zones.  
  Future work aims to support a wider variety of **head orientations, facial movements, and subtle posture changes**, instead of relying on a single “standard” gesture.

- 💻 **Cross-platform accessibility**  
  The current implementation focuses on Windows, but the long-term goal is to gradually explore support for **other platforms such as Linux and macOS**.

- ♿ **Real-world usage feedback**  
  Which gestures cause fatigue?  
  Which voice commands feel natural?  
  Which interactions look good in theory but fail in practice?  
  These insights can only come from real users and real scenarios.

- 🔌 **Extensible input and interaction methods**  
  Including, but not limited to:  
  eye tracking, external switches, hybrid inputs, and multi-modal interaction designs.

### 💬 How You Can Participate

Whether you are:

- a user with accessibility needs  
- someone interested in accessibility and inclusive design  
- a developer curious about computer vision, speech recognition, or HCI  
- or simply someone with an idea that starts with “what if…”

You are very welcome to:

- open an Issue  
- share usage experiences  
- propose ideas or improvement plans  
- participate in discussions or contribute code

Even a small piece of feedback can help make this project **more usable, more inclusive, and more human**.

**Accessibility is not a feature.  
It’s a responsibility.**

🚀 Enjoy hands-free computing!
