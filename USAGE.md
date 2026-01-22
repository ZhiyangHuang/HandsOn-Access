# HandsOn-Access Usage Guide

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZhiyangHuang/HandsOn-Access.git
   cd HandsOn-Access
   ```

2. **Install Python 3.6+** (if not already installed)
   - On Windows, download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

3. **Install dependencies** (when needed for full functionality):
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Basic GUI
To launch the main GUI:
```bash
python main.py
```

### Individual Modules

#### Train Face Recognition
```python
from Initialization.face_training import train_face_model
train_face_model()
```

#### Test Facial Expressions
```python
from Initialization.face_training import test_expressions
test_expressions()
```

#### Keyboard Control
```python
from Keyboard.keyboard_control import KeyboardController
kb = KeyboardController()
kb.press_key('A')
kb.type_text("Hello World")
```

#### Main Accessibility Controller
```python
from Main.integration import AccessibilityController
controller = AccessibilityController()
controller.enable_face_control()
controller.enable_voice_control()
controller.start()
```

#### Tutorial Mode (Safe Testing)
```python
from Test.tutorials import safe_test_mode, run_tutorial
safe_test_mode()  # No device commands will be executed
run_tutorial("Getting Started")
```

## Project Structure

- **Initialization/**: Face recognition training and expression testing
- **Keyboard/**: Keyboard control (Windows-focused, extendable)
- **Main/**: Integrates face and voice control
- **Test/**: Tutorials and safe testing environment
- **Model/**: AI models and training files storage
- **main.py**: Main GUI application

## Features

### Current Implementation
- ✓ Project structure and module organization
- ✓ GUI interface for all features
- ✓ Module placeholders for future development
- ✓ Safe testing mode
- ✓ Extensible architecture

### Planned Features
- Face recognition using InsightFace
- Facial gesture detection using MediaPipe
- Voice recognition using Whisper & VOSK
- Hands-free mouse control
- Hand and body gesture support
- Multi-platform support (Linux, macOS)

## Platform Support

- **Primary**: Windows
- **Future**: Linux, macOS

## License

Non-commercial use only. See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing structure
- New features include documentation
- Testing is done before submitting

## Support

For issues and questions, please use the GitHub Issues page.
