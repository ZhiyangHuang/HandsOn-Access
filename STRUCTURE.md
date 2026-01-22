# Project Structure Documentation

## Overview
HandsOn-Access is a free, open-source accessibility tool for Windows, focused on face and voice recognition, with future support for hand and body gestures and other platforms.

## Directory Structure

### Initialization/
**Purpose**: Trains face recognition models and tests facial expressions.

**Files**:
- `__init__.py`: Module initialization
- `face_training.py`: Contains functions for training face recognition models using InsightFace and testing facial expressions using MediaPipe

**Key Functions**:
- `train_face_model()`: Train face recognition models
- `test_expressions()`: Test facial expression detection

### Keyboard/
**Purpose**: Keyboard control functionality (Windows only, extendable to other platforms).

**Files**:
- `__init__.py`: Module initialization
- `keyboard_control.py`: Implements keyboard control functionality

**Key Classes**:
- `KeyboardController`: Handles keyboard simulation and control
  - `press_key(key)`: Simulate pressing a key
  - `type_text(text)`: Type text using keyboard simulation

**Platform Support**: Currently optimized for Windows, designed to be extendable to other platforms.

### Main/
**Purpose**: Integrates face and voice control features.

**Files**:
- `__init__.py`: Module initialization
- `integration.py`: Main integration module combining face and voice control

**Key Classes**:
- `AccessibilityController`: Main controller integrating all accessibility features
  - `enable_face_control()`: Enable face recognition and gesture control
  - `enable_voice_control()`: Enable voice recognition and control
  - `start()`: Start the accessibility controller

**Technologies Used**:
- InsightFace: Face recognition
- MediaPipe: Facial gesture detection
- Whisper & VOSK: Voice recognition and typing

### Test/
**Purpose**: Provides tutorials and safe testing environment without sending actual device commands.

**Files**:
- `__init__.py`: Module initialization
- `tutorials.py`: Tutorial system and safe testing mode

**Key Functions**:
- `run_tutorial(tutorial_name)`: Run a specific tutorial
- `safe_test_mode()`: Enter safe testing mode where no device commands are executed

**Safety**: This module is designed to help users learn and test the system without affecting their actual device operations.

### Model/
**Purpose**: Stores AI models and training files.

**Contents**:
- AI models for face recognition (InsightFace)
- Facial gesture detection models (MediaPipe)
- Voice recognition models (Whisper & VOSK)
- User-specific training data and configurations

**Note**: Models are not included in the repository. Download or train models separately.

### main.py
**Purpose**: Main GUI application integrating all features.

**Description**: 
- Provides a user-friendly GUI built with Tkinter
- Integrates all modules (Initialization, Keyboard, Main, Test)
- Allows users to:
  - Enable/disable face control
  - Enable/disable voice control
  - Train face recognition models
  - Test facial expressions
  - Access tutorial mode

**Usage**:
```bash
python main.py
```

## Technology Stack

- **Face Recognition**: InsightFace
- **Facial Gesture Detection**: MediaPipe
- **Voice Recognition**: Whisper & VOSK
- **GUI Framework**: Tkinter
- **Platform**: Windows (with extensibility for other platforms)

## License

Non-commercial use only. See LICENSE file for details.

## Future Development

- Hand and body gesture support
- Multi-platform support (macOS, Linux)
- Enhanced voice commands
- Customizable gesture mappings
