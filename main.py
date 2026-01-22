"""
HandsOn-Access Main GUI
Integrates all features including face recognition, voice control, and keyboard control.
"""

import tkinter as tk
from tkinter import ttk, messagebox


class HandsOnAccessGUI:
    """
    Main GUI for HandsOn-Access accessibility tool.
    Integrates face recognition, voice control, and keyboard control features.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("HandsOn-Access - Accessibility Tool")
        self.root.geometry("600x400")
        
        # Status variables
        self.face_control_active = False
        self.voice_control_active = False
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create and layout GUI widgets."""
        # Title
        title_label = ttk.Label(
            self.root, 
            text="HandsOn-Access", 
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = ttk.Label(
            self.root,
            text="Free, open-source accessibility tool for Windows",
            font=("Arial", 10)
        )
        subtitle_label.pack(pady=5)
        
        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text="Control Features", padding=20)
        control_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Face Control
        face_frame = ttk.Frame(control_frame)
        face_frame.pack(fill="x", pady=10)
        
        ttk.Label(face_frame, text="Face Control:").pack(side="left", padx=5)
        self.face_button = ttk.Button(
            face_frame,
            text="Enable",
            command=self.toggle_face_control
        )
        self.face_button.pack(side="left", padx=5)
        self.face_status = ttk.Label(face_frame, text="[Inactive]", foreground="red")
        self.face_status.pack(side="left", padx=5)
        
        # Voice Control
        voice_frame = ttk.Frame(control_frame)
        voice_frame.pack(fill="x", pady=10)
        
        ttk.Label(voice_frame, text="Voice Control:").pack(side="left", padx=5)
        self.voice_button = ttk.Button(
            voice_frame,
            text="Enable",
            command=self.toggle_voice_control
        )
        self.voice_button.pack(side="left", padx=5)
        self.voice_status = ttk.Label(voice_frame, text="[Inactive]", foreground="red")
        self.voice_status.pack(side="left", padx=5)
        
        # Additional Features
        features_frame = ttk.LabelFrame(self.root, text="Additional Features", padding=10)
        features_frame.pack(pady=10, padx=20, fill="x")
        
        ttk.Button(
            features_frame,
            text="Train Face Recognition",
            command=self.train_face_model
        ).pack(side="left", padx=5)
        
        ttk.Button(
            features_frame,
            text="Test Expressions",
            command=self.test_expressions
        ).pack(side="left", padx=5)
        
        ttk.Button(
            features_frame,
            text="Tutorial Mode",
            command=self.open_tutorial
        ).pack(side="left", padx=5)
        
        # Status Bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_face_control(self):
        """Toggle face control on/off."""
        self.face_control_active = not self.face_control_active
        
        if self.face_control_active:
            self.face_button.config(text="Disable")
            self.face_status.config(text="[Active]", foreground="green")
            self.status_bar.config(text="Face control enabled")
            # TODO: Initialize face control
        else:
            self.face_button.config(text="Enable")
            self.face_status.config(text="[Inactive]", foreground="red")
            self.status_bar.config(text="Face control disabled")
            # TODO: Stop face control
    
    def toggle_voice_control(self):
        """Toggle voice control on/off."""
        self.voice_control_active = not self.voice_control_active
        
        if self.voice_control_active:
            self.voice_button.config(text="Disable")
            self.voice_status.config(text="[Active]", foreground="green")
            self.status_bar.config(text="Voice control enabled")
            # TODO: Initialize voice control
        else:
            self.voice_button.config(text="Enable")
            self.voice_status.config(text="[Inactive]", foreground="red")
            self.status_bar.config(text="Voice control disabled")
            # TODO: Stop voice control
    
    def train_face_model(self):
        """Open face model training."""
        messagebox.showinfo(
            "Face Training",
            "Face recognition training module will be launched.\n\n"
            "This feature uses InsightFace for face recognition training."
        )
        self.status_bar.config(text="Training face recognition model...")
        # TODO: Launch Initialization.face_training.train_face_model()
    
    def test_expressions(self):
        """Test facial expressions."""
        messagebox.showinfo(
            "Expression Testing",
            "Facial expression testing will be launched.\n\n"
            "This feature uses MediaPipe for facial gesture detection."
        )
        self.status_bar.config(text="Testing facial expressions...")
        # TODO: Launch Initialization.face_training.test_expressions()
    
    def open_tutorial(self):
        """Open tutorial mode."""
        messagebox.showinfo(
            "Tutorial Mode",
            "Tutorial mode provides safe testing without sending device commands.\n\n"
            "Perfect for learning how to use HandsOn-Access."
        )
        self.status_bar.config(text="Opening tutorial mode...")
        # TODO: Launch Test.tutorials.safe_test_mode()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = HandsOnAccessGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
