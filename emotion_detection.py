import cv2
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, QProgressBar, QDialog
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
import sys
import signal
import random
import requests
from io import BytesIO
import sounddevice as sd
import soundfile as sf
import numpy as np
import openai
import wave
import pyaudio
import threading
import queue
import os

# Initialize QApplication at the start
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

# Load and cache the logo
LOGO_URL = "https://u1.padletusercontent.com/uploads/padlet-uploads/772896347/22fe8796a917ff2902adc9e02d271adf/CC05AB1C_F8E7_40D9_9BD4_ACB5498A6C5C.png?token=ovZCJ2DsQTTdlrr926tnqto8AUdkuKZ6x6FAMVy6n7-poSxZhGqP1uYapDZt8Es7IMGYQjFrEytvk7pzxUbF0ZlpwL7yxCdvO6iDm5WlOzfTObiilaDsI_hq4KXpaBL-s4mYciDJyFntmotxKg-aQxUpB_kiRpBK0O_29HZRZtIdrP2NTcWgqBeY5UJJyexQ4R-VkvmElRAotbzpocl68yGO5ttivDC8fzfy0Xs0OUkrm2lP2sSLdeeLhJ1YvscOJQtKKhXOOts-9TuqBJRasA=="

def load_logo():
    try:
        response = requests.get(LOGO_URL)
        image_data = BytesIO(response.content)
        pixmap = QPixmap()
        pixmap.loadFromData(image_data.getvalue())
        
        # Scale logo while preserving aspect ratio and using smooth transformation
        scaled_pixmap = pixmap.scaled(
            200, 200,  # Target size
            Qt.KeepAspectRatio,  # Preserve aspect ratio
            Qt.SmoothTransformation  # Use high-quality scaling
        )
        return scaled_pixmap
    except Exception as e:
        print(f"Error loading logo: {e}")
        return None

# Create a timer for processing events
event_timer = QTimer()
event_timer.timeout.connect(lambda: None)  # Keep event loop active
event_timer.start(100)  # Process events every 100ms

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Constants for thresholds and timing
SCREEN_TIME_THRESHOLD = 1500  # 25 minutes
SCREEN_TIME_COOLDOWN = 15  # 15 seconds between alerts
SADNESS_THRESHOLD = 0.25  # Threshold for sadness detection
POPUP_COOLDOWN = 5  # Cooldown between emotion popups
HISTORY_LENGTH = 10  # Length of prediction history for smoothing
NO_FACE_THRESHOLD = 20  # Time threshold in seconds before resetting timer

# Therapy options and messages
THERAPY_OPTIONS = {
    1: {
        "title": "Microtherapy Session",
        "options": [
            "Let's take a deep breath together.",
            "Focus on something positive in your environment.",
            "Remember a happy memory.",
            "Think about someone who makes you smile."
        ]
    },
    2: {
        "title": "Positive Message",
        "options": [
            "You are stronger than you know!",
            "This feeling is temporary, and better days are ahead.",
            "You've overcome challenges before, and you can do it again.",
            "Your feelings are valid, but they don't define you."
        ]
    },
    3: {
        "title": "Suggestion to Improve",
        "options": [
            "Take a short walk outside.",
            "Call a friend or family member.",
            "Listen to your favorite uplifting music.",
            "Try some light exercise or stretching."
        ]
    }
}

# Screen time tracking variables
screen_time_start = time.time()  # Initialize at start
last_screen_time_alert = 0
showing_screen_time_popup = False

# Variables for popup timing
last_popup_time = 0
showing_popup = False

print("Starting emotion detection... Press 'q' to quit")
print(f"Popup will show when sadness > {SADNESS_THRESHOLD}")

# Add screen time suggestions
SCREEN_TIME_SUGGESTIONS = [
    "Do 20 pushups to get your blood flowing!",
    "Take a 5-minute walk around your space.",
    "Stand up and do 10 jumping jacks.",
    "Go talk to a friend or family member.",
    "Do some quick stretching exercises.",
    "Take a short break and drink some water.",
    "Do 20 squats to energize yourself.",
    "Step outside for some fresh air."
]

class AudioRecorder(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        self.is_recording = True
        
        while self.is_recording:
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save recording
        filename = "temp_recording.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        self.finished.emit(filename)
    
    def stop(self):
        self.is_recording = False

class AudioTherapySession(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Therapy Session")
        self.showMaximized()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Apply the same style as other windows
        central_widget.setStyleSheet("""
            QWidget {
                background: #b6c9c3;
            }
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #E0E7FF,
                    stop: 1 #FFF0F7
                );
                border: none;
                color: #4568DC;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
                border-radius: 25px;
                margin: 10px 50px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #FFF0F7,
                    stop: 1 #E0E7FF
                );
                color: #B06AB3;
            }
            QLabel {
                color: #333333;
                font-size: 16px;
                padding: 20px;
                background: none;
            }
        """)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(50, 30, 50, 30)
        layout.setSpacing(15)
        
        # Add logo
        logo_label = QLabel()
        logo_pixmap = load_logo()
        if logo_pixmap:
            logo_pixmap = logo_pixmap.scaledToWidth(200)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
        
        # Instructions label
        self.status_label = QLabel("Hold to Speak")
        self.status_label.setFont(QFont('Arial', 20))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar for recording
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4568DC;
                border-radius: 5px;
                text-align: center;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #4568DC;
            }
        """)
        layout.addWidget(self.progress)
        
        # Record button
        self.record_button = QPushButton("🎤 Press and hold to speak")
        self.record_button.setFixedWidth(300)
        self.record_button.pressed.connect(self.start_recording)
        self.record_button.released.connect(self.stop_recording)
        layout.addWidget(self.record_button, alignment=Qt.AlignCenter)
        
        # Close button
        close_btn = QPushButton("Close Session")
        close_btn.setFixedWidth(300)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        # Initialize audio recorder
        self.recorder = AudioRecorder()
        self.recorder.finished.connect(self.process_audio)
        
        # Timer for progress bar
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        
        self.center()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
    
    def center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def start_recording(self):
        self.status_label.setText("Recording... Keep speaking")
        self.progress_value = 0
        self.progress.setValue(0)
        self.timer.start(100)  # Update every 100ms
        self.recorder.start()
    
    def stop_recording(self):
        self.recorder.stop()
        self.timer.stop()
        self.status_label.setText("Processing your message...")
        self.progress.setValue(100)
    
    def update_progress(self):
        self.progress_value = min(100, self.progress_value + 2)
        self.progress.setValue(self.progress_value)
    
    def process_audio(self, filename):
        try:
            # Send audio to GPT-4 for processing
            client = openai.OpenAI(
                api_key="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"
            )
            
            with open(filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Get response from GPT-4
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a compassionate therapist having a conversation with someone who might be feeling sad or stressed. Keep responses concise and supportive."},
                    {"role": "user", "content": transcription.text}
                ]
            )
            
            # Convert response to speech
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=response.choices[0].message.content
            )
            
            # Save and play response
            speech_file = "temp_response.mp3"
            speech_response.stream_to_file(speech_file)
            
            # Play the response using system default audio player
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {speech_file}")
            else:  # Other platforms
                os.system(f"start {speech_file}")
            
            self.status_label.setText("Press and hold the button to speak")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error processing audio: {e}")
        
        finally:
            # Clean up temporary files
            try:
                os.remove(filename)
                os.remove("temp_response.mp3")
            except:
                pass

class TherapyMessage(QMainWindow):
    def __init__(self, title, message):
        super().__init__()
        self.setWindowTitle(title)
        self.showMaximized()  # Make window fullscreen
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet(f"""
            QWidget {{
                background: #b6c9c3;
            }}
            QPushButton {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #E0E7FF,
                    stop: 1 #FFF0F7
                );
                border: none;
                color: #4568DC;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
                border-radius: 25px;
                margin: 10px 50px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #FFF0F7,
                    stop: 1 #E0E7FF
                );
                color: #B06AB3;
            }}
            QLabel {{
                color: #333333;
                font-size: 16px;
                padding: 20px;
                background: none;
            }}
        """)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(50, 30, 50, 30)
        layout.setSpacing(15)
        
        # Add logo with larger size
        logo_label = QLabel()
        logo_pixmap = load_logo()
        if logo_pixmap:
            logo_pixmap = logo_pixmap.scaledToWidth(200)  # Make logo bigger
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
            layout.addSpacing(15)  # Reduced spacing after logo
        
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setAlignment(Qt.AlignCenter)
        msg_label.setFont(QFont('Arial', 20))  # Slightly smaller font
        layout.addWidget(msg_label)
        
        layout.addSpacing(20)  # Reduced spacing before button
        
        close_btn = QPushButton("I understand")
        close_btn.setFixedWidth(300)  # Smaller fixed width for button
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        layout.addStretch()  # Add stretch at the bottom
        
        self.center()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
    
    def center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

class ScreenTimeAlert(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⏰ Screen Time Alert!")
        self.showMaximized()  # Make window fullscreen
        
        self.setStyleSheet(f"""
            QWidget {{
                background: #b6c9c3;
                border: 2px solid #a0b5ae;
            }}
            QPushButton {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #E0E7FF,
                    stop: 1 #FFF0F7
                );
                border: none;
                color: #4568DC;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
                border-radius: 25px;
                margin: 10px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #FFF0F7,
                    stop: 1 #E0E7FF
                );
                color: #B06AB3;
            }}
            QLabel {{
                color: #333333;
                background: none;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 30, 50, 30)
        layout.setSpacing(15)
        
        # Add logo with larger size
        logo_label = QLabel()
        logo_pixmap = load_logo()
        if logo_pixmap:
            logo_pixmap = logo_pixmap.scaledToWidth(200)  # Make logo bigger
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
            layout.addSpacing(15)  # Reduced spacing after logo
        
        message = QLabel("⚠️ Screen Time Alert! ⚠️\nTime for a mindful break!")
        message.setFont(QFont('Arial', 28, QFont.Bold))  # Slightly smaller font
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(message)
        
        layout.addSpacing(20)  # Reduced spacing
        
        # Create a container for buttons
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(15)  # Reduced spacing between buttons
        
        advice_button = QPushButton("💡 Give me advice")
        advice_button.setFixedWidth(300)  # Smaller fixed width
        advice_button.clicked.connect(self.show_advice)
        button_layout.addWidget(advice_button, alignment=Qt.AlignCenter)
        
        ignore_button = QPushButton("❌ Ignore")
        ignore_button.setFixedWidth(300)  # Smaller fixed width
        ignore_button.clicked.connect(self.close)
        button_layout.addWidget(ignore_button, alignment=Qt.AlignCenter)
        
        layout.addWidget(button_container)
        layout.addStretch()
        
        self.setLayout(layout)
        self.center()
        
        # Print debug information
        print("\n=== Screen Time Alert Created ===")
        print("Window size:", self.size())
        print("Window flags:", self.windowFlags())
        print("==============================\n")
    
    def center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)
        print(f"Window positioned at: ({x}, {y})")
    
    def show_advice(self):
        suggestion = random.choice(SCREEN_TIME_SUGGESTIONS)
        msg = QMessageBox()
        msg.setWindowTitle("💪 Activity Suggestion")
        msg.setText(suggestion)
        msg.setWindowFlags(Qt.WindowStaysOnTopHint)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: #FF5733;
                font-size: 14px;
                padding: 20px;
            }
            QPushButton {
                background-color: #FF5733;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
        """)
        msg.exec_()
        self.close()
    
    def closeEvent(self, event):
        global showing_screen_time_popup, last_screen_time_alert, screen_time_start
        showing_screen_time_popup = False
        last_screen_time_alert = time.time()
        screen_time_start = time.time()  # Reset the screen time counter
        print("\n=== Screen Time Alert Closed ===")
        print("Cooldown started at:", time.strftime("%H:%M:%S"))
        print("Screen time reset to 0")
        print("Next alert available in:", SCREEN_TIME_COOLDOWN, "seconds")
        print("==============================\n")
        super().closeEvent(event)
    
    def show(self):
        super().show()
        self.raise_()
        self.activateWindow()

def show_screen_time_alert():
    global showing_screen_time_popup, app
    if not showing_screen_time_popup:
        showing_screen_time_popup = True
        print("\n=== Creating Screen Time Alert ===")
        print("Current time:", time.strftime("%H:%M:%S"))
        print("Alert window will be created and shown")
        print("==============================\n")
        
        alert = ScreenTimeAlert()
        
        # Process events before showing window
        app.processEvents()
        
        # Show and activate window
        alert.show()
        alert.raise_()
        alert.activateWindow()
        
        # Process events after showing window
        app.processEvents()
        
        # Force window to stay on top and be active
        alert.setWindowState(alert.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        alert.raise_()
        
        # Keep reference to prevent garbage collection
        app.alert_window = alert
        
        print("\n=== Screen Time Alert Shown ===")
        print("Window state:", alert.windowState())
        print("Window is visible:", alert.isVisible())
        print("Window is active:", alert.isActiveWindow())
        print("==============================\n")

def check_screen_time():
    global last_screen_time_alert, showing_screen_time_popup, screen_time_start
    current_time = time.time()
    elapsed_time = current_time - screen_time_start
    time_since_last_alert = current_time - last_screen_time_alert

    print(f"Screen time - Elapsed: {elapsed_time:.1f}s, Since last alert: {time_since_last_alert:.1f}s, Showing popup: {showing_screen_time_popup}")

    if (elapsed_time >= SCREEN_TIME_THRESHOLD and 
        time_since_last_alert >= SCREEN_TIME_COOLDOWN and 
        not showing_screen_time_popup):
        print("\n=== Screen Time Check ===")
        print(f"Elapsed time: {elapsed_time:.1f}s")
        print(f"Time since last alert: {time_since_last_alert:.1f}s")
        print(f"Threshold met: {elapsed_time >= SCREEN_TIME_THRESHOLD}")
        print(f"Cooldown passed: {time_since_last_alert >= SCREEN_TIME_COOLDOWN}")
        print("Showing alert window...")
        print("==============================\n")
        show_screen_time_alert()

class SupportWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotional Support")
        self.showMaximized()
        
        # Initialize timer
        self.countdown = 20
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # Update every second
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet(f"""
            QWidget {{
                background: #b6c9c3;
            }}
            QPushButton {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #E0E7FF,
                    stop: 1 #FFF0F7
                );
                border: none;
                color: #4568DC;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
                border-radius: 25px;
                margin: 10px 50px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #FFF0F7,
                    stop: 1 #E0E7FF
                );
                color: #B06AB3;
            }}
            QLabel {{
                color: #333333;
                font-size: 16px;
                padding: 15px;
                background: none;
            }}
        """)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(50, 30, 50, 30)
        layout.setSpacing(15)
        
        # Add logo with larger size
        logo_label = QLabel()
        logo_pixmap = load_logo()
        if logo_pixmap:
            logo_pixmap = logo_pixmap.scaledToWidth(200)  # Make logo bigger
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
            layout.addSpacing(15)  # Reduced spacing after logo
        
        # Add timer label
        self.timer_label = QLabel(f"Time remaining: {self.countdown}s")
        self.timer_label.setFont(QFont('Arial', 16))
        self.timer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.timer_label)
        
        title1 = QLabel("We notice you're experiencing some emotions.")
        title1.setFont(QFont('Arial', 28, QFont.Bold))  # Slightly smaller font
        title1.setAlignment(Qt.AlignCenter)
        layout.addWidget(title1)
        
        title2 = QLabel("Let's work through this together.")
        title2.setFont(QFont('Arial', 20))  # Slightly smaller font
        title2.setAlignment(Qt.AlignCenter)
        layout.addWidget(title2)
        
        layout.addSpacing(15)  # Reduced spacing
        
        # Create a container for buttons
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(15)  # Reduced spacing between buttons
        
        for option_num, option in THERAPY_OPTIONS.items():
            btn = QPushButton(option["title"])
            btn.setFixedWidth(400)  # Slightly smaller fixed width
            btn.clicked.connect(lambda checked, num=option_num: self.handle_option(num))
            button_layout.addWidget(btn, alignment=Qt.AlignCenter)
        
        ignore_btn = QPushButton("Ignore")
        ignore_btn.setFixedWidth(400)  # Slightly smaller fixed width
        ignore_btn.clicked.connect(self.close)
        button_layout.addWidget(ignore_btn, alignment=Qt.AlignCenter)
        
        layout.addWidget(button_container)
        layout.addStretch()
        
        self.center()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
    
    def update_timer(self):
        self.countdown -= 1
        self.timer_label.setText(f"Time remaining: {self.countdown}s")
        
        if self.countdown <= 0:
            self.timer.stop()
            self.handle_anxiety_intervention()
    
    def handle_anxiety_intervention(self):
        """Handle anxiety intervention when no user interaction is detected."""
        self.close()
        
        try:
            # Initialize OpenAI client
            client = openai.OpenAI(
                api_key="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"
            )
            
            # Generate calming message using GPT-4
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a calming presence helping someone with anxiety. Provide a short, gentle breathing exercise and reassurance. Keep it under 4 sentences."},
                    {"role": "user", "content": "I'm feeling anxious, can you help me with a quick breathing exercise?"}
                ]
            )
            
            calming_message = response.choices[0].message.content
            
            # Convert to speech
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=calming_message
            )
            
            # Save and play response
            speech_file = "calming_message.mp3"
            speech_response.stream_to_file(speech_file)
            
            # Play the audio
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {speech_file}")
            else:  # Other platforms
                os.system(f"start {speech_file}")
            
        except Exception as e:
            print(f"Error generating calming message: {e}")
            calming_message = (
                "Let's take a moment to breathe together:\n\n"
                "1. Take a deep breath in through your nose for 4 counts\n"
                "2. Hold your breath for 4 counts\n"
                "3. Exhale slowly through your mouth for 6 counts\n"
                "4. Repeat this cycle 4 times\n\n"
                "Remember: This feeling is temporary. You are safe and supported."
            )
        
        # Create and show the calming window
        calming_window = QDialog()
        calming_window.setWindowTitle("Calming Intervention")
        calming_window.setStyleSheet("background-color: #f0f5ff;")
        
        layout = QVBoxLayout()
        
        message_label = QLabel(calming_message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(
            "font-size: 14px; color: #2c3e50; margin: 20px;"
        )
        
        layout.addWidget(message_label)
        
        close_button = QPushButton("I feel better now")
        close_button.setStyleSheet(
            "background-color: #3498db; color: white; padding: 10px;"
            "border: none; border-radius: 5px; font-size: 12px;"
        )
        
        def start_microtherapy():
            calming_window.close()
            self.therapy_window = AudioTherapySession()
            self.therapy_window.show()
        
        close_button.clicked.connect(start_microtherapy)
        
        layout.addWidget(close_button, alignment=Qt.AlignCenter)
        
        calming_window.setLayout(layout)
        calming_window.resize(400, 300)
        calming_window.exec_()
    
    def center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def handle_option(self, option_num):
        self.timer.stop()  # Stop the timer when an option is selected
        self.close()
        if option_num in THERAPY_OPTIONS:
            option = THERAPY_OPTIONS[option_num]
            if option["title"] == "Microtherapy Session":
                self.therapy_window = AudioTherapySession()
                self.therapy_window.show()
            else:
                try:
                    # Initialize OpenAI client
                    client = openai.OpenAI(
                        api_key="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"
                    )
                    
                    # Prepare prompt based on option type
                    if option["title"] == "Positive Message":
                        system_prompt = "You are an empathetic and supportive friend. Generate a positive, uplifting message that is 2-3 lines long. Make it personal and encouraging."
                        user_prompt = "I'm feeling a bit down. Can you give me a positive message to help me feel better?"
                    else:  # Suggestion to Improve
                        system_prompt = "You are a supportive wellness coach. Provide a practical, actionable suggestion to improve mood and wellbeing. Make it 2-3 lines long and specific."
                        user_prompt = "I'm not feeling my best. What's a practical suggestion to help improve my mood?"
                    
                    # Generate message using GPT-4
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    
                    message = response.choices[0].message.content
                    
                except Exception as e:
                    print(f"Error generating message: {e}")
                    # Fallback to default messages if API call fails
                    message = np.random.choice(option["options"])
                
                self.therapy_window = TherapyMessage(option["title"], message)
                self.therapy_window.show()
    
    def closeEvent(self, event):
        self.timer.stop()  # Ensure timer is stopped when window is closed
        super().closeEvent(event)

def show_support_window():
    global showing_popup, last_popup_time
    if not showing_popup:
        showing_popup = True
        last_popup_time = time.time()
        window = SupportWindow()
        window.show()
        
        def on_close():
            global showing_popup
            showing_popup = False
        
        window.destroyed.connect(on_close)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load OpenCV's face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For smoothing predictions
prediction_history = []

# Variables for popup timing
last_popup_time = 0
showing_popup = False
running = True  # Added flag for clean exit

def cleanup():
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    app.quit()

def signal_handler(sig, frame):
    global running
    print("\nCtrl+C detected - cleaning up...")
    running = False

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    global screen_time_start, last_screen_time_alert, showing_screen_time_popup
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    screen_time_start = time.time()
    prediction_history = []
    HISTORY_LENGTH = 10
    SADNESS_THRESHOLD = 0.25
    
    # Add variable to track when face was last seen
    last_face_detected = time.time()
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
            
        # Process events to keep GUI responsive
        app.processEvents()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        current_time = time.time()
        
        if len(faces) == 0:
            # Calculate how long no face has been detected
            time_without_face = current_time - last_face_detected
            if time_without_face >= NO_FACE_THRESHOLD:
                screen_time_start = current_time
                print(f"No face detected for {time_without_face:.1f}s - Timer reset to 0")
        else:
            # Update last face detection time
            last_face_detected = current_time
            # Check screen time only when face is detected
            check_screen_time()
        
        for (x, y, w, h) in faces:
            try:
                # Generate random predictions with more variation
                emotion_pred = np.zeros((1, 7))
                dominant_emotion = np.random.randint(0, 7)
                emotion_pred[0][dominant_emotion] = np.random.uniform(0.4, 0.6)
                
                remaining_prob = 1.0 - emotion_pred[0][dominant_emotion]
                other_probs = np.random.dirichlet(np.ones(6)) * remaining_prob
                
                j = 0
                for i in range(7):
                    if i != dominant_emotion:
                        emotion_pred[0][i] = other_probs[j]
                        j += 1
                
                prediction_history.append(emotion_pred[0])
                if len(prediction_history) > HISTORY_LENGTH:
                    prediction_history.pop(0)
                
                avg_prediction = np.mean(prediction_history, axis=0)
                emotion_idx = np.argmax(avg_prediction)
                emotion = emotion_labels[emotion_idx]
                confidence = float(avg_prediction[emotion_idx])
                
                # Show popup when sadness is detected naturally
                sad_index = emotion_labels.index('Sad')
                current_time = time.time()
                sad_value = avg_prediction[sad_index]
                
                # Print debug info about sadness
                print(f"Sadness: {sad_value:.2f}, Time since last popup: {current_time - last_popup_time:.1f}s, Showing popup: {showing_popup}")
                
                if (sad_value >= SADNESS_THRESHOLD and
                    current_time - last_popup_time >= POPUP_COOLDOWN and
                    not showing_popup):
                    print("Showing support window!")
                    show_support_window()
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display emotion probabilities with improved formatting
                for i, (emotion, prob) in enumerate(zip(emotion_labels, avg_prediction)):
                    if i == emotion_idx:
                        color = (0, 255, 0)  # Green for dominant emotion
                    elif i == sad_index:
                        color = (0, 0, 255)  # Red for sadness
                    else:
                        color = (200, 200, 200)  # Gray for others
                    
                    text = f"{emotion}: {prob:.2f}"
                    cv2.putText(image, text, (10, 30 + i * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw sadness threshold line
                threshold_y = int(30 + sad_index * 30)
                cv2.line(image, (150, threshold_y), (150 + int(SADNESS_THRESHOLD * 200), threshold_y),
                        (0, 0, 255), 2)  # Red line at threshold

                # Add stopwatch display at the bottom
                elapsed_time = int(current_time - screen_time_start)
                minutes = elapsed_time // 60
                seconds = elapsed_time % 60
                timer_text = f"Time: {minutes:02d}:{seconds:02d}"
                cv2.putText(image, timer_text, (image.shape[1] - 200, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Add text showing how to quit
                cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Display frame
        cv2.imshow('Emotion Detection', image)
        
        # Process events to keep GUI responsive
        app.processEvents()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        print("Starting emotion detection... Press 'q' to quit")
        print(f"Popup will show when sadness > {SADNESS_THRESHOLD}")
        main()
    finally:
        print("Cleaning up...")
        app.quit() 