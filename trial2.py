"""
this code has the led turn on if a bird is detected thus only turbued n when it is in bird active response mode
Bird Deterrent System using Raspberry Pi
This script implements a bird deterrent system using computer vision and various deterrent mechanisms.
"""

# Standard library imports
import time
import threading

# Third-party imports
import pygame
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO

# ============= Configuration Constants =============
# GPIO Pin Configuration
PIN_ARM = 17      # Arm 1 & Arm 2 control
PIN_LASER = 27    # LASER control
PIN_HEAD = 23     # HEAD control
PIN_LED = 22      # LED indicator for bird detection
PIN_ERROR_LED = 24  # LED indicator for system errors

# Audio Configuration
SOUND_FILE = "deterrent_sound.mp3"

# Timing Configuration
HEAD_ROTATE_TIME = 3    # Head rotation duration in seconds
DETERRENT_TIME = 5      # Deterrent activation duration in seconds

# Object Detection Classes
BIRDS_AND_FLOCK = [1, 2]        # Bird and flock class IDs
HUMANS_AND_FAKE_BIRDS = [0, 3]  # Fake bird and human class IDs

# ============= Hardware Initialization =============
# Initialize audio system
pygame.mixer.init()
try:
    pygame.mixer.music.load(SOUND_FILE)
except pygame.error:
    print(f"Error loading sound file: {SOUND_FILE}")

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_ARM, GPIO.OUT)
GPIO.setup(PIN_LASER, GPIO.OUT)
GPIO.setup(PIN_HEAD, GPIO.OUT)
GPIO.setup(PIN_LED, GPIO.OUT)  # Setup LED pin as output
GPIO.setup(PIN_ERROR_LED, GPIO.OUT)  # Setup error LED pin as output

# Initialize PWM for head control
head_pwm = GPIO.PWM(PIN_HEAD, 1000)
head_pwm.start(0)

# ============= Camera and Model Setup =============
# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLO model
model = YOLO("best.pt")
print(model.names)

# ============= Global State Variables =============
detected_objects = []
detection_lock = threading.Lock()
in_interval_mode = True
bird_response_running = False

# ============= Core Functions =============
def detect_objects():
    """
    Continuously detects objects using the camera and YOLO model.
    Updates the global detected_objects list with current detections.
    """
    global detected_objects
    while True:
        frame = picam2.capture_array()
        start_time = time.time()
        results = model(frame, imgsz=640)
        end_time = time.time()
        inference_time = end_time - start_time

        with detection_lock:
            detected_objects = results[0].boxes.cls.tolist()

        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"Detected Objects: {[model.names[int(cls)] for cls in detected_objects]}")
        time.sleep(0.5)

def play_deterrent_sound():
    """Plays the deterrent sound using pygame mixer."""
    pygame.mixer.music.play()

def activate_deterrents():
    """Activates all deterrent mechanisms."""
    GPIO.output(PIN_LASER, GPIO.HIGH)
    GPIO.output(PIN_ARM, GPIO.HIGH)
    play_deterrent_sound()
    head_pwm.ChangeDutyCycle(0)

def deactivate_deterrents():
    """Deactivates all deterrent mechanisms."""
    GPIO.output(PIN_LASER, GPIO.LOW)
    GPIO.output(PIN_ARM, GPIO.LOW)

def bird_detected_response():
    """
    Handles the response when birds are detected.
    Activates deterrents and manages the response cycle.
    """
    global in_interval_mode, bird_response_running
    
    with detection_lock:
        bird_response_running = True
        in_interval_mode = False

    print("Bird detected! Activating deterrents.")
    GPIO.output(PIN_LED, GPIO.HIGH)  # Turn on LED when bird is detected
    activate_deterrents()
    time.sleep(DETERRENT_TIME)
    deactivate_deterrents()
    GPIO.output(PIN_LED, GPIO.LOW)  # Turn off LED after deterrent cycle
    print("Deterrents turned off. Returning to interval mode.")

    with detection_lock:
        bird_response_running = False
        in_interval_mode = True

def interval_mode_cycle():
    """
    Manages the interval mode cycle of the deterrent system.
    Alternates between head rotation and deterrent activation.
    """
    while True:
        if in_interval_mode:
            print("Interval Mode: Rotating Head.")
            head_pwm.ChangeDutyCycle(90)
            time.sleep(HEAD_ROTATE_TIME)

            print("Interval Mode: Stopping head, activating deterrents.")
            head_pwm.ChangeDutyCycle(0)
            activate_deterrents()
            time.sleep(DETERRENT_TIME)

            print("Interval Mode: Turning off deterrents.")
            deactivate_deterrents()

        time.sleep(1)

# ============= Main Program =============
def main():
    """Main program entry point."""
    # Start detection thread
    detection_thread = threading.Thread(target=detect_objects, daemon=True)
    detection_thread.start()

    # Start interval mode thread
    interval_thread = threading.Thread(target=interval_mode_cycle, daemon=True)
    interval_thread.start()

    try:
        while True:
            with detection_lock:
                if any(cls in BIRDS_AND_FLOCK for cls in detected_objects):
                    if not bird_response_running:
                        threading.Thread(target=bird_detected_response, daemon=True).start()
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Program interrupted by the user.")
    finally:
        cleanup()

def cleanup():
    """Cleanup resources before program exit."""
    picam2.close()
    head_pwm.stop()
    GPIO.cleanup()
    print("Resources cleaned up!")

if __name__ == "__main__":
    main()