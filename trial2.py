import time
import threading
import pygame
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load the deterrent sound file
SOUND_FILE = "deterrent_sound.mp3"  # Ensure this file exists in the same directory
try:
    pygame.mixer.music.load(SOUND_FILE)
except pygame.error:
    print(f"Error loading sound file: {SOUND_FILE}")

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup(17, GPIO.OUT)  # Arm 1 & Arm 2 
GPIO.setup(27, GPIO.OUT)  # LASER control
GPIO.setup(23, GPIO.OUT)  # HEAD control

# Initialize PWM for head speed control
head_pwm = GPIO.PWM(23, 1000)  # Head control (Pin 23), frequency 100 Hz
head_pwm.start(0)  # Start with 0% duty cycle (stopped)

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO model
model = YOLO("best.pt")
print(model.names)

# Class IDs for object detection
birds_and_flock = [1, 2]  # Bird and flock
humans_and_fake_birds = [0, 3]  # Fake bird and human

# Shared detection variable
detected_objects = []
detection_lock = threading.Lock()

# Interval mode cycle durations
HEAD_ROTATE_TIME = 3  # 3 seconds
DETERRENT_TIME = 5  # 5 seconds

# Global Flags
in_interval_mode = True
bird_response_running = False  # Prevent multiple deterrent threads


def detect_objects():
    """Continuously detects objects and updates the global variable."""
    global detected_objects
    while True:
        frame = picam2.capture_array()
        start_time = time.time()
        results = model(frame, imgsz=640)
        end_time = time.time()
        inference_time = end_time - start_time

        # Update detected objects safely
        with detection_lock:
            detected_objects = results[0].boxes.cls.tolist()

        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"Detected Objects: {[model.names[int(cls)] for cls in detected_objects]}")

        time.sleep(0.5)  # Small delay to avoid excessive processing


def play_deterrent_sound():
    """Plays the deterrent sound."""
    pygame.mixer.music.play()


def bird_detected_response():
    """Activates deterrents when a bird is detected and stops when birds leave."""
    global in_interval_mode, bird_response_running
    with detection_lock:
        bird_response_running = True  # Mark deterrent as active
        in_interval_mode = False  # Stop interval mode

    print("Bird detected! Activating deterrents.")

    # Activate deterrents
    GPIO.output(27, GPIO.HIGH)  # Turn on laser
    GPIO.output(17, GPIO.HIGH)  # Turn on arms
    play_deterrent_sound()  # Play the sound 
    head_pwm.ChangeDutyCycle(0)  # Stop head rotation
    

    # # Keep deterrents ON while birds are present
    # time_elapsed = 0
    # while time_elapsed < 20:  # Max 20 seconds deterrent time
    #     with detection_lock:
    #         if not any(cls in birds_and_flock for cls in detected_objects):
    #             print("Birds left, stopping deterrents early.")
    #             break  # Exit early if birds are gone
    #         else:
    #             print("Still detecting...")

    time.sleep(5)
        # time_elapsed += 1

    # Turn off deterrents
    GPIO.output(27, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)
    print("Deterrents turned off. Returning to interval mode.")

    with detection_lock:
        bird_response_running = False  # Allow future deterrent activations
        in_interval_mode = True  # Resume interval mode


def interval_mode_cycle():
    """Runs the interval mode cycle indefinitely, but pauses when birds are detected."""
    while True:
        if in_interval_mode:
            print("Interval Mode: Rotating Head.")
            head_pwm.ChangeDutyCycle(90)  # Rotate head
            time.sleep(HEAD_ROTATE_TIME)

            print("Interval Mode: Stopping head, activating deterrents.")
            head_pwm.ChangeDutyCycle(0)  # Stop head rotation
            GPIO.output(27, GPIO.HIGH)  # Turn on laser
            GPIO.output(17, GPIO.HIGH)  # Turn on arms
            play_deterrent_sound()  # Play the sound 
            time.sleep(DETERRENT_TIME)

            print("Interval Mode: Turning off deterrents.")
            GPIO.output(27, GPIO.LOW)
            GPIO.output(17, GPIO.LOW)

        time.sleep(1)  # Small delay to prevent CPU overload


# Start detection thread
detection_thread = threading.Thread(target=detect_objects, daemon=True)
detection_thread.start()

# Start interval mode thread (ALWAYS RUNNING)
interval_thread = threading.Thread(target=interval_mode_cycle, daemon=True)
interval_thread.start()

# Main control loop
try:
    while True:
        with detection_lock:
            if any(cls in birds_and_flock for cls in detected_objects):
                if not bird_response_running:  # Prevent multiple deterrent activations
                    threading.Thread(target=bird_detected_response, daemon=True).start()

        time.sleep(0.5)  # Small delay to balance CPU usage

except KeyboardInterrupt:
    print("Program interrupted by the user.")

finally:
    picam2.close()
    head_pwm.stop()
    GPIO.cleanup()
    print("Resources cleaned up!")