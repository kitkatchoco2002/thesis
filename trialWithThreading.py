import time
import threading
from picamera2 import Picamera2
from ultralytics import YOLO
import RPi.GPIO as GPIO

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

# Flag to control interval mode
in_interval_mode = True


def detect_objects():
    """Continuously detects objects and updates the global variable."""
    global detected_objects
    while True:
        # Capture a frame
        frame = picam2.capture_array()
        start_time = time.time()
        results = model(frame, imgsz=640)
        end_time = time.time()
        inference_time = end_time - start_time

        # Update detected objects
        with detection_lock:
            detected_objects = results[0].boxes.cls.tolist()

        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"Detected Objects: {[model.names[int(cls)] for cls in detected_objects]}")

        time.sleep(0.5)  # Small delay to avoid excessive processing


def bird_detected_response():
    """Activates deterrents when a bird or flock is detected."""
    global in_interval_mode
    in_interval_mode = False  # Exit interval mode immediately

    print("Bird detected! Activating deterrents immediately.")

    # Activate deterrents
    GPIO.output(27, GPIO.HIGH)  # Turn on laser
    GPIO.output(17, GPIO.HIGH)  # Turn on arms
    head_pwm.ChangeDutyCycle(0)  # Stop head rotation

    time.sleep(20)  # Deterrents active for 20 seconds

    # Deactivate deterrents
    GPIO.output(27, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)
    in_interval_mode = True  # Return to interval mode


def interval_mode_cycle():
    """Runs the interval mode cycle in a loop."""
    global in_interval_mode
    while True:
        if in_interval_mode:
            print("Interval Mode: Head Rotating...")
            head_pwm.ChangeDutyCycle(90)  # Start head rotation
            time.sleep(HEAD_ROTATE_TIME)

            print("Interval Mode: Stopping head, activating deterrents.")
            head_pwm.ChangeDutyCycle(0)  # Stop head rotation
            GPIO.output(27, GPIO.HIGH)  # Turn on laser
            GPIO.output(17, GPIO.HIGH)  # Turn on arms
            time.sleep(DETERRENT_TIME)

            print("Interval Mode: Turning off deterrents, restarting cycle.")
            GPIO.output(27, GPIO.LOW)
            GPIO.output(17, GPIO.LOW)

        time.sleep(1)  # Small delay to prevent excessive CPU usage


# Start detection thread
detection_thread = threading.Thread(target=detect_objects, daemon=True)
detection_thread.start()

# Start interval mode in the main thread
try:
    while True:
        with detection_lock:
            detects_bird_or_flock = any(cls in birds_and_flock for cls in detected_objects)

        if detects_bird_or_flock:
            bird_detected_response()

        time.sleep(0.5)  # Small delay to balance CPU usage

except KeyboardInterrupt:
    print("Program interrupted by the user.")

finally:
    picam2.close()
    head_pwm.stop()
    GPIO.cleanup()
    print("Resources cleaned up!")
