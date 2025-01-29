import time
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

# Timers for state switching
no_bird_timer = time.time()  # Track time since last bird detection
time_interval_timer = time.time()  # Track timed activation interval
in_time_interval_mode = True  # Track if system is in time interval mode

#TODO: SPEAKER TO BE ADDED
def bird_detected_response():
    print("Bird or Flock detected: Activating deterrents.")
    
    global in_time_interval_mode, no_bird_timer
    in_time_interval_mode = False  # Reset state
    no_bird_timer = time.time()  # Reset no bird timer

    GPIO.output(27, GPIO.HIGH)  # Turn on laser
    GPIO.output(17, GPIO.HIGH)  # Turn on arms
    head_pwm.ChangeDutyCycle(0)  # Stop head rotation
    
    time.sleep(20)  # Keep deterrents active for 20 seconds

    GPIO.output(27, GPIO.LOW)  # Turn off laser
    GPIO.output(17, GPIO.LOW)  # Turn off arms
    in_time_interval_mode = True

def no_bird_detected_response():
    elapsed_no_bird_time = time.time() - no_bird_timer

    if elapsed_no_bird_time >= 10:  # 2 minutes of no bird detected (current number is to be changed for testing purposes only)
        in_time_interval_mode = True
        print("Switching to time-interval mode.")

    if not in_time_interval_mode:
        print("No bird detected: Normal mode, rotating head.")
        # GPIO.output(27, GPIO.LOW)  # Turn off laser
        # GPIO.output(17, GPIO.LOW)  # Turn off Arm 1
        # head_pwm.ChangeDutyCycle(50)  # Rotate head at 90% speed

try:
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Measure inference start time
        start_time = time.time()

        # Run object detection on the frame
        results = model(frame, imgsz=640)

        # Measure inference end time
        end_time = time.time()
        inference_time = end_time - start_time

        # Extract detected object classes
        detected_objects = results[0].boxes.cls.tolist()

        # Determine the state based on detection
        detects_bird_or_flock = any(cls in birds_and_flock for cls in detected_objects)
        detects_human_or_fake_bird = any(cls in humans_and_fake_birds for cls in detected_objects)

        # Print inference time and detected objects
        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"Detected Objects: {[model.names[int(cls)] for cls in detected_objects]}")

        # --- Normal State Logic ---
        # if bird is detected then turn on laser and arms for 20 seconds
        # if bird is detected then turn off the head rotation for 20 seconds
        if detects_bird_or_flock:
           bird_detected_response()
            

        elif detects_human_or_fake_bird or not detected_objects:
            no_bird_detected_response()

        # --- Time Interval Activation Mode ---
        if in_time_interval_mode:
            elapsed_time_interval = time.time() - time_interval_timer

            if elapsed_time_interval >= 300:  # Every 5 minutes (300 seconds)
                print("Time Interval Mode: Activating arms and laser for 20 seconds.")

                # Stop head rotation
                head_pwm.ChangeDutyCycle(0)

                # Activate arms and laser
                GPIO.output(23, GPIO.LOW)  # Turn on laser
                GPIO.output(27, GPIO.HIGH)  # Turn on laser
                GPIO.output(17, GPIO.HIGH)  # Turn on arms
                time.sleep(20)  # Run for 20 seconds
                head_pwm.ChangeDutyCycle(90)
                GPIO.output(23, GPIO.HIGH)  # Turn on laser
                GPIO.output(17, GPIO.LOW)  # Turn off arms
                GPIO.output(27, GPIO.LOW)  # Turn off laser

                # # Resume head rotation
                # head_pwm.ChangeDutyCycle(90)

                # Reset time interval timer
                time_interval_timer = time.time()

        # Sleep for a short time to prevent excessive looping
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Program interrupted by the user.")

finally:
    # Clean up resources
    picam2.close()
    head_pwm.stop()  # Stop the PWM for the head
    GPIO.cleanup()  # Reset GPIO pins
    print("Resources cleaned up!")
