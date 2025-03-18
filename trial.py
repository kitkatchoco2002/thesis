# Standard library imports
import time
import threading
import os

# Third-party imports  
import pygame
from picamera2 import Picamera2
from ultralytics import YOLO
import lgpio as GPIO  # Use lgpio instead of RPi.GPIO   

# ============= Configuration Constants =============
# GPIO Pin Configuration
PIN_ARM1 = 17     # Arm 1 control
PIN_ARM2 = 25     # Arm 2 control
PIN_LASER = 27    # LASER control
PIN_HEAD = 23     # HEAD control
PIN_LED = 22      # LED indicator for bird detection
PIN_ACTIVE_INDICATOR = 24  # LED indicator for active system state

# Audio Configuration
SOUND_FILE = "deterrent_sound.mp3"

# Timing Configuration
HEAD_ROTATE_TIME = 3    # Head rotation duration in seconds
DETERRENT_TIME = 15      # Deterrent activation duration in seconds
BIRD_COOLDOWN_TIME = 0.5 # Time to wait before allowing another bird response

# Operation Cycle Configuration
ACTIVE_HOURS = 14 * 60 * 60  # 14 hours in seconds
INACTIVE_HOURS = 10 * 60 * 60  # 10 hours in seconds

# Object Detection Classes
BIRDS_AND_FLOCK = [1, 2]        # Bird and flock class IDs
HUMANS_AND_FAKE_BIRDS = [0, 3]  # Fake bird and human class IDs

# ============= Hardware Initialization =============
# Initialize audio system
pygame.mixer.init()
os.environ['SDL_AUDIODRIVER'] = 'alsa'  # Set the audio driver explicitly
try:
    pygame.mixer.music.load(SOUND_FILE)
except pygame.error:
    print(f"Error loading sound file: {SOUND_FILE}")

# Initialize GPIO
h = GPIO.gpiochip_open(0)
GPIO.gpio_claim_output(h, PIN_ARM1)
GPIO.gpio_claim_output(h, PIN_ARM2)
GPIO.gpio_claim_output(h, PIN_LASER)
GPIO.gpio_claim_output(h, PIN_HEAD)
GPIO.gpio_claim_output(h, PIN_LED)  
GPIO.gpio_claim_output(h, PIN_ACTIVE_INDICATOR)

# Initialize PWM for head control
PWM_FREQUENCY = 1000  # 1 kHz
GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0)  # Start PWM at 0% duty cycle

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
system_state_lock = threading.Lock()
in_interval_mode = True
bird_response_running = False
last_bird_response_time = 0
system_active = True  # Flag to indicate if the system is in active hours

# ============= Core Functions =============
def detect_objects():
    """
    Continuously detects objects using the camera and YOLO model.
    Updates the global detected_objects list with current detections.
    """
    global detected_objects
    while True:
        with system_state_lock:
            if not system_active:
                time.sleep(5)  # Sleep longer when system is inactive
                continue
                
        try:
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
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            time.sleep(1)

def play_deterrent_sound():
    """Plays the deterrent sound using pygame mixer."""
    print("Sound playing")
    pygame.mixer.music.play()

def activate_deterrents():
    """Activates all deterrent mechanisms."""
    print("deterrent activated")
    play_deterrent_sound()
    time.sleep(0.1)
    GPIO.gpio_write(h, PIN_LASER, 1)
    GPIO.gpio_write(h, PIN_ARM1, 1)
    GPIO.gpio_write(h, PIN_ARM2, 1)
    GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0) 
    print("deterrents activated")

def deactivate_deterrents():
    """Deactivates all deterrent mechanisms."""
    GPIO.gpio_write(h, PIN_LASER, 0)
    GPIO.gpio_write(h, PIN_ARM1, 0)
    GPIO.gpio_write(h, PIN_ARM2, 0)
    pygame.mixer.music.stop()  

def bird_detected_response():
    """
    Handles the response when birds are detected.
    Activates deterrents for the full duration without interruption.
    """
    global in_interval_mode, bird_response_running, last_bird_response_time
    
    with system_state_lock:
        if not system_active or bird_response_running:
            return
        bird_response_running = True
        in_interval_mode = False
        last_bird_response_time = time.time()

    try:
        print("Bird detected! Activating deterrents.")
        GPIO.gpio_write(h, PIN_LED, 1)  # Turn on LED to indicate bird detection
        activate_deterrents()
        
        # Run deterrents for the full duration without interruption
        complete_deterrent_time = time.time() + DETERRENT_TIME
        while time.time() < complete_deterrent_time:
            with system_state_lock:
                if not system_active:
                    break
            time.sleep(0.1)  # Small sleep to prevent CPU hogging
            
        # Deterrent cycle complete, deactivate everything
        deactivate_deterrents()
        GPIO.gpio_write(h, PIN_LED, 0)  # Turn off LED
        print("Deterrents turned off. Returned to interval mode.")
    
    except Exception as e:
        print(f"Error in bird response: {str(e)}")
    
    finally:
        # Wait for cooldown period before allowing another bird response
        cooldown_time = BIRD_COOLDOWN_TIME - (time.time() - last_bird_response_time)
        if cooldown_time > 0:
            time.sleep(cooldown_time)
            
        with system_state_lock:
            bird_response_running = False
            in_interval_mode = True

def interval_mode_cycle():
    """
    Manages the interval mode cycle of the deterrent system.
    Alternates between head rotation and deterrent activation.
    """
    while True:
        with system_state_lock:
            current_interval_mode = in_interval_mode
            currently_active = system_active
        
        if not currently_active:
            time.sleep(5)  # Sleep longer when system is inactive
            continue
            
        if current_interval_mode:
            print("Interval Mode: Rotating Head.")
            GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 50) 
            
            # Check if we're still in interval mode after head rotation
            rotation_end_time = time.time() + HEAD_ROTATE_TIME
            while time.time() < rotation_end_time:
                with system_state_lock:
                    if not in_interval_mode or not system_active:
                        break
                time.sleep(0.1)
            
            with system_state_lock:
                current_interval_mode = in_interval_mode
                currently_active = system_active
            
            if current_interval_mode and currently_active:
                print("Interval Mode: Stopping head, activating deterrents.")
                GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0) 
                activate_deterrents()
                
                # Check if we're still in interval mode during deterrent activation
                deterrent_end_time = time.time() + DETERRENT_TIME
                while time.time() < deterrent_end_time:
                    with system_state_lock:
                        if not in_interval_mode or not system_active:
                            break
                    time.sleep(0.1)
                
                with system_state_lock:
                    current_interval_mode = in_interval_mode
                    currently_active = system_active
                
                if current_interval_mode and currently_active:
                    print("Interval Mode: Turning off deterrents.")
                    deactivate_deterrents()
        
        time.sleep(1)

def time_cycle_controller():
    """
    Controls the active/inactive cycle of the system based on time.
    Runs for ACTIVE_HOURS, then stops for INACTIVE_HOURS, and repeats.
    Uses PIN_ACTIVE_INDICATOR as a constant indicator when system is active.
    """
    global system_active
    
    while True:
        # Start active period
        print("==== SYSTEM ACTIVE PERIOD STARTED ====")
        print(f"System will be active for {ACTIVE_HOURS//3600} hours")
        
        # Turn on ACTIVE_INDICATOR to show system is active
        GPIO.gpio_write(h, PIN_ACTIVE_INDICATOR, 1)
        
        with system_state_lock:
            system_active = True
        
        # Stay active for ACTIVE_HOURS
        time.sleep(ACTIVE_HOURS)
        
        # Start inactive period
        print("==== SYSTEM INACTIVE PERIOD STARTED ====")
        print(f"System will be inactive for {INACTIVE_HOURS//3600} hours")
        
        # Turn off ACTIVE_INDICATOR to show system is inactive
        GPIO.gpio_write(h, PIN_ACTIVE_INDICATOR, 0)
        
        with system_state_lock:
            system_active = False
            
        # Ensure all deterrents are off during inactive period
        deactivate_deterrents()
        GPIO.gpio_write(h, PIN_LED, 0)
        GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0)
        
        # Stay inactive for INACTIVE_HOURS
        time.sleep(INACTIVE_HOURS)

# ============= Main Program =============
def main():
    """Main program entry point."""
    try:
        # Start detection thread
        detection_thread = threading.Thread(target=detect_objects, daemon=True)
        detection_thread.start()

        # Start interval mode thread
        interval_thread = threading.Thread(target=interval_mode_cycle, daemon=True)
        interval_thread.start()
        
        # Start time cycle controller thread
        time_cycle_thread = threading.Thread(target=time_cycle_controller, daemon=True)
        time_cycle_thread.start()

        while True:
            try:
                with system_state_lock:
                    currently_active = system_active
                
                if currently_active:
                    # Check for birds
                    bird_detected = False
                    with detection_lock:
                        bird_detected = any(cls in BIRDS_AND_FLOCK for cls in detected_objects)
                    
                    # Start bird response if needed
                    if bird_detected:
                        with system_state_lock:
                            if not bird_response_running:
                                threading.Thread(target=bird_detected_response, daemon=True).start()
                
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(1)

    except KeyboardInterrupt:
        deactivate_deterrents()
        print("Program interrupted by the user.")
        
    except Exception as e:
        print(f"Critical error in main program: {str(e)}")
        time.sleep(1)
    finally:
        cleanup()

def cleanup():
    """Cleanup resources before program exit."""
    try:                                             
        picam2.close()
        GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0)
        GPIO.gpiochip_close(h)
        print("Resources cleaned up!")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        time.sleep(1)

if __name__ == "__main__":
    main()
