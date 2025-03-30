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
PIN_SERVO = 5    # New servo motor control pin

# Audio Configuration
SOUND_FILE = "deterrent_sound.mp3"
INACTIVE_SOUND_FILE = "inactive_sound.mp3"  # New sound file for inactive periods
AUDIO_VOLUME = 2.0 

# Timing Configuration
HEAD_ROTATE_TIME = 10    # Head rotation duration in seconds
DETERRENT_TIME = 5      # Deterrent activation duration in seconds
BIRD_COOLDOWN_TIME = 0.5 # Time to wait before allowing another bird response
SERVO_SURPRISE_TIME = 0.1  # Time for quick surprise motion (seconds)
SERVO_RESET_TIME = 1.0     # Slower time for resetting servo (seconds)

# Servo Configuration
SERVO_FREQ = 50       # Standard servo frequency (50Hz)
SERVO_UP_DUTY = 9.0   # Max duty cycle for up position (adjust as needed for full tension)
SERVO_DOWN_DUTY = 2.5 # Duty cycle for down position (adjust as needed)
SERVO_MAX_SPEED = 100 # Maximum allowable speed parameter for fast movement

# Operation Cycle Configuration
ACTIVE_HOURS =  12*60*60 # for testing active for 30 second14s
INACTIVE_HOURS = 12*60*60  # for testing inactive for 30 seconds

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

# Check if inactive sound file exists
try:
    # Just verify it can be loaded but don't load it yet
    temp_sound = pygame.mixer.Sound(INACTIVE_SOUND_FILE)
    temp_sound = None  # Release the sound resource
    inactive_sound_available = True
except pygame.error:
    print(f"Warning: Inactive sound file could not be loaded: {INACTIVE_SOUND_FILE}")
    inactive_sound_available = False

# Initialize GPIO
h = GPIO.gpiochip_open(0)
GPIO.gpio_claim_output(h, PIN_ARM1)
GPIO.gpio_claim_output(h, PIN_ARM2)
GPIO.gpio_claim_output(h, PIN_LASER)
GPIO.gpio_claim_output(h, PIN_HEAD)
GPIO.gpio_claim_output(h, PIN_LED)  
GPIO.gpio_claim_output(h, PIN_ACTIVE_INDICATOR)
GPIO.gpio_claim_output(h, PIN_SERVO)  # Claim the servo pin

# Initialize PWM for head control
PWM_FREQUENCY = 1000  # 1 kHz
GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0)  # Start PWM at 0% duty cycle

# Initialize PWM for servo control
GPIO.tx_pwm(h, PIN_SERVO, SERVO_FREQ, 0)  # Start servo PWM at 0% duty cycle

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
servo_position = "down"  # Track current servo position
last_surprise_time = 0  # Track last surprise activation time

# ============= Core Functions =============
def detect_objects():
    """
    Continuously detects objects using the camera and YOLO model.
    Updates the global detected_objects list with current detections.
    """
    global detected_objects
    while True:
        # Check system state with minimal lock time
        with system_state_lock:
            currently_active = system_active
            
        if not currently_active:
            time.sleep(0.5)  # Short sleep when inactive
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

def quick_servo_surprise():
    """Performs a fast upward servo movement to create a surprise action."""
    global servo_position, last_surprise_time
    
    # If we did a surprise recently, wait a bit to avoid potential damage
    current_time = time.time()
    if current_time - last_surprise_time < 3.0:  # Minimum 3 seconds between surprises
        time.sleep(current_time - last_surprise_time)
    
    print("SURPRISE! Fast servo activation!")
    
    # Quick snap to up position for surprise effect
    GPIO.tx_pwm(h, PIN_SERVO, SERVO_FREQ, SERVO_UP_DUTY)
    time.sleep(SERVO_SURPRISE_TIME)  # Very short time for surprise effect
    
    servo_position = "up"
    last_surprise_time = time.time()

def move_servo_down():
    """Moves the servo to the down position at a moderate speed."""
    global servo_position
    
    # If already down, do nothing
    if servo_position == "down":
        return
    
    print("Slowly resetting servo to down position")
    GPIO.tx_pwm(h, PIN_SERVO, SERVO_FREQ, SERVO_DOWN_DUTY)
    time.sleep(SERVO_RESET_TIME)  # Longer time for a more gentle reset
    servo_position = "down"

def play_deterrent_sound():
    """Plays the deterrent sound using pygame mixer."""
    print("Sound playing")
    pygame.mixer.music.load(SOUND_FILE)  # Ensure the correct sound is loaded
    pygame.mixer.music.set_volume(AUDIO_VOLUME)
    pygame.mixer.music.play()

def play_inactive_sound():
    """Plays the inactive period sound using pygame mixer."""
    if inactive_sound_available:
        print("Inactive sound playing")
        pygame.mixer.music.load(INACTIVE_SOUND_FILE)  # Load the inactive sound file
        pygame.mixer.music.set_volume(AUDIO_VOLUME)
        pygame.mixer.music.play()
    else:
        print("Inactive sound file not available, skipping playback")

def activate_deterrents():
    """Activates all deterrent mechanisms."""
    print("deterrent activated")
    
    # Synchronize the sound and servo surprise movement
    quick_servo_surprise()  # Fast surprise action first
    play_deterrent_sound()  # Play sound immediately after
    
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
    
    # Return servo to down position when deterrents are deactivated
    move_servo_down()

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
        
        activate_deterrents()  # This will trigger the surprise servo action
        
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
    Now includes surprise servo actions during interval cycles.
    """
    global servo_position
    
    while True:
        # Check system state with minimal lock time
        with system_state_lock:
            current_interval_mode = in_interval_mode
            currently_active = system_active
        
        if currently_active:
            # ACTIVE MODE INTERVALS
            if current_interval_mode:
                print("Interval Mode: Rotating Head with pulses.")
                
                # Make sure servo is in down position at start of interval
                if servo_position != "down":
                    move_servo_down()
                
                # Calculate timing for pulsed rotation
                rotation_end_time = time.time() + HEAD_ROTATE_TIME
                pulse_on_time = 0.2  # Time the head rotates before pausing
                pulse_off_time = 0.5  # Time the head pauses
                
                # Perform pulsed rotation until rotation time is complete
                while time.time() < rotation_end_time:
                    # Turn head on
                    GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 15)
                    
                    # Wait for pulse_on_time or until interval mode ends
                    pulse_on_end = time.time() + pulse_on_time
                    while time.time() < pulse_on_end:
                        with system_state_lock:
                            if not in_interval_mode or not system_active:
                                break
                        time.sleep(0.02)  # Short sleep to prevent CPU hogging
                    
                    # Check if we're still in interval mode and active
                    with system_state_lock:
                        if not in_interval_mode or not system_active:
                            break
                    
                    # Turn head off briefly
                    GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0)
                    
                    # Wait for pulse_off_time or until interval mode ends
                    pulse_off_end = time.time() + pulse_off_time
                    while time.time() < pulse_off_end:
                        with system_state_lock:
                            if not in_interval_mode or not system_active:
                                break
                        time.sleep(0.02)
                    
                    # Check if we're still in interval mode and active
                    with system_state_lock:
                        if not in_interval_mode or not system_active:
                            break
                    
                with system_state_lock:
                    current_interval_mode = in_interval_mode
                    currently_active = system_active
                
                if current_interval_mode and currently_active:
                    print("Interval Mode: Stopping head, activating deterrents.")
                    GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0) 
                    
                    # This will create the surprise effect with fast servo movement
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
                        deactivate_deterrents()  # This will also move servo down
        else:
            # INACTIVE MODE - Only play the sound at intervals, no physical deterrents
            print("Inactive Mode: Playing inactive sound.")
            
            # Ensure servo is in down position during inactive mode
            if servo_position != "down":
                move_servo_down()
                
            play_inactive_sound()
            
            # Wait for the sound duration (using DETERRENT_TIME for consistency)
            sound_end_time = time.time() + DETERRENT_TIME
            while time.time() < sound_end_time:
                with system_state_lock:
                    if system_active:  # Exit if we switch to active mode
                        break
                time.sleep(0.1)
                
            pygame.mixer.music.stop()  # Ensure sound stops
            
            # Wait for the equivalent of HEAD_ROTATE_TIME before playing sound again
            pause_end_time = time.time() + HEAD_ROTATE_TIME
            while time.time() < pause_end_time:
                with system_state_lock:
                    if system_active:  # Exit if we switch to active mode
                        break
                time.sleep(0.5)
        
        time.sleep(0.5)  # Shorter sleep to be more responsive
        
def time_cycle_controller():
    """
    Controls the active and inactive periods of the system.
    Simplified to immediately toggle system state with direct timing.
    """
    global system_active
    
    while True:
        # Active period
        print("==== SYSTEM ACTIVE PERIOD STARTED ====")
        with system_state_lock:
            system_active = True
        
        GPIO.gpio_write(h, PIN_ACTIVE_INDICATOR, 1)
        
        # Simply sleep for the active period
        time.sleep(ACTIVE_HOURS)
        
        # Inactive period
        print("==== SYSTEM INACTIVE PERIOD STARTED ====")
        with system_state_lock:
            system_active = False
        
        GPIO.gpio_write(h, PIN_ACTIVE_INDICATOR, 0)
        # Deactivate physical deterrents but don't stop audio (handled by interval_mode_cycle)
        GPIO.gpio_write(h, PIN_LASER, 0)
        GPIO.gpio_write(h, PIN_ARM1, 0)
        GPIO.gpio_write(h, PIN_ARM2, 0)
        GPIO.gpio_write(h, PIN_LED, 0)
        GPIO.tx_pwm(h, PIN_HEAD, PWM_FREQUENCY, 0)
        
        # Ensure servo is in down position during inactive period
        move_servo_down()
        
        # Simply sleep for the inactive period
        time.sleep(INACTIVE_HOURS)
        
        print("==== INACTIVE PERIOD COMPLETE, RETURNING TO ACTIVE STATE ====")
        # The loop will automatically set system_active to True at the start of the next iteration

# ============= Main Program =============
def main():
    """Main program entry point."""
    try:
        # Initialize servo to down position at startup
        move_servo_down()
        
        # Start detection thread
        detection_thread = threading.Thread(target=detect_objects, daemon=True)
        detection_thread.start()

        # Start interval mode thread
        interval_thread = threading.Thread(target=interval_mode_cycle, daemon=True)
        interval_thread.start()
        
        # Start time cycle controller thread
        time_cycle_thread = threading.Thread(target=time_cycle_controller, daemon=True)
        time_cycle_thread.start()

        # Main loop
        while True:
            try:
                # Check system state with minimal lock time
                with system_state_lock:
                    currently_active = system_active
                
                if not currently_active:
                    # System is inactive, but we don't need to do anything here
                    # as the interval_mode_cycle thread handles the inactive sound playing
                    time.sleep(1)
                    continue
                
                # Process bird detection when active
                with detection_lock:
                    bird_detected = any(cls in BIRDS_AND_FLOCK for cls in detected_objects)
                
                if bird_detected:
                    with system_state_lock:
                        if not bird_response_running:
                            threading.Thread(target=bird_detected_response, daemon=True).start()
                
                time.sleep(0.1)  # Shorter sleep to be more responsive
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
        GPIO.tx_pwm(h, PIN_SERVO, SERVO_FREQ, 0)  # Stop servo PWM
        GPIO.gpiochip_close(h)
        print("Resources cleaned up!")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        time.sleep(1)

if __name__ == "__main__":
    main()
