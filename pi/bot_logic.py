import math
import time
import requests
from gpiozero import PWMOutputDevice, DigitalOutputDevice, DistanceSensor
from mpu6050 import mpu6050

# --- HARDWARE SETUP ---
left_pwm = PWMOutputDevice(18)
left_dir1 = DigitalOutputDevice(17)
left_dir2 = DigitalOutputDevice(27)

right_pwm = PWMOutputDevice(13)
right_dir1 = DigitalOutputDevice(23)
right_dir2 = DigitalOutputDevice(24)

sensor = DistanceSensor(echo=26, trigger=16)
imu = mpu6050(0x68)

# --- NAVIGATION CONSTANTS ---
SPEED_CM_S = 15.0     # Adjust based on your calibration
TURN_SPEED = 0.6      # Speed for turning
FLASK_URL = "http://localhost:5000/update_location"

# --- STATE VARIABLES ---
x, y = 0.0, 0.0
heading = 0.0
search_phase = "STRAIGHT" # STRAIGHT, TURN_1, SHIFT, TURN_2

def motor_control(left, right):
    left_pwm.value = abs(left)
    left_dir1.value = 1 if left > 0 else 0
    left_dir2.value = 0 if left > 0 else 1
    
    right_pwm.value = abs(right)
    right_dir1.value = 1 if right > 0 else 0
    right_dir2.value = 0 if right > 0 else 1

def update_map(dist_moved, angle_change=0):
    global x, y, heading
    heading = (heading + angle_change) % 360
    rad = math.radians(heading)
    x += dist_moved * math.cos(rad)
    y += dist_moved * math.sin(rad)
    
    try:
        requests.post(FLASK_URL, json={'x': round(x, 2), 'y': round(y, 2), 'h': heading}, timeout=0.1)
    except:
        pass

def turn_degrees(target_deg):
    """Uses the IMU Gyro to attempt a precise turn"""
    start_time = time.time()
    print(f"Turning {target_deg} degrees...")
    
    # Simple timed turn (Adjust time based on your robot's weight)
    # 1.0s at 0.6 speed is roughly 90 degrees on many chassis
    duration = abs(target_deg / 90.0) * 0.8 
    
    direction = 1 if target_deg > 0 else -1
    motor_control(TURN_SPEED * direction, -TURN_SPEED * direction)
    time.sleep(duration)
    motor_control(0, 0)
    update_map(0, target_deg)

def area_search():
    global search_phase
    turn_direction = 90 # Alternates between 90 and -90 to snake
    
    try:
        while True:
            dist = sensor.distance * 100
            
            if dist > 25: # Path is clear
                motor_control(0.5, 0.5)
                step_time = 0.2
                update_map(SPEED_CM_S * 0.5 * step_time)
                time.sleep(step_time)
            else:
                # 1. Obstacle Detected: Stop
                motor_control(0, 0)
                print("Wall detected. Shifting to next lane...")
                
                # 2. Perform 'S-Turn' to start next lane
                turn_degrees(turn_direction)  # Turn 90
                
                # Move sideways a bit (the width of the robot)
                motor_control(0.5, 0.5)
                time.sleep(1.0) 
                update_map(SPEED_CM_S * 0.5 * 1.0)
                
                turn_degrees(turn_direction)  # Turn another 90
                
                # 3. Flip direction for next lane
                turn_direction *= -1 
                
    except KeyboardInterrupt:
        motor_control(0, 0)

if __name__ == "__main__":
    area_search()