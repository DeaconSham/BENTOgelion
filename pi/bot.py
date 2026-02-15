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
TURN_SPEED = 0.42      # Speed for turning
MAC_BACKEND_IP = "172.20.10.3"  # Your Mac's IP address
FLASK_URL = f"http://{MAC_BACKEND_IP}:5002/update_location"

# --- STATE VARIABLES ---
x, y = 0.0, 0.0
heading = 0.0
search_phase = "STRAIGHT" # STRAIGHT, TURN_1, SHIFT, TURN_2
#DO NOT ADJUST BIAS IF IT GOES STRAIGHT
LEFT_MOTOR_BIAS = 0.98  # The left motor will only run at 91.5% of the requested speed (lower if too strong, higher if too weak)
RIGHT_MOTOR_BIAS = 1.00  # The right motor runs at 100%

def motor_control(left, right):
    # Apply the bias to the inputs
    final_left = left * LEFT_MOTOR_BIAS
    final_right = right * RIGHT_MOTOR_BIAS
    
    left_pwm.value = abs(final_left)
    left_dir1.value = 1 if final_left > 0 else 0
    left_dir2.value = 0 if final_left > 0 else 1
    
    right_pwm.value = abs(final_right)
    right_dir1.value = 1 if final_right > 0 else 0
    right_dir2.value = 0 if final_right > 0 else 1

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



# --- CALIBRATION ---
TIME_LEFT_90 = 0.94   # Time for a 90-deg turn to the LEFT (decrease if turns left too much)
TIME_RIGHT_90 = 0.90  # Time for a 90-deg turn to the RIGHT (decrease if right turn too much)

 
def turn_degrees(target_deg):
    if target_deg > 0:
        magic_number = TIME_LEFT_90
        motor_control(-TURN_SPEED, TURN_SPEED) # Swing turn left
    else:
        magic_number = TIME_RIGHT_90
        motor_control(TURN_SPEED, -TURN_SPEED) # Swing turn right

    duration = abs(target_deg / 90.0) * magic_number
    
    # --- GYRO MEASUREMENT START ---
    start_time = time.time()
    total_rotated = 0
    last_time = start_time
    
    while (time.time() - start_time) < duration:
        current_time = time.time()
        dt = current_time - last_time
        
        # Read the Z-axis (yaw) angular velocity
        gyro_z = imu.get_gyro_data()['z'] 
        
        # Integrate velocity to get degrees: change = velocity * time
        total_rotated += gyro_z * dt
        
        last_time = current_time
        time.sleep(0.01) # High frequency sampling for accuracy
    
    motor_control(0, 0)
    
    # Instead of telling the map "target_deg", tell it what the GYRO saw!
    # Note: Gyro values might need a minus sign depending on orientation
    print(f"Target: {target_deg}, Gyro measured: {total_rotated}")
    update_map(0, total_rotated)

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