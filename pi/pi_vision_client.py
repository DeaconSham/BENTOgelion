import io
import time
import requests
try:
    from picamera2 import Picamera2
    from libcamera import Transform
except ImportError:
    import cv2

MAC_VISION_SERVER_IP = "192.168.1.100"  # Update with your Mac's IP
MAC_VISION_SERVER_PORT = 5001
VISION_SERVER_URL = f"http://{MAC_VISION_SERVER_IP}:{MAC_VISION_SERVER_PORT}/upload_frame"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 2
JPEG_QUALITY = 75

def init_picamera():
    try:
        camera = Picamera2()
        config = camera.create_still_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT)},
            transform=Transform(hflip=False, vflip=False)
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)
        print("✓ Pi Camera initialized")
        return camera, "picamera"
    except Exception as e:
        print(f"Pi Camera error: {e}")
        return None, None

def init_usb_camera():
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not camera.isOpened():
            return None, None
        
        print("✓ USB Camera initialized")
        return camera, "usb"
    except Exception as e:
        print(f"USB Camera error: {e}")
        return None, None

def capture_and_send_frame_picamera(camera):
    try:
        frame = camera.capture_array()
        
        from PIL import Image
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        buffer.seek(0)
        
        files = {'image': ('frame.jpg', buffer, 'image/jpeg')}
        response = requests.post(VISION_SERVER_URL, files=files, timeout=2.0)
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def capture_and_send_frame_usb(camera):
    try:
        ret, frame = camera.read()
        if not ret:
            return False
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        files = {'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
        response = requests.post(VISION_SERVER_URL, files=files, timeout=2.0)
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print(f"Sending frames to: {VISION_SERVER_URL}")
    print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE} FPS\n")
    
    camera, camera_type = init_picamera()
    if camera is None:
        camera, camera_type = init_usb_camera()
    
    if camera is None:
        print("No camera available")
        return
    
    capture_func = capture_and_send_frame_picamera if camera_type == "picamera" else capture_and_send_frame_usb
    
    print("Testing connection...")
    try:
        health_url = f"http://{MAC_VISION_SERVER_IP}:{MAC_VISION_SERVER_PORT}/health"
        response = requests.get(health_url, timeout=3.0)
        if response.status_code == 200:
            print("✓ Connected to Mac\n")
    except Exception as e:
        print(f"⚠ Cannot reach Mac: {e}\n")
    
    print("Starting capture... (Ctrl+C to stop)\n")
    
    frame_count = 0
    success_count = 0
    interval = 1.0 / FRAME_RATE
    
    try:
        while True:
            start_time = time.time()
            
            if capture_func():
                success_count += 1
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Frames: {frame_count} (success: {success_count/frame_count*100:.1f}%)")
            
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))
    
    except KeyboardInterrupt:
        print(f"\nSent {success_count}/{frame_count} frames")
        
        if camera_type == "picamera":
            camera.stop()
        elif camera_type == "usb":
            camera.release()
        
        print("Stopped")

if __name__ == "__main__":
    main()
