# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def capture_image(output_path="captured_image.jpg"):
    pipeline = gstreamer_pipeline(flip_method=0)
    print(pipeline)
    
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not video_capture.isOpened():
        print("Error: Unable to open camera")
        return False

    try:
        # Read a few frames to let auto-exposure / white balance settle
        for _ in range(10):
            ret_val, frame = video_capture.read()
            if not ret_val:
                time.sleep(0.05)

        # Capture the final photo
        ret_val, frame = video_capture.read()
        if ret_val and frame is not None:
            cv2.imwrite(output_path, frame)
            print(f"Image successfully saved to {output_path}")
            return True
        else:
            print("Error: Failed to read frame from camera")
            return False
            
    finally:
        video_capture.release()


if __name__ == "__main__":
    capture_image("test_photo.jpg")