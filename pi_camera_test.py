#!/usr/bin/env python3
"""
Lightweight Raspberry Pi Camera Live Stream
Simply displays a live feed from the Pi Camera.
Press Ctrl+C to exit.
"""

# This script uses picamera2, the modern library for Raspberry Pi cameras
from picamera2 import Picamera2
import time

print("Starting Pi Camera live stream...")
print("Press Ctrl+C to exit")

# Initialize the camera
picam2 = Picamera2()

# Configure with default preview configuration
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

# Try different preview methods based on what's available
try:
    # Method 1: Try using the built-in preview if available
    print("Starting camera with built-in preview...")
    picam2.start_preview()
    picam2.start()
    
    print("Camera preview started. Press Ctrl+C to exit.")
    # Keep the preview running until interrupted
    while True:
        time.sleep(1)
        
except (ImportError, AttributeError, TypeError) as e:
    print(f"Built-in preview not available: {e}")
    print("Trying alternative method...")
    
    # Method 2: Just start the camera without preview
    picam2.start()
    
    # Take a test image to verify camera works
    print("Taking a test image...")
    picam2.capture_file("test_image.jpg")
    print("Test image saved as 'test_image.jpg'")
    
    print("Camera running for 30 seconds...")
    # Run for 30 seconds
    time.sleep(30)
    
finally:
    # Clean up
    try:
        picam2.stop_preview()
    except:
        pass
    picam2.stop()
    print("Camera closed")
    
print("\nIf you didn't see a preview, check for test_image.jpg")
print("You can also try these commands directly:")
print("  libcamera-hello -t 10000    # Preview for 10 seconds")
print("  libcamera-still -o test.jpg # Take a photo")
