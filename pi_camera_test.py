#!/usr/bin/env python3
"""
Ultra Simple Raspberry Pi Camera Test
Just takes a photo to verify the camera works.
"""

import subprocess
import time
import os

print("===== Raspberry Pi Camera Test =====\n")

# Method 1: Try using libcamera-still (most reliable)
print("Method 1: Testing with libcamera-still...")
try:
    print("Taking a photo with libcamera-still...")
    subprocess.run(["libcamera-still", "-o", "test_photo.jpg", "-t", "2000"], 
                 check=True)
    print("✓ Success! Check test_photo.jpg")
except Exception as e:
    print(f"Error with libcamera-still: {e}")
    print("Trying next method...")

# Method 2: Try using libcamera-hello
print("\nMethod 2: Testing with libcamera-hello...")
try:
    print("Starting preview for 5 seconds...")
    subprocess.run(["libcamera-hello", "-t", "5000"], check=True)
    print("✓ Success! Preview completed")
except Exception as e:
    print(f"Error with libcamera-hello: {e}")
    print("Trying next method...")

# Method 3: Try using raspistill (legacy method)
print("\nMethod 3: Testing with raspistill (legacy)...")
try:
    print("Taking a photo with raspistill...")
    subprocess.run(["raspistill", "-o", "legacy_photo.jpg", "-t", "2000"], 
                 check=True)
    print("✓ Success! Check legacy_photo.jpg")
except Exception as e:
    print(f"Error with raspistill: {e}")

print("\n===== Test Complete =====")
print("If any method succeeded, your camera is working!")
print("If all methods failed, check your camera connection")
print("and make sure the camera is enabled in raspi-config.")

