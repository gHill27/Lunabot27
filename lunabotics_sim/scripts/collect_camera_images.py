#!/usr/bin/env python3
"""
Subscribes to the simulated RealSense-style camera and saves a frame
(RGB + depth) to disk at a fixed interval, to demonstrate the sensor is
actually producing usable data.

Requires: cv_bridge, opencv-python (rosdep install / apt should cover
cv_bridge; opencv-python via pip if not already present).

Run with:
    python3 collect_camera_images.py
(after sourcing your ROS 2 workspace, with the sim already running)

Images are saved to ./captured_images/ as:
    rgb_0000.png, depth_0000.png, rgb_0001.png, depth_0001.png, ...
"""

import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

SAVE_DIR = "captured_images"
CAPTURE_PERIOD_SEC = 2.0  # save a frame every N seconds


class CameraCollector(Node):
    def __init__(self):
        super().__init__('camera_collector')
        os.makedirs(SAVE_DIR, exist_ok=True)

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_count = 0

        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)

        self.timer = self.create_timer(CAPTURE_PERIOD_SEC, self.save_frame)

        self.get_logger().info(
            f"Listening on /camera/image and /camera/depth_image, "
            f"saving a frame every {CAPTURE_PERIOD_SEC}s to ./{SAVE_DIR}/"
        )

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        # Depth comes through as 32FC1 (meters) from the rgbd_camera sensor
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.latest_depth = depth

    def save_frame(self):
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn("No image data received yet — is the sim running?")
            return

        rgb_path = os.path.join(SAVE_DIR, f"rgb_{self.frame_count:04d}.png")
        depth_path = os.path.join(SAVE_DIR, f"depth_{self.frame_count:04d}.png")

        cv2.imwrite(rgb_path, self.latest_rgb)

        # Normalize depth (meters, float32) to 0-255 for a viewable PNG.
        # Raw float depth isn't saved here — this is just for quick visual
        # confirmation the sensor is producing real depth data.
        depth_vis = np.nan_to_num(self.latest_depth, nan=0.0, posinf=0.0)
        depth_vis = np.clip(depth_vis, 0, 10.0)  # matches <far>10.0</far> in the sensor
        depth_vis = (depth_vis / 10.0 * 255).astype('uint8')
        cv2.imwrite(depth_path, depth_vis)

        self.get_logger().info(f"Saved {rgb_path} and {depth_path}")
        self.frame_count += 1


def main():
    rclpy.init()
    node = CameraCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
