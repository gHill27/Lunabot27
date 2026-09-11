#!/usr/bin/env python3
"""
Live viewer for the simulated RealSense-style camera, using OpenCV windows
instead of rqt_image_view (which has known issues auto-scaling 32FC1 float
depth images).

Run with:
    python3 live_camera_view.py
(after sourcing your ROS 2 workspace, with the sim already running)

Press 'q' in either window to quit.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

DEPTH_MAX_M = 10.0  # matches <far>10.0</far> in the sensor definition


class LiveCameraView(Node):
    def __init__(self):
        super().__init__('live_camera_view')
        self.bridge = CvBridge()

        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)

        self.get_logger().info("Live viewer running. Press 'q' in a window to quit.")

    def rgb_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('RGB', img)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        depth_vis = np.nan_to_num(depth, nan=0.0, posinf=DEPTH_MAX_M, neginf=0.0)
        depth_vis = np.clip(depth_vis, 0, DEPTH_MAX_M)
        depth_vis = (depth_vis / DEPTH_MAX_M * 255).astype('uint8')
        # Optional: apply a colormap instead of grayscale for easier reading
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow('Depth', depth_color)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = LiveCameraView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
