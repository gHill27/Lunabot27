#!/usr/bin/env python3

import os

import cv2 # type : ignore
import numpy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class OrbDetector(Node):
    def __init__(self):
        super().__init__('OrbDetector')
        pass

    
def main():
    rclpy.init()
    node = OrbDetector()
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
