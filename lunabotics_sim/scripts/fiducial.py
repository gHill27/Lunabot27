#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#Fiducial tracker node
class FiducialTracker(Node):
    def __init__(self):
        super.__init__("fiducial_tracker")
        self.bridge = CvBridge() #allows ROS2 <sensor_msgs/msg/Image> to be compatable with cv2 algorithms

        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10) #only rgb to track the marker

    def rgb_callback(self, image:Image) -> None:
        pass