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

class CannyEdgeDector(Node):
    def __init__(self):
        super().__init__('EdgeDetector')
        self.bridge = CvBridge()
        
        #publisher: 
        self.pub = self.create_publisher(Image,'/edge_detector/debug_image', 10)

        #sub: 
        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10) #only rgb to track the marker
        pass

    def rgb_callback(self,msg:Image):
        #take the image and convert to cv2
        cv2image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        kernal = (5, 5)
        blurred_image = cv2.GaussianBlur(cv2image, kernal, 0) #0 means calculate standard deviation automatically
        edges = cv2.Canny(image=blurred_image, threshold1=100, threshold2=200)
        self.pub.publish(self.bridge.cv2_to_imgmsg(edges.copy(), encoding='mono8'))

def main():
    rclpy.init()
    node = CannyEdgeDector()
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
