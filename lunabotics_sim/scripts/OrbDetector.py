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
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=1000)


        self.pub = self.create_publisher(Image,'/OrbDetector/debug_image', 10)
        self.create_subscription(Image, '/camera/image', self.rgb_callback, qos_profile_sensor_data) #only rgb to track the marker
        pass

    def rgb_callback(self,msg:Image):
        try:
            cv_imgae = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'mono8')
        except Exception as e:
            self.get_logger().error(f'image not able to be read with error {e}')
            return
        
        keypoints, descriptors = self.orb.detectAndCompute(cv_imgae, None)
        out = cv2.drawKeypoints(cv_imgae, keypoints, None, color=(0, 255, 0))
        self.pub.publish(self.bridge.cv2_to_imgmsg(out.copy(), encoding='bgr8'))


def main():
    rclpy.init()
    node = OrbDetector()
    try:
       rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
