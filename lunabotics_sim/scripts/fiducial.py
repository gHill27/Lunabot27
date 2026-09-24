#!/usr/bin/env python3

import os

import cv2
import numpy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#Fiducial tracker node
class FiducialTracker(Node):
    def __init__(self):
        super().__init__('fiducial_tracker')
        self.bridge = CvBridge() #allows ROS2 <sensor_msgs/msg/Image> to be compatable with cv2 algorithms

        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10) #only rgb to track the marker
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000) # our tag is either 733 or 669 so should easily be above 249 thus 1000 value dictionary is used. (idk which of the two values it is yet)
        self.aruco_params = cv2.aruco.DetectorParameters_create() #default for now 

    def rgb_callback(self, image:Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cv_image,corners,ids)
        cv2.imshow("ArUco Detection", cv_image)
        cv2.waitKey(1)


  
def main():
    rclpy.init()
    node = FiducialTracker()
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
