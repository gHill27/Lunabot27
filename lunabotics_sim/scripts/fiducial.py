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
        self.current_frame = None
        self.debug_pub = self.create_publisher(Image, '/fiducial_tracker/debug_image', 10)

        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10) #only rgb to track the marker
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000) # our tag is either 733 or 669 so should easily be above 249 thus 1000 value dictionary is used. (idk which of the two values it is yet)
        self.aruco_params = cv2.aruco.DetectorParameters_create() #default for now 

    def rgb_callback(self, image:Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            # self.get_logger().inzfo(f'tag of id {ids} found')
            cv2.aruco.drawDetectedMarkers(cv_image,corners,ids)
        self.current_frame = cv_image
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))
  
def main():
    rclpy.init()
    node = FiducialTracker()
    try:
       rclpy.spin(node)
        # while rclpy.ok():
        #     rclpy.spin_once(node, timeout_sec=0.01)  # process one callback
        #     if node.current_frame is not None:
        #         cv2.imshow("ArUco Detection", node.current_frame)
        #         cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
