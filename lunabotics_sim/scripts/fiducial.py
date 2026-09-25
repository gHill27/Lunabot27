#!/usr/bin/env python3

import os

import cv2
import numpy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

#Fiducial tracker node
class FiducialTracker(Node):
    def __init__(self):
        super().__init__('fiducial_tracker')
        self.bridge = CvBridge() #allows ROS2 <sensor_msgs/msg/Image> to be compatable with cv2 algorithms
        self.current_frame = None
        self.qos_profile = None
        self.small_tag_length = 0.12 # size of the tag in m
        self.marker_points = np.array([
            [-self.small_tag_length/2,  self.small_tag_length/2, 0],
            [ self.small_tag_length/2,  self.small_tag_length/2, 0],
            [ self.small_tag_length/2, -self.small_tag_length/2, 0],
            [-self.small_tag_length/2, -self.small_tag_length/2, 0]
        ],dtype=np.float32)

        self.debug_pub = self.create_publisher(Image, '/fiducial_tracker/debug_image', 10)
        self.compressed_img_sub = self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', self.compressed_callback, qos_profile_sensor_data)


        self.camera_matrix = None # calibration matrix for the camera
        self.dist_coeffs = None # distortion coefficients for the camera
        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10) #only rgb to track the marker
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data
        )
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000) # our tag is either 733 or 669 so should easily be above 249 thus 1000 value dictionary is used. (idk which of the two values it is yet)
        self.aruco_params = cv2.aruco.DetectorParameters_create() #default for now 

    
    def camera_info_callback(self, msg: CameraInfo): # this is a  callback to camera information topic which populates 
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def rgb_callback(self, image:Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            # self.get_logger().inzfo(f'tag of id {ids} found')
            cv2.aruco.drawDetectedMarkers(cv_image,corners,ids)
        self.current_frame = cv_image.copy()
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image.copy(), encoding='bgr8'))

    def compressed_callback(self, compressed_image:CompressedImage) -> None:
        cv_image = self.bridge.compressed_imgmsg_to_cv2(compressed_image, desired_encoding='bgr8')
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            if self.camera_matrix is None:
                self.get_logger().warn('No camera_info received yet, skipping pose estimation')
            else:
                cv2.aruco.drawDetectedMarkers(cv_image,corners,ids)
                for i, marker_id in enumerate(ids.flatten()): # this changes the ids to be one array of length 1xn so its iterable. 
                    img_points = corners[i].reshape(4, 2).astype(np.float32) # reshape the corners to 4x2 (4 x y pairs) and convert to float32
                    success, rvec, tvec = cv2.solvePnP( # tvec = translation vector, rvec = rotation vector
                        self.marker_points, img_points,
                        self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                if success:
                    distance = np.linalg.norm(tvec) # computes the magnitude of the vector 
                    self.get_logger().info(f'Marker {marker_id}: distance={distance:.3f}m, tvec={tvec.flatten()}') #works!

        self.current_frame = cv_image.copy()
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image.copy(), encoding='bgr8'))
  
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
