#!/usr/bin/env python3

import os

import cv2
import numpy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import qos_profile_sensor_data

#Fiducial tracker node
class FiducialTracker(Node):
    def __init__(self):
        super().__init__('fiducial_tracker')

        # --- Image conversion ---
        self.bridge = CvBridge()  # allows ROS2 <sensor_msgs/msg/Image> to be compatible with cv2 algorithms
        self.debug_counter = 0 
        self.current_frame = None

        # --- ArUco marker setup ---
        self.small_tag_length = 0.12  # size of the tag in m
        self.marker_points = np.array([
            [-self.small_tag_length/2,  self.small_tag_length/2, 0],
            [ self.small_tag_length/2,  self.small_tag_length/2, 0],
            [ self.small_tag_length/2, -self.small_tag_length/2, 0],
            [-self.small_tag_length/2, -self.small_tag_length/2, 0]
        ], dtype=np.float32)

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000) # our tag is above 249, need 1000
        self.aruco_params = cv2.aruco.DetectorParameters_create()  # default for now

        # --- Camera calibration state (populated by camera_info_callback) ---
        self.camera_matrix = None  # calibration matrix for the camera
        self.dist_coeffs = None    # distortion coefficients for the camera

        # --- Publishers ---
        self.debug_pub = self.create_publisher(Image, '/fiducial_tracker/debug_image', 10)

        # --- Subscriptions ---
        self.compressed_img_sub = self.create_subscription( #for real life camera
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.compressed_callback,
            qos_profile_sensor_data
        )

        # self.create_subscription( #sim camera
        #     Image,
        #     '/camera/camera/color/image_raw',
        #     self.rgb_callback,
        #     10
        # )  # only rgb to track the marker

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.rgb_sub = message_filters.Subscriber(
            self,Image, '/camera/camera/color/image_raw',
            qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/depth/image_raw',
            qos_profile=qos_profile_sensor_data
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.synced_callback)

    #-------------------------------------CALLBACKS---------------------------------------------
    def camera_info_callback(self, msg: CameraInfo): # this is a  callback to camera information topic which populates 
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        # self.get_logger().info(f'{self.camera_matrix}')

    def rgb_callback(self, image:Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
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
                    # self.get_logger().info(f'Marker {marker_id}: distance={distance:.3f}m, tvec={tvec.flatten()}') #works!

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
                    # self.get_logger().info(f'Marker {marker_id}: distance={distance:.3f}m, tvec={tvec.flatten()}') #works!

        self.current_frame = cv_image.copy()
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image.copy(), encoding='bgr8'))

    def synced_callback(self, rgbraw:Image, depthraw:Image):
        if self.camera_matrix is None:
            self.get_logger().warn('No camera_info received yet, skipping frame')
            return
        cv_image = self.bridge.imgmsg_to_cv2(rgbraw, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depthraw,desired_encoding='passthrough')

        corners, ids, rejected = cv2.aruco.detectMarkers(
        cv_image, self.aruco_dict, parameters=self.aruco_params
        )
        if ids is None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')) # no tags found but keep camera feed alive
            return
        
        cv2.aruco.drawDetectedMarkers(cv_image, corners, ids) #if they exist draw them
        for i, marker_id in enumerate(ids.flatten()):
            img_points = corners[i].reshape(4, 2).astype(np.float32) #2d locations of the 4 corners of the marker. 

            # --- PnP: gives orientation + a rough distance ---
            success, rvec, tvec = cv2.solvePnP(
                self.marker_points, img_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                continue

            #going into CV math logic.
            # we are going to use the depth camera's z sensing instead of using the marker length for better results
            # This will basically invert the known pinhole approximation reversing it to solve for the x,y unknowns.

            center_u = int(round(np.mean(img_points[:, 0]))) #gets the middle of the corners
            center_v = int(round(np.mean(img_points[:, 1]))) #gets the middle of the corners

            if not (0 <= center_v < depth_image.shape[0] and 0 <= center_u < depth_image.shape[1]): #sanity check that its a valid point
                continue
        
            raw_depth = depth_image[center_v, center_u]

            # Real RealSense = uint16 mm; sim / REP-118 = float32 meters
            if depth_image.dtype == np.uint16:
                depth_m = float(raw_depth) / 1000.0
            else:
                depth_m = float(raw_depth)

            if depth_m <= 0.0 or np.isnan(depth_m):
                self.get_logger().warn(f'No valid depth at marker {marker_id} center, using PnP distance only')
                corrected_tvec = tvec
            else:
                pnp_z = tvec[2][0]  # the Z-component from PnP, along the optical axis
                if pnp_z > 0:
                    corrected_tvec = tvec * (depth_m / pnp_z)
                else:
                    corrected_tvec = tvec

            distance = np.linalg.norm(corrected_tvec)
            self.debug_counter += 1
            if self.debug_counter > 180:
                self.debug_counter = 0
                self.get_logger().info(
                    f'Marker {marker_id}: distance={distance:.3f}m  tvec={corrected_tvec.flatten()}'
                )

            # rvec, corrected_tvec are ready here for TF broadcast / Pose publish

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))
  
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
