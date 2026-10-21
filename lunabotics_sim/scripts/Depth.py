#!/usr/bin/env python3

import os
import cv2 # type : ignore
import cv_bridge
import open3d as o3d #type: ignore
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from rclpy.qos import qos_profile_sensor_data, ReliabilityPolicy, QoSProfile, HistoryPolicy

class DepthMapper(Node):
    """NOTE: REALSENSE CAMERA HAS A LOT OF THESE FEATURES IN THE SDK TO INVESTIGATE LATER ON REAL ROBOT!"""

    def __init__(self):
        super().__init__('depthMapper')
        self.have_info = False
        self.step = 2 # how much to fractionally reduce the amount of pixels in the point cloud 1/step so 4 is 1/4 the pixel amount
        self.bridge = CvBridge()
        viz_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, '/camera/camera/depth/image_raw', self.depth_cb, qos_profile=qos_profile_sensor_data)


        self.depth_filter_pub = self.create_publisher(Image ,'/depth_filtered', qos_profile=qos_profile_sensor_data)
        # self.depth_filtered_image_pub = self.create_publisher(Image ,'/depth_filtered/viz', qos_profile=qos_profile_sensor_data)
        self.pub_cloud= self.create_publisher(PointCloud2,'/depth_filtered/points', qos_profile=viz_qos)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, qos_profile_sensor_data)

    def info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]
        self.have_info = True

    def depth_cb(self,msg:Image):
        if not self.have_info:
            return
        LAP_THRESHOLD = 10 #tune this later 
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Start by filtering out noise:
        #main sources: 
        # Invalid pixels
        # Outlyer Pixels
        # small noise fluctuations
        # Regions with no data

        #Mask NAN or 0 or -1 
        depth = depth.astype(np.float32) #raw depth in m (from sim at least)
        valid = (depth > 0) & (depth < 4.0) #mask + clip
        depth[~valid] = 0

        #Median Filter: take the median of the 5x5 to keep get rid of extreme noise  (spatial) 
        median_depth = cv2.medianBlur(depth.astype(np.float32), 5)

        #Bilateral filter: (spatial, edge perserving)
        #this one is more involved: explain when i get here
        filtered_depth = cv2.bilateralFilter(
            median_depth.astype(np.float32),
            d = 7, #neighborhood diameter in pixels
            sigmaColor=0.05, #in meters
            sigmaSpace=5 #in pixels

        )
        #pass the filtered depth for logic 
        filtered_depth_msg = self.bridge.cv2_to_imgmsg(filtered_depth, encoding='32FC1')
        filtered_depth_msg.header = msg.header
        self.depth_filter_pub.publish(filtered_depth_msg)

        #pass the filtered depth point cloud2 for visualizations
        # self.get_logger().info('publishing a point cloud')
        self.pub_cloud.publish(self.make_cloud(depth, msg.header))
        



    def make_cloud(self, depth, header):
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        s = self.step

        z = depth[::s, ::s]
        v, u = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
        valid = z > 0
        z = z[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        pts = np.column_stack((x, y, z)).astype(np.float32)
        return point_cloud2.create_cloud_xyz32(header, pts)

        #Temportal Filtering: (uses a median filter over the last N frames to keep "error memory")
        #reduces no data regions by using old data 
        #TODO does work well while moving so im going to worry about it later 


        # Hole filling: fill empty cells using neighboring values  
        #TODO


def main():
    rclpy.init()
    node = DepthMapper()
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
