#!/usr/bin/env python3

import os
import cv2 # type : ignore
import cv_bridge
import open3d as o3d
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class DepthMapper(Node):
    def __init__(self):
        super().__init__('depthMapper')
        self.bridge = CvBridge()
        self.point_cloud = o3d.geometry.PointCloud()

    def depth_cb(self,msg:Image):
        LAP_THRESHOLD = 10 #tune this later 
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Start by filtering out noise:
        #main sources: 
        # Invalid pixels
        # Outlyer Pixels
        # small noise fluctuations
        # Regions with no data

        #Mask NAN or 0 or -1 
        depth = depth.astype(np.float16) #raw depth in m (from sim at least)
        valid = (depth > 0) & (depth < 4.0) #mask + clip
        depth[~valid] = 0

        #Median Filter: take the median of the 5x5 to keep get rid of extreme noise  (spatial) 
        median_depth = cv2.medianBlur(depth, 5)

        #Bilateral filter: (spatial, edge perserving)
        #this one is more involved: explain when i get here

        #Temportal Filtering: (uses a median filter over the last N frames to keep "error memory")
        #reduces no data regions by using old data 
        # 
        # Hole filling: fill empty cells using neighboring values  












        # #clean the NAN or -inf from gazebo using a mask:
        # valid = np.isfinite(depth) & (depth > 0.1) & (depth < 4.0)
        # depth_clean = np.where(valid, depth, 0.0).astype(np.float32)
        # #this does a double deravative (laplacian) to remove the constant sloped background and get key feature changes
        # lap = cv2.Laplacian(depth_clean, cv2.CV_32F, ksize=3)
        
        # #create a non valid 8 bit mask to ignore those invalid cells that were replaced with 0:
        # not_valid = (~valid).astype(np.uint8) #~valid flips this boolean array. #uint8 so we can use the 0 and 1 from the bools.

        # # > 0 is a python trick to return a bool mask back so we can use it as a logic overlay. 
        # invalid_near = cv2.dilate(not_valid, np.ones((5, 5), np.uint8)) > 0 

        # # -- Next we create a mask: -- #
        # #Which pixels have a strong edge? (True/False image)
        # strong_edge = np.abs(lap) > self.lap_thresh

        # #Which pixels are safe to trust? (not near any bad-depth pixel)
        # safe = ~invalid_near

        # # Keep only strong edges that are also safe
        # good_edge = strong_edge & safe

        # # Convert True/False into 255/0 so OpenCV can treat it as an image
        # mask = good_edge.astype(np.uint8) * 255

        # pass