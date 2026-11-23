#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry import LaserProjection

class ScanToCloud(Node):
    def __init__(self):
        super().__init__('scan_to_cloud')
        self.lp = LaserProjection()
        self.pub = self.create_publisher(PointCloud2, '/scan_cloud_in', 10)
        self.create_subscription(LaserScan, '/scan', self.cb, 10)

    def cb(self, scan):
        self.pub.publish(self.lp.projectLaser(scan))

def main():
    rclpy.init()
    rclpy.spin(ScanToCloud())

if __name__ == '__main__':
    main()