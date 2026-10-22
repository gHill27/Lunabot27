#!/usr/bin/env python3

import os
import cv2 # type : ignore
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class test_driver(Node):
    def __init__(self):
        super().__init__('driver')
        self.publsiher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.driveinCircle()

    def driveinCircle(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.2
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0

        # Set angular velocity (roll, pitch, yaw)
        twist_msg.angular.x = 0.1
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.3
        try:
            while True:
                self.publsiher.publish(twist_msg)
                time.sleep(0.1) # 10 Hz
        except KeyboardInterrupt:
            print("Stopped publishing.")
        pass

def main():
    rclpy.init()
    node = test_driver()
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
